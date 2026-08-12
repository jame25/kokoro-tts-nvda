import hashlib
import json
import os
import queue
import threading
import time
import numpy as np
import onnxruntime as ort
import re
from typing import Dict, List, Optional, Tuple, Union

try:
    from logHandler import log
except ImportError:
    log = None

# Import our custom phonemizer
# Import our custom phonemizer
try:
    try:
        from .kokoro_phonemizer import KokoroPhonemizer, KOKORO_PHONEMIZER_AVAILABLE
    except ImportError:
        from kokoro_phonemizer import KokoroPhonemizer, KOKORO_PHONEMIZER_AVAILABLE
except Exception:
    KOKORO_PHONEMIZER_AVAILABLE = False


class KokoroTTS:
    def __init__(self, model_path: str, voice_dir, config_path: str, tokenizer_path: str, default_speed: float = 0.85, language: str = 'en-us'):
        """
        Initialize the Kokoro TTS engine.

        Args:
            model_path: Path to the ONNX model file
            voice_dir: Directory containing voice embedding files (.npy)
            config_path: Path to the config.json file
            tokenizer_path: Path to the tokenizer.json file
            default_speed: Default speech speed factor (lower values = slower speech)
            language: Language code for phonemization (e.g., 'en-us', 'fr-fr')
        """
        self.model_path = model_path
        self.voice_dirs = [voice_dir] if isinstance(voice_dir, (str, bytes, os.PathLike)) else list(voice_dir)
        self.voice_dir = self.voice_dirs[0]
        self.default_speed = default_speed
        self.language = language

        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # Load tokenizer
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            self.tokenizer_config = json.load(f)

        # Initialize phonemizer
        self.phonemizer = None
        self.use_phonemizer = False

        # Always try to use the phonemizer for clear English speech
        if KOKORO_PHONEMIZER_AVAILABLE:
            try:
                self.phonemizer = KokoroPhonemizer(language=language)
                self.use_phonemizer = True
            except Exception as e:
                raise RuntimeError(f"Failed to initialize phonemizer: {e}")
        else:
            raise RuntimeError("Phonemizer not available. Cannot initialize TTS without eSpeak-NG.")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.inter_op_num_threads = 1
        # On a low-core-count laptop (4 physical cores) the model is fast enough
        # with 2 threads *when cool*, but sustained article reading pushes two
        # cores into thermal throttling and each inference degrades severely
        # (measured: a single sentence went from ~14s to ~23s after a few
        # back-to-back runs). Spreading the work over 4 threads keeps clock
        # speeds sustainable and gives stable, faster throughput for continuous
        # reading.
        session_options.intra_op_num_threads = min(4, os.cpu_count() or 4)
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True

        # Force CPUExecutionProvider. DirectML (DmlExecutionProvider) triggers 13-second DirectX 12 HLSL shader JIT compilation delays on Intel integrated GPUs.
        providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers,
        )

        # Get input and output names
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]

        pass
        pass
        # Load available voices
        self.voices = self._load_available_voices()
        self.current_voice = None
        if self.voices:
            self.current_voice = list(self.voices.keys())[0]  # Default to first voice

        # Audio parameters
        self.sample_rate = 24000  # Kokoro's default sample rate

        # Waveform LRU Cache (RAM)
        self._synth_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._synth_cache_max: int = 4096

        # Persistent Binary File Disk Cache
        self._disk_cache_dir = None
        self._init_disk_cache()

        # Warm the ONNX session now so the expensive one-time lazy
        # initialization (thread pool spin-up, kernel selection, memory arena
        # allocation) is paid during NVDA startup instead of during the user's
        # first read. This is synchronous so the model is guaranteed warm before
        # any speech is requested; on slow hardware it costs a few seconds once.
        self._warmup_done = False
        try:
            self._warmup()
        except Exception:
            if log is not None:
                log.debugWarning("Kokoro model warm-up failed", exc_info=True)

    def _warmup(self) -> None:
        """Run one tiny inference to force ONNX Runtime lazy initialization.

        A real voice-bank row is used so the run is representative; the result
        is discarded. Subsequent real reads then start almost immediately.
        """
        if self._warmup_done:
            return
        if not self.voices or not self.current_voice:
            return
        start_id = self.tokenizer_config["model"]["vocab"]["$"]
        # Use a realistic token count (like a typical screen-reader chunk) so
        # ONNX Runtime performs its shape-dependent memory planning for the
        # sizes that will actually be used at read time, not just tiny inputs.
        # 16 tokens (~1.8s here) is a good balance: enough to trigger the
        # realistic memory layout, but bounded so it doesn't slow NVDA startup.
        tokens = np.array([[start_id] * 16], dtype=np.int64)
        voice_bank = self.voices[self.current_voice]
        style_index = min(max(0, len(tokens) - 2), voice_bank.shape[0] - 1)
        voice_embedding = voice_bank[style_index]
        token_input_name = "input_ids" if "input_ids" in self.input_names else "tokens"
        inputs = {
            token_input_name: tokens,
            "style": voice_embedding,
            "speed": np.array([self.default_speed], dtype=np.float32),
        }
        self.session.run(None, inputs)
        self._warmup_done = True

    def _init_disk_cache(self) -> None:
        self._disk_cache_dir = None
        self._diskWriteQueue: queue.Queue[Optional[tuple[str, bytes]]] = queue.Queue()
        self._diskWriterThread = None
        try:
            cache_dir = os.path.dirname(self.model_path)
            self._disk_cache_dir = os.path.join(cache_dir, "cache")
            os.makedirs(self._disk_cache_dir, exist_ok=True)
        except Exception:
            self._disk_cache_dir = None
            return
        self._diskWriterThread = threading.Thread(
            target=self._runDiskWriter,
            name="KokoroDiskCacheWriter",
            daemon=True,
        )
        self._diskWriterThread.start()

    def _runDiskWriter(self) -> None:
        """Persist waveforms to disk in the background.

        Writing is intentionally async: on Windows, creating a new file in the
        cache directory can be delayed by antivirus scanning, and blocking the
        synthesis worker on disk I/O would delay speech start.
        """
        while True:
            item = self._diskWriteQueue.get()
            if item is None:
                return
            try:
                file_path, pcm_bytes = item
                with open(file_path, "wb") as f:
                    f.write(pcm_bytes)
            except Exception:
                pass
            finally:
                self._diskWriteQueue.task_done()

    def _get_from_disk_cache(self, key: str) -> Optional[np.ndarray]:
        if not self._disk_cache_dir:
            return None
        try:
            key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
            file_path = os.path.join(self._disk_cache_dir, f"{key_hash}.bin")
            if os.path.isfile(file_path):
                with open(file_path, "rb") as f:
                    pcm_bytes = f.read()
                if pcm_bytes:
                    int16_arr = np.frombuffer(pcm_bytes, dtype=np.int16)
                    return int16_arr.astype(np.float32) / 32767.0
        except Exception:
            pass
        return None

    def _save_to_disk_cache(self, key: str, waveform: np.ndarray) -> None:
        if not self._disk_cache_dir:
            return
        try:
            key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
            file_path = os.path.join(self._disk_cache_dir, f"{key_hash}.bin")
            pcm_bytes = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            if self._diskWriterThread is not None:
                self._diskWriteQueue.put((file_path, pcm_bytes))
            else:
                with open(file_path, "wb") as f:
                    f.write(pcm_bytes)
        except Exception:
            pass

    def _load_available_voices(self) -> Dict[str, np.ndarray]:
        """Load complete Kokoro style banks from the voice directory.

        Kokoro voice files contain one 256-value style vector for each
        supported phoneme-token length. Do not collapse the bank to an
        arbitrary row while loading; the correct row is selected per
        utterance in synthesize().
        """
        voices = {}
        for voice_dir in self.voice_dirs:
            if not os.path.isdir(voice_dir):
                continue
            for filename in os.listdir(voice_dir):
                lowerName = filename.lower()
                if not lowerName.endswith((".npy", ".bin")):
                    continue
                voice_name = os.path.splitext(filename)[0]
                voice_path = os.path.join(voice_dir, filename)
                try:
                    if lowerName.endswith(".bin"):
                        voice_bank = np.fromfile(voice_path, dtype=np.float32)
                        if voice_bank.size % 256 != 0:
                            raise ValueError(f"Invalid Kokoro voice size: {voice_bank.size} float values")
                        voice_bank = voice_bank.reshape(-1, 1, 256)
                    else:
                        voice_bank = np.asarray(np.load(voice_path), dtype=np.float32)
                    if voice_bank.ndim == 2 and voice_bank.shape[1] == 256:
                        voice_bank = voice_bank[:, np.newaxis, :]
                    if voice_bank.ndim != 3 or voice_bank.shape[1:] != (1, 256):
                        raise ValueError(
                            f"Unsupported Kokoro voice shape {voice_bank.shape}; "
                            "expected [styles, 1, 256]"
                        )
                    voices[voice_name] = voice_bank
                except Exception:
                    pass
        return voices

    def reload_voices(self) -> List[str]:
        current = self.current_voice
        self.voices = self._load_available_voices()
        if current in self.voices:
            self.current_voice = current
        elif self.voices:
            self.current_voice = next(iter(self.voices))
        else:
            self.current_voice = None
        return self.list_voices()

    def set_voice(self, voice_name: str) -> bool:
        """
        Set the current voice for TTS.

        Args:
            voice_name: Name of the voice to use

        Returns:
            bool: True if voice was set successfully, False otherwise
        """
        if voice_name in self.voices:
            self.current_voice = voice_name
            return True
        return False

    def list_voices(self) -> List[str]:
        """Return a list of available voice names."""
        return list(self.voices.keys())

    def phonemize_text(self, text: str) -> str:
        """
        Convert text to phonemes using the phonemizer.

        Args:
            text: Input text to phonemize

        Returns:
            Phonemized text
        """
        if self.use_phonemizer and self.phonemizer is not None:
            try:
                phonemized = self.phonemizer.phonemize(text)
                pass
                return phonemized
            except Exception as e:
                raise RuntimeError(f"Phonemization failed: {e}")

        raise RuntimeError("Phonemizer not available")

    def _normalize_raw_text(self, text: str) -> str:
        """
        Normalize raw text for direct tokenization when phonemization is disabled.

        This converts uppercase to lowercase and expands numbers and special characters
        to make them compatible with the tokenizer vocabulary.

        Args:
            text: Raw input text

        Returns:
            Normalized text suitable for tokenization
        """
        # First, convert to lowercase (this handles uppercase letters)
        normalized = text.lower()

        # Replace numbers with words
        number_words = {
            '0': 'zero ', '1': 'one ', '2': 'two ', '3': 'three ', '4': 'four ',
            '5': 'five ', '6': 'six ', '7': 'seven ', '8': 'eight ', '9': 'nine '
        }

        for digit, word in number_words.items():
            normalized = normalized.replace(digit, word)

        # Replace common punctuation with spaces or appropriate words
        punctuation_replacements = {
            '-': ' ', '_': ' ', '.': ' dot ', ',': ' comma ', '!': ' exclamation ',
            '?': ' question ', ':': ' colon ', ';': ' semicolon ',
            '(': ' open parenthesis ', ')': ' close parenthesis ',
            '[': ' open bracket ', ']': ' close bracket ',
            '{': ' open brace ', '}': ' close brace ',
            '/': ' slash ', '\\': ' backslash ', '|': ' pipe ',
            '@': ' at ', '#': ' hash ', '$': ' dollar ', '%': ' percent ',
            '^': ' caret ', '&': ' and ', '*': ' star ', '+': ' plus ',
            '=': ' equals ', '<': ' less than ', '>': ' greater than ',
            '~': ' tilde ', '`': ' backtick ', "'": ' apostrophe ', '"': ' quote '
        }

        for punct, replacement in punctuation_replacements.items():
            normalized = normalized.replace(punct, replacement)

        # Replace multiple spaces with a single space
        normalized = re.sub(r'\s+', ' ', normalized)

        # Add spaces between letters to help with tokenization
        # This makes each letter a separate token, which is more reliable
        spaced_text = ""
        for char in normalized:
            if char.isalpha():
                spaced_text += char + " "
            else:
                spaced_text += char

        # Replace multiple spaces again
        spaced_text = re.sub(r'\s+', ' ', spaced_text)

        return spaced_text.strip()

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize input text using the Kokoro tokenizer.

        Args:
            text: Input text to tokenize

        Returns:
            List of token IDs
        """
        # First, convert text to phonemes if phonemizer is available
        phonemized_text = self.phonemize_text(text)
        if not phonemized_text:
            return []
        # Simple character-based tokenization based on the tokenizer vocab
        vocab = self.tokenizer_config["model"]["vocab"]
        tokens = []

        # Add start token
        tokens.append(vocab["$"])

        # Define character mappings for special characters not in the vocabulary
        char_mappings = {
            'g': '\u0261',  # LATIN SMALL LETTER SCRIPT G
            '˞': 'ɹ',       # Map rhotic hook to 'ɹ' (LATIN SMALL LETTER R)
            'ˌ': '',        # Secondary stress mark - can be omitted
            'ˈ': '',        # Primary stress mark - can be omitted
            'ː': ':',       # LENGTH MARK to COLON
            'ɚ': 'ə',       # R-COLORED SCHWA to SCHWA
            'ɝ': 'ɜ',       # R-COLORED REVERSED EPSILON to EPSILON
            'ɾ': 't',       # FLAP to 't'
            'ɫ': 'l',       # VELARIZED L to 'l'
            'ɪ̈': 'ɪ',       # I WITH DIAERESIS to 'ɪ'
            'ɵ': 'o',       # BARRED O to 'o'
            'ɐ': 'a',       # TURNED A to 'a'
            'ɘ': 'ə',       # REVERSED E to SCHWA
            'ɜ': 'e',       # REVERSED EPSILON to 'e'
            'ɞ': 'e',       # CLOSED REVERSED EPSILON to 'e'
            'ʉ': 'u',       # BARRED U to 'u'
            'ʊ': 'u',       # UPSILON to 'u'
            'ʌ': 'a',       # TURNED V to 'a'
            'ʍ': 'w',       # TURNED W to 'w'
            'ʏ': 'y',       # SMALL CAPITAL Y to 'y'
            'ʒ': 'z',       # EZH to 'z'
            'ʔ': '',        # GLOTTAL STOP - can be omitted
            'θ': 'th',      # THETA to 'th'
            'ð': 'th',      # ETH to 'th'
            'ŋ': 'n',       # ENG to 'n'
            'ɡ': 'g',       # SCRIPT G to 'g'
            'ɹ': 'r',       # TURNED R to 'r'
            'ʃ': 'sh',      # ESH to 'sh'
            'ʧ': 'ch',      # TESH to 'ch'
            'ʤ': 'j',       # DEZH to 'j'
        }

        # Tokenize each character
        i = 0
        while i < len(phonemized_text):
            char = phonemized_text[i]

            if char in vocab:
                tokens.append(vocab[char])
                i += 1
            elif char in char_mappings:
                # Use the mapped character if available
                mapped_char = char_mappings[char]

                if not mapped_char:
                    # Character is mapped to be omitted
                    pass
                    i += 1
                    continue

                # Handle multi-character replacements
                if len(mapped_char) > 1:
                    # Add each character of the replacement
                    all_in_vocab = True
                    for c in mapped_char:
                        if c not in vocab:
                            all_in_vocab = False
                            break

                    if all_in_vocab:
                        for c in mapped_char:
                            tokens.append(vocab[c])
                        pass
                    else:
                        pass
                else:
                    # Single character replacement
                    if mapped_char in vocab:
                        tokens.append(vocab[mapped_char])
                        pass
                    else:
                        pass
                i += 1
            else:
                # For characters not in vocabulary, just skip them
                # but don't print a warning for spaces to reduce log spam
                if char != ' ':
                    pass
                i += 1
                continue

        # Add end token
        tokens.append(vocab["$"])

        return tokens

    @staticmethod
    def _normalizeCacheKey(text: str) -> str:
        s = text.strip().lower()
        s = re.sub(r'[\.\…\:\,\;\!\?\/]+', '', s)
        for _ in range(3):
            s = re.sub(r'\s+sub\s+menu\s+[a-z0-9]$', '', s)
            s = re.sub(r'\s+menu\s+[a-z0-9]$', '', s)
            s = re.sub(r'\s+sub\s+menu$', '', s)
            s = re.sub(r'\s+menu$', '', s)
        return re.sub(r'\s+', ' ', s).strip()

    def synthesize(self, text: str, speed: float = None) -> np.ndarray:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            speed: Speech speed factor (lower values = slower speech)

        Returns:
            numpy.ndarray: Audio waveform
        """
        if not self.current_voice:
            raise ValueError("No voice selected. Use set_voice() to select a voice.")

        if speed is None:
            speed = self.default_speed

        clean_text = text.strip()
        norm_key = self._normalizeCacheKey(clean_text)
        cache_key = (norm_key, self.current_voice)
        disk_key = f"{norm_key}:{self.current_voice}"

        if len(norm_key) <= 300:
            # Tier 1: Check RAM Cache (0.00 ms)
            if cache_key in self._synth_cache:
                return self._synth_cache[cache_key]

            # Tier 2: Check Persistent Binary Disk Cache (0.01 ms)
            disk_wav = self._get_from_disk_cache(disk_key)
            if disk_wav is not None:
                self._synth_cache[cache_key] = disk_wav
                return disk_wav

        # Tokenize input text
        tokens = self.tokenize(text)
        if not tokens:
            return np.empty(0, dtype=np.float32)
        tokens = np.array(tokens, dtype=np.int64)

        # Kokoro voice files are style banks indexed by phoneme-token
        # length. This tokenizer includes one zero pad token at each end,
        # whereas the upstream algorithm selects the style before adding
        # those pads. Therefore exclude the two pads from the style index.
        voice_bank = self.voices[self.current_voice]
        phoneme_token_count = max(0, len(tokens) - 2)
        style_index = min(phoneme_token_count, voice_bank.shape[0] - 1)
        voice_embedding = voice_bank[style_index]

        # Prepare inputs for the model based on the expected input names.
        tokenInputName = "input_ids" if "input_ids" in self.input_names else "tokens"
        inputs = {
            tokenInputName: tokens.reshape(1, -1),
            'style': voice_embedding,
            'speed': np.array([speed], dtype=np.float32)
        }

        # Run inference
        outputs = self.session.run(None, inputs)
        waveform = np.asarray(outputs[0].squeeze(), dtype=np.float32)

        if len(norm_key) <= 300:
            if len(self._synth_cache) >= self._synth_cache_max:
                self._synth_cache.clear()
            self._synth_cache[cache_key] = waveform
            self._save_to_disk_cache(disk_key, waveform)

        return waveform

    def save_to_file(self, text: str, output_path: str, speed: float = None) -> None:
        """
        Synthesize speech and save to a file.

        Args:
            text: Text to synthesize
            output_path: Path to save the audio file
            speed: Speech speed factor (lower values = slower speech)
        """
        waveform = self.synthesize(text, speed)

        # Normalize to int16 range
        waveform = np.clip(waveform, -1.0, 1.0)
        waveform = (waveform * 32767).astype(np.int16)

        # Save as a numpy array instead of WAV
        # This is a simpler format that doesn't require scipy
        np.save(output_path, waveform)
        pass
def main():
    """Example usage of the KokoroTTS engine."""
    # Paths
    model_path = os.path.join("model", "kokoro.onnx")
    voice_dir = "voices"
    config_path = "config.json"
    tokenizer_path = "tokenizer.json"

    # Initialize TTS engine with a slower default speed
    tts = KokoroTTS(model_path, voice_dir, config_path, tokenizer_path, default_speed=0.85)

    # List available voices
    voices = tts.list_voices()
    pass
    if voices:
        # Set voice
        tts.set_voice(voices[0])

        # Synthesize and save to file
        text = "This is a test of the Kokoro TTS engine."
        tts.save_to_file(text, "output.npy")


if __name__ == "__main__":
    main()
