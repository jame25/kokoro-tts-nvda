import json
import os
import time
import numpy as np
import onnxruntime as ort
import re
from typing import Dict, List, Optional, Tuple, Union

# Import our custom phonemizer
try:
    from kokoro_phonemizer import KokoroPhoneimzer, KOKORO_PHONEMIZER_AVAILABLE
except ImportError:
    KOKORO_PHONEMIZER_AVAILABLE = False
    print("Warning: kokoro_phonemizer module not found. Using fallback character-based tokenization.")

class KokoroTTS:
    def __init__(self, model_path: str, voice_dir: str, config_path: str, tokenizer_path: str, default_speed: float = 0.85, language: str = 'en-us'):
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
        self.voice_dir = voice_dir
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
                self.phonemizer = KokoroPhoneimzer(language=language)
                self.use_phonemizer = True
                print(f"Using custom phonemizer with language: {language}")
            except Exception as e:
                print(f"Error initializing phonemizer: {e}")
                print("Falling back to character-based tokenization")
        else:
            print("Phonemizer not available. Using fallback character-based tokenization.")
        
        # Initialize ONNX runtime session with CPU provider
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )
        
        # Get input and output names
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"Model input names: {self.input_names}")
        print(f"Model output names: {self.output_names}")
        
        # Load available voices
        self.voices = self._load_available_voices()
        self.current_voice = None
        if self.voices:
            self.current_voice = list(self.voices.keys())[0]  # Default to first voice
        
        # Audio parameters
        self.sample_rate = 24000  # Kokoro's default sample rate
    
    def _load_available_voices(self) -> Dict[str, np.ndarray]:
        """Load all available voice embeddings from the voice directory."""
        voices = {}
        for filename in os.listdir(self.voice_dir):
            if filename.endswith('.npy'):
                voice_name = os.path.splitext(filename)[0]
                voice_path = os.path.join(self.voice_dir, filename)
                try:
                    # Load the voice embedding
                    voice_embedding = np.load(voice_path)
                    print(f"Original voice {voice_name} shape: {voice_embedding.shape}")
                    
                    # The model expects a 2D tensor with shape [1, 256]
                    # If the embedding is 3D with shape [510, 1, 256], we need to extract a single embedding
                    if len(voice_embedding.shape) == 3 and voice_embedding.shape[1] == 1 and voice_embedding.shape[2] == 256:
                        # For female voices, use embedding index 100 which produces good results
                        # NOTE: After extensive testing, we found that using embedding index 100 for female voices
                        # produces much cleaner audio without distortion or volume inconsistencies.
                        # The first embedding (index 0) tends to produce mechanical-sounding distortion.
                        if voice_name.startswith('af_'):
                            embedding_index = min(100, voice_embedding.shape[0] - 1)
                            voice_embedding = voice_embedding[embedding_index]
                            print(f"Using embedding at index {embedding_index} for female voice {voice_name}")
                        else:
                            # For male voices, use the first embedding
                            voice_embedding = voice_embedding[0]
                    
                    # If the embedding is 2D but not [1, 256]
                    if len(voice_embedding.shape) == 2:
                        if voice_embedding.shape[0] != 1:
                            # Take the first row if multiple rows
                            voice_embedding = voice_embedding[0:1]
                        
                        # Ensure the second dimension is 256
                        if voice_embedding.shape[1] != 256:
                            # Reshape or pad as needed
                            if voice_embedding.shape[1] > 256:
                                voice_embedding = voice_embedding[:, :256]
                            else:
                                # Pad with zeros
                                padded = np.zeros((1, 256))
                                padded[:, :voice_embedding.shape[1]] = voice_embedding
                                voice_embedding = padded
                    
                    # If the embedding is 1D, reshape to [1, 256]
                    if len(voice_embedding.shape) == 1:
                        if voice_embedding.size >= 256:
                            voice_embedding = voice_embedding[:256].reshape(1, 256)
                        else:
                            # Pad with zeros
                            padded = np.zeros((1, 256))
                            padded[0, :voice_embedding.size] = voice_embedding
                            voice_embedding = padded
                    
                    # Final check to ensure shape is [1, 256]
                    if voice_embedding.shape != (1, 256):
                        print(f"Warning: Reshaping voice {voice_name} from {voice_embedding.shape} to (1, 256)")
                        # Try to reshape or create a new array
                        if voice_embedding.size >= 256:
                            voice_embedding = voice_embedding.flatten()[:256].reshape(1, 256)
                        else:
                            # Not enough data, pad with zeros
                            padded = np.zeros((1, 256))
                            padded[0, :min(voice_embedding.size, 256)] = voice_embedding.flatten()[:min(voice_embedding.size, 256)]
                            voice_embedding = padded
                    
                    voices[voice_name] = voice_embedding
                    print(f"Final voice {voice_name} shape: {voice_embedding.shape}")
                except Exception as e:
                    print(f"Error loading voice {voice_name}: {e}")
        return voices
    
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
                print(f"Successfully phonemized text: {text[:50]}{'...' if len(text) > 50 else ''}")
                return phonemized
            except Exception as e:
                print(f"Error during phonemization: {e}")
                # Fall back to the original text if phonemization fails
                return text
        
        # Fallback to the original text if phonemizer is not available
        print("Phonemizer not available, using original text")
        return text
    
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
        print(f"Text for tokenization: {phonemized_text[:50]}{'...' if len(phonemized_text) > 50 else ''}")
        
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
                    print(f"Omitting character: '{char}'")
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
                        print(f"Replacing '{char}' with '{mapped_char}'")
                    else:
                        print(f"Warning: Not all characters in '{mapped_char}' are in vocabulary, skipping '{char}'")
                else:
                    # Single character replacement
                    if mapped_char in vocab:
                        tokens.append(vocab[mapped_char])
                        print(f"Replacing '{char}' with '{mapped_char}'")
                    else:
                        print(f"Warning: Mapped character '{mapped_char}' for '{char}' not in vocabulary, skipping.")
                
                i += 1
            else:
                # For characters not in vocabulary, just skip them
                # but don't print a warning for spaces to reduce log spam
                if char != ' ':
                    print(f"Warning: Character '{char}' not in vocabulary, skipping.")
                i += 1
                continue
        
        # Add end token
        tokens.append(vocab["$"])
        
        return tokens
    
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
        
        # Use default speed if not specified
        if speed is None:
            speed = self.default_speed
        
        # Tokenize input text
        tokens = self.tokenize(text)
        tokens = np.array(tokens, dtype=np.int64)
        
        # Get voice embedding
        voice_embedding = self.voices[self.current_voice]
        
        # Prepare inputs for the model based on the expected input names
        inputs = {
            'tokens': tokens.reshape(1, -1),
            'style': voice_embedding,
            'speed': np.array([speed], dtype=np.float32)
        }
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run(self.output_names, inputs)
        inference_time = time.time() - start_time
        
        # Process output (assuming the first output is the waveform)
        waveform = outputs[0].squeeze()
        
        # Only apply basic normalization to prevent clipping
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform)) * 0.9
        
        print(f"Synthesized {len(text)} characters in {inference_time:.2f} seconds (speed={speed:.2f})")
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
        print(f"Saved audio data to {output_path}")


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
    print(f"Available voices: {', '.join(voices)}")
    
    if voices:
        # Set voice
        tts.set_voice(voices[0])
        
        # Synthesize and save to file
        text = "This is a test of the Kokoro TTS engine."
        tts.save_to_file(text, "output.npy")


if __name__ == "__main__":
    main() 