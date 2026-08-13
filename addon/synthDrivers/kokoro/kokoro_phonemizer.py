import re
import unicodedata
from typing import List, Dict, Optional, Union
import os
import subprocess
import tempfile
import ctypes
import threading

# Flag to indicate if the phonemizer is available
KOKORO_PHONEMIZER_AVAILABLE = True

# Regex patterns for text normalization
CURRENCY_RE = re.compile(r'([€$£¥])([0-9,]*[0-9]+)')
NUMBER_RE = re.compile(r'([0-9]+),([0-9]+)')
DECIMAL_RE = re.compile(r'([0-9]+)\.([0-9]+)')
WHITESPACE_RE = re.compile(r'\s+')
ABBREVIATIONS_RE = re.compile(r'\b[A-Z]\.+(?:[A-Z]\.+)+\b')
URL_RE = re.compile(r'(https?://|www\.)[A-Za-z0-9\-\.]+\.[a-zA-Z]{2,}(?:/[A-Za-z0-9\-\._~:/?#\[\]@!$&\'\(\)\*\+,;=]*)?')
EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

# Phoneme mapping for post-processing
PHONEME_MAPPING = {
    'ɚ': 'ə˞',
    'ɝ': 'ɜ˞',
    'ɫ': 'l',
    'ɬ': 'l',
    'ɧ': 'ʃ',
    'ɮ': 'l',
    'ɱ': 'm',
    'ɺ': 'ɾ',
    'ʍ': 'w',
    'ʑ': 'ʒ',
    'ʡ': 'ʔ',
    'ʢ': 'ʔ',
    'ʘ': 'p',
    'ǀ': 't',
    'ǁ': 'k',
    'ǂ': 'k',
    'ǃ': 'k',
    'ʄ': 'ɟ',
    'ʛ': 'ɡ',
    'ɓ': 'b',
    'ɗ': 'd',
    'ɠ': 'ɡ',
    'ʛ': 'ɡ',
    'ʜ': 'h',
    'ʢ': 'h',
    'ʡ': 'ʔ',
    'ɕ': 'ʃ',
    'ʭ': 'ʔ',
    'ʩ': 'n',
    'ʪ': 'l',
    'ʫ': 'l',
    'ᵻ': 'ɨ',
    'ᵿ': 'ʉ',
    'ɘ': 'ə',
    'ɞ': 'ə',
    'ʚ': 'ə',
    'ɞ': 'ə',
    'ɜ': 'ə',
    'ɐ': 'ə',
    'ɵ': 'ə',
    'ɶ': 'æ',
    'ʉ': 'u',
    'ɨ': 'i',
    'ɪ̈': 'ɪ',
    'ʏ̈': 'ʏ',
    'ɤ': 'o',
    'ɭ': 'l',
    'ɽ': 'ɾ',
    'ɳ': 'n',
    'ɲ': 'n',
    'ʈ': 't',
    'ɖ': 'd',
    'ʂ': 'ʃ',
    'ʐ': 'ʒ',
    'ɦ': 'h',
    'ɹ': 'ɹ',
    'ʋ': 'v',
    'ʙ': 'b',
    'ʀ': 'ɹ',
    'ⱱ': 'v',
    'ɡ': 'g',
    'ɑ̃': 'ɑ',
    'ɛ̃': 'ɛ',
    'ɔ̃': 'ɔ',
    'œ̃': 'œ',
    'ɒ̃': 'ɒ',
    'ʌ̃': 'ʌ',
    'ɪ̃': 'ɪ',
    'ʊ̃': 'ʊ',
    'ə̃': 'ə',
    'ɯ̃': 'ɯ',
    'ɨ̃': 'ɨ',
    'ʉ̃': 'ʉ',
    'ɘ̃': 'ɘ',
    'ɵ̃': 'ɵ',
    'ɤ̃': 'ɤ',
    'ɞ̃': 'ɞ',
    'ʏ̃': 'ʏ',
    'ø̃': 'ø',
    'ɶ̃': 'ɶ',
    'ɐ̃': 'ɐ',
    'ɜ̃': 'ɜ',
    'ɚ̃': 'ɚ',
    'ɝ̃': 'ɝ',
    'ɑ̃': 'ɑ',
    'ɔ̃': 'ɔ',
    'ɛ̃': 'ɛ',
    'ɒ̃': 'ɒ',
    'ɨ̃': 'ɨ',
    'ʉ̃': 'ʉ',
    'ɯ̃': 'ɯ',
    'ɪ̃': 'ɪ',
    'ʊ̃': 'ʊ',
    'ɘ̃': 'ɘ',
    'ɵ̃': 'ɵ',
    'ɤ̃': 'ɤ',
    'ɞ̃': 'ɞ',
    'ʏ̃': 'ʏ',
    'ø̃': 'ø',
    'ɶ̃': 'ɶ',
    'ɐ̃': 'ɐ',
    'ɜ̃': 'ɜ',
    'ɚ̃': 'ɚ',
    'ɝ̃': 'ɝ',
}

class KokoroPhonemizer:
    """
    A phonemizer for Kokoro TTS that uses eSpeak-NG for phonemization.
    """

    def __init__(self, language: str = 'en-us'):
        """
        Initialize the phonemizer.

        Args:
            language: Language code for phonemization (e.g., 'en-us', 'fr-fr')
        """
        self.language = language

        # Always try to find eSpeak-NG path
        self.espeak_path = self._find_espeak_path()

        # Prefer the bundled eSpeak-NG DLL. Calling the library directly keeps
        # the phonemizer initialized between utterances and avoids creating a
        # process and temporary file for every short screen-reader message.
        self._dll = None
        self._dllDirectoryHandle = None
        self._dllLock = threading.Lock()
        if self.espeak_path:
            try:
                self._initialize_espeak_dll()
            except Exception:
                # The command-line executable remains a safe compatibility
                # fallback if a future eSpeak build changes its exported API.
                self._dll = None

        if not self.espeak_path and self._dll is None:
            global KOKORO_PHONEMIZER_AVAILABLE
            KOKORO_PHONEMIZER_AVAILABLE = False

    def _find_espeak_path(self) -> Optional[str]:
        """Find the path to the bundled eSpeak-NG executable."""
        # Get the directory of the current script (addon directory)
        addon_dir = os.path.dirname(os.path.abspath(__file__))

        # Path to bundled espeak-ng.exe
        bundled_espeak = os.path.join(addon_dir, "espeak", "espeak-ng.exe")

        # Only use the bundled version
        if os.path.exists(bundled_espeak):
            pass
            return bundled_espeak

        pass
        return None

    def _initialize_espeak_dll(self) -> None:
        """Initialize bundled libespeak-ng once for the lifetime of the synth."""
        if os.name != "nt":
            return
        addon_dir = os.path.dirname(os.path.abspath(__file__))
        espeak_dir = os.path.join(addon_dir, "espeak")
        dll_path = os.path.join(espeak_dir, "libespeak-ng.dll")
        if not os.path.isfile(dll_path):
            return

        # Keep this handle alive so dependent DLL lookup continues to include
        # the bundled eSpeak directory for the lifetime of the driver.
        if hasattr(os, "add_dll_directory"):
            self._dllDirectoryHandle = os.add_dll_directory(espeak_dir)
        dll = ctypes.CDLL(dll_path)
        dll.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        dll.espeak_Initialize.restype = ctypes.c_int
        dll.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        dll.espeak_SetVoiceByName.restype = ctypes.c_int
        dll.espeak_TextToPhonemes.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
        ]
        dll.espeak_TextToPhonemes.restype = ctypes.c_void_p

        sample_rate = dll.espeak_Initialize(2, 0, espeak_dir.encode("utf-8"), 0)
        if sample_rate <= 0:
            espeak_data_dir = os.path.join(espeak_dir, "espeak-ng-data")
            sample_rate = dll.espeak_Initialize(2, 0, espeak_data_dir.encode("utf-8"), 0)
        if sample_rate <= 0:
            sample_rate = dll.espeak_Initialize(2, 0, None, 0)
        if sample_rate <= 0:
            raise RuntimeError("libespeak-ng initialization failed")

        voice_set = False
        for vname in (self.language, self.language.replace("-", "_"), self.language.partition("-")[0], "en-us", "en", "default"):
            try:
                if dll.espeak_SetVoiceByName(vname.encode("utf-8")) == 0:
                    voice_set = True
                    break
            except Exception:
                pass
        if not voice_set:
            try:
                dll.espeak_SetVoiceByName(b"en")
            except Exception:
                pass
        self._dll = dll

    def _phonemize_with_dll(self, text: str) -> str:
        """Return IPA phonemes using the persistent bundled eSpeak library."""
        encoded = text.encode("utf-8") + b"\0"
        buffer = ctypes.create_string_buffer(encoded)
        cursor = ctypes.c_void_p(ctypes.addressof(buffer))
        chunks = []
        # espeak_TextToPhonemes returns one clause at a time and advances the
        # supplied text pointer. Bound the loop defensively against a bad DLL.
        with self._dllLock:
            for _ in range(1024):
                result = self._dll.espeak_TextToPhonemes(
                    ctypes.byref(cursor),
                    1,  # espeakCHARS_UTF8
                    2,  # IPA output
                )
                if result:
                    chunk = ctypes.string_at(result).decode("utf-8", errors="replace")
                    if chunk:
                        chunks.append(chunk)
                if not cursor.value or ctypes.string_at(cursor.value, 1) == b"\0":
                    break
            else:
                raise RuntimeError("libespeak-ng phonemizer did not consume its input")
        return " ".join(chunks).replace("_", " ").strip()

    def normalize_text(self, text: str) -> str:
        """
        Normalize text before phonemization.

        Args:
            text: Input text to normalize

        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()

        # Replace URLs with "URL"
        text = URL_RE.sub(' URL ', text)

        # Replace emails with "EMAIL"
        text = EMAIL_RE.sub(' EMAIL ', text)

        # Replace abbreviations
        text = ABBREVIATIONS_RE.sub(lambda m: ' '.join(m.group(0).replace('.', ' ')), text)

        # Normalize currency
        text = CURRENCY_RE.sub(r'\1 \2', text)

        # Normalize numbers with commas
        text = NUMBER_RE.sub(r'\1.\2', text)

        # Normalize decimal numbers
        text = DECIMAL_RE.sub(r'\1 point \2', text)

        # Normalize whitespace
        text = WHITESPACE_RE.sub(' ', text)

        # Strip whitespace
        text = text.strip()

        return text

    def _post_process_phonemes(self, phonemes: str) -> str:
        """
        Post-process phonemes to match the expected format.

        Args:
            phonemes: Raw phonemes from the phonemizer

        Returns:
            Processed phonemes
        """
        # Replace phonemes according to the mapping
        for old, new in PHONEME_MAPPING.items():
            phonemes = phonemes.replace(old, new)

        # Remove stress markers (numbers)
        phonemes = re.sub(r'[0-9]', '', phonemes)

        # Remove extra spaces
        phonemes = re.sub(r'\s+', ' ', phonemes)

        return phonemes.strip()

    def phonemize(self, text: str) -> str:
        """
        Convert text to phonemes using bundled eSpeak-NG.

        Args:
            text: Input text to phonemize

        Returns:
            Phonemized text
        """
        if not hasattr(self, "_phonemize_cache"):
            self._phonemize_cache = {}
            self._phonemize_cache_max = 4096

        cache_key = text
        if cache_key in self._phonemize_cache:
            return self._phonemize_cache[cache_key]

        try:
            norm_text = self.normalize_text(text)
            norm_text = ''.join(ch for ch in norm_text if ch >= ' ' or ch in '\t\n')
            if not norm_text:
                return ""

            if self._dll is not None:
                raw_phonemes = self._phonemize_with_dll(norm_text)
                res = self._clean_phonemes(raw_phonemes) if raw_phonemes else ""
            else:
                res = norm_text

            if len(self._phonemize_cache) >= self._phonemize_cache_max:
                self._phonemize_cache.clear()
            self._phonemize_cache[cache_key] = res
            return res
        except Exception:
            return text

    def _clean_phonemes(self, phonemes: str) -> str:
        """Clean up the phonemes output from eSpeak-NG."""
        # Remove extra spaces
        phonemes = re.sub(r'\s+', ' ', phonemes)

        # Remove some symbols that might cause issues
        phonemes = phonemes.replace('(en)', '')

        return phonemes.strip()



# For testing
if __name__ == "__main__":
    phonemizer = KokoroPhoneimzer(language='en-us')

    test_texts = [
        "Hello, this is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Numbers like 123.45 and $67,890 should be normalized.",
        "URLs like https://example.com and emails like user@example.com should be replaced.",
        "Abbreviations like U.S.A. and Ph.D. should be expanded."
    ]

    for text in test_texts:
        print(f"Original: {text}")
        normalized = phonemizer.normalize_text(text)
        print(f"Normalized: {normalized}")
        phonemes = phonemizer.phonemize(text)
        print(f"Phonemes: {phonemes}")
        print()
