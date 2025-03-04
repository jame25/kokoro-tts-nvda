import re
import unicodedata
from typing import List, Dict, Optional, Union
import os
import subprocess
import tempfile

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

class KokoroPhoneimzer:
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
        
        # Check if eSpeak-NG is available
        if not self.espeak_path:
            print("Warning: eSpeak-NG not found. Phonemization will not be available.")
            global KOKORO_PHONEMIZER_AVAILABLE
            KOKORO_PHONEMIZER_AVAILABLE = False
            print("Using fallback basic phonemization")
    
    def _find_espeak_path(self) -> Optional[str]:
        """Find the path to the eSpeak-NG executable."""
        # Common paths for eSpeak-NG
        common_paths = [
            r"C:\Program Files\eSpeak NG\espeak-ng.exe",
            r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
            "/usr/bin/espeak-ng",
            "/usr/local/bin/espeak-ng"
        ]
        
        # Check if espeak-ng is in PATH
        try:
            # Use 'where' on Windows, 'which' on Unix
            if os.name == 'nt':  # Windows
                result = subprocess.run(
                    ["where", "espeak-ng"], 
                    capture_output=True, 
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:  # Unix
                result = subprocess.run(
                    ["which", "espeak-ng"], 
                    capture_output=True, 
                    text=True
                )
            
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except Exception as e:
            print(f"Error checking for espeak-ng in PATH: {e}")
        
        # Check common paths
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        return None
    
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
        Convert text to phonemes using eSpeak-NG.
        
        Args:
            text: Input text to phonemize
            
        Returns:
            Phonemized text
        """
        if not self.espeak_path:
            print("Error during phonemization: eSpeak-NG not found")
            return self._basic_phonemize(text)
        
        try:
            # Create a temporary file for the input text
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as f:
                f.write(text)
                temp_filename = f.name
            
            # Run eSpeak-NG with phoneme output
            cmd = [
                self.espeak_path,
                "--ipa",
                "-v", self.language,
                "-q",  # Quiet mode
                "-f", temp_filename
            ]
            
            # Use CREATE_NO_WINDOW flag on Windows to prevent console window from appearing
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            
            # Run the process with a timeout
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',  # Explicitly set encoding to utf-8
                errors='replace',   # Replace invalid characters
                creationflags=creation_flags
            )
            
            # Use communicate with timeout to prevent hanging
            stdout, stderr = process.communicate(timeout=2.0)
            
            # Clean up the temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass
            
            if process.returncode != 0:
                print(f"Error during phonemization: {stderr}")
                return self._basic_phonemize(text)
            
            # Process the output
            phonemes = stdout.strip() if stdout else ""
            
            if not phonemes:
                print("No output from phonemizer, using fallback")
                return self._basic_phonemize(text)
            
            # Clean up the phonemes
            phonemes = self._clean_phonemes(phonemes)
            
            return phonemes
            
        except subprocess.TimeoutExpired:
            print("Error during phonemization: Process timed out")
            # Kill the process if it's still running
            try:
                process.kill()
            except:
                pass
            return self._basic_phonemize(text)
        except Exception as e:
            print(f"Error during phonemization: {e}")
            return self._basic_phonemize(text)
    
    def _clean_phonemes(self, phonemes: str) -> str:
        """Clean up the phonemes output from eSpeak-NG."""
        # Remove extra spaces
        phonemes = re.sub(r'\s+', ' ', phonemes)
        
        # Remove some symbols that might cause issues
        phonemes = phonemes.replace('(en)', '')
        
        return phonemes.strip()
    
    def _basic_phonemize(self, text: str) -> str:
        """
        Basic phonemization for English when eSpeak-NG is not available.
        This is a very simplified version and won't be as accurate.
        
        Args:
            text: Input text to phonemize
            
        Returns:
            Basic phonemized text
        """
        print("Falling back to basic phonemization")
        
        # Define a simple mapping for common English sounds
        # This is very basic and won't handle many cases correctly
        phoneme_map = {
            'a': 'æ',
            'e': 'ɛ',
            'i': 'ɪ',
            'o': 'ɑ',
            'u': 'ʌ',
            'th': 'θ',
            'sh': 'ʃ',
            'ch': 'tʃ',
            'j': 'dʒ',
            'ng': 'ŋ',
            'ou': 'aʊ',
            'ow': 'aʊ',
            'oi': 'ɔɪ',
            'ay': 'eɪ',
            'ai': 'eɪ',
            'ee': 'i',
            'ea': 'i',
            'oo': 'u',
            'ar': 'ɑr',
            'er': 'ɜr',
            'ir': 'ɪr',
            'or': 'ɔr',
            'ur': 'ʊr',
        }
        
        # Convert text to lowercase for processing
        text = text.lower()
        
        # Replace digraphs first (two-letter combinations)
        for digraph, phoneme in phoneme_map.items():
            if len(digraph) > 1:
                text = text.replace(digraph, phoneme)
        
        # Then replace single letters
        result = ""
        for char in text:
            if char in phoneme_map and len(char) == 1:
                result += phoneme_map[char]
            else:
                result += char
        
        return result


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
