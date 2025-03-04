import os
import sys
# Fix queue import for Python 2/3 compatibility
try:
    import queue
except ImportError:
    # Python 2 compatibility
    import Queue as queue
import threading
import time
import numpy as np
from typing import Optional, Dict, Any

# Add the current directory to the path so we can import our TTS module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the Kokoro TTS engine
from kokoro_tts import KokoroTTS

# Import NVDA specific modules
import addonHandler
import globalPluginHandler
import speech
import synthDriverHandler
import config
import ui
import gui
import wx
import nvwave
from logHandler import log

# Initialize the addon
addonHandler.initTranslation()

class SynthDriver(synthDriverHandler.SynthDriver):
    """
    NVDA Synthesizer driver for Kokoro TTS.
    """
    name = "kokoro"
    description = "Kokoro TTS"
    
    # Supported settings
    supportedSettings = (
        synthDriverHandler.SynthDriver.VoiceSetting(),
        synthDriverHandler.SynthDriver.RateSetting(),
        synthDriverHandler.SynthDriver.VolumeSetting(),
    )
    
    # Supported commands - updated for compatibility with current NVDA versions
    supportedCommands = {
        speech.commands.IndexCommand,
        speech.commands.CharacterModeCommand,
        speech.commands.LangChangeCommand,
        speech.commands.BreakCommand,
        speech.commands.PitchCommand,
        speech.commands.RateCommand,
        speech.commands.VolumeCommand,
    }
    
    # Add the availableVoices property
    @property
    def availableVoices(self):
        """Return a dictionary of available voices."""
        voices = {}
        for voice in self.tts.list_voices():
            voices[voice] = synthDriverHandler.VoiceInfo(voice, voice)
        return voices
    
    def __init__(self):
        """Initialize the Kokoro TTS driver."""
        super(SynthDriver, self).__init__()
        
        # Paths to model files
        self.model_path = os.path.join(current_dir, "model", "kokoro.onnx")
        self.voice_dir = os.path.join(current_dir, "voices")
        self.config_path = os.path.join(current_dir, "config.json")
        self.tokenizer_path = os.path.join(current_dir, "tokenizer.json")
        
        # Initialize the TTS engine
        try:
            # We want phonemization to work for clear English speech
            # Make sure the environment variable is NOT set to disable phonemization
            if "KOKORO_DISABLE_ESPEAK" in os.environ:
                del os.environ["KOKORO_DISABLE_ESPEAK"]
            
            log.info("Enabling phonemization for clear English speech")
            
            self.tts = KokoroTTS(
                self.model_path,
                self.voice_dir,
                self.config_path,
                self.tokenizer_path
            )
            log.info(f"Kokoro TTS initialized with {len(self.tts.list_voices())} voices")
        except Exception as e:
            log.error(f"Failed to initialize Kokoro TTS: {e}")
            raise
        
        # Set default parameters
        self._rate = 50  # Default rate (0-100)
        self._volume = 100  # Default volume (0-100)
        self._voice = None
        
        # Initialize NVDA's audio system
        try:
            # Get the output device from NVDA's configuration
            # In NVDA 2025.1+, the audio output device is in config.conf["audio"]["outputDevice"]
            # In older versions, it's in config.conf["speech"]["outputDevice"]
            # We'll try both to ensure compatibility
            outputDevice = None
            try:
                if hasattr(config.conf, "get") and "audio" in config.conf:
                    outputDevice = config.conf["audio"].get("outputDevice")
                if not outputDevice and "speech" in config.conf:
                    outputDevice = config.conf["speech"].get("outputDevice")
            except:
                log.warning("Could not get output device from NVDA config, using default")
            
            # Initialize the wave player with the sample rate from our TTS engine
            self._player = nvwave.WavePlayer(
                channels=1,
                samplesPerSec=self.tts.sample_rate,
                bitsPerSample=16,
                outputDevice=outputDevice
            )
            log.info(f"NVDA WavePlayer initialized with sample rate {self.tts.sample_rate}")
        except Exception as e:
            log.error(f"Failed to initialize NVDA WavePlayer: {e}")
            raise
        
        # Set up the speech queue and processing thread
        self._speech_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._processing_thread = threading.Thread(target=self._process_speech_queue)
        self._processing_thread.daemon = True
        self._processing_thread.start()
        
        # Set the default voice
        voices = self.tts.list_voices()
        if voices:
            self._voice = voices[0]
            log.info(f"Setting default voice to: {self._voice}")
            self.tts.set_voice(self._voice)
        else:
            log.error("No voices found. TTS will not work properly.")
    
    def terminate(self):
        """Clean up resources when the synth is terminated."""
        self._stop_event.set()
        if self._processing_thread.is_alive():
            self._processing_thread.join(timeout=0.5)
        if hasattr(self, '_player'):
            self._player.close()
        super(SynthDriver, self).terminate()
    
    def _process_speech_queue(self):
        """Process speech requests from the queue."""
        while not self._stop_event.is_set():
            try:
                # Get the next item from the queue with a short timeout
                # Using a shorter timeout makes the system more responsive to cancellation
                item = self._speech_queue.get(timeout=0.05)
                if item is None:
                    # None is a signal to stop speaking
                    if hasattr(self, '_player'):
                        self._player.stop()
                    self._speech_queue.task_done()
                    continue
                
                text, rate, volume = item
                
                # Convert NVDA rate (0-100) to speed factor for Kokoro
                # Lower values in NVDA mean slower speech, but lower values in speed factor mean faster speech
                # So we need to invert the relationship
                speed_factor = 1.0 + ((50 - rate) / 50.0)  # Range: 0.0 to 2.0
                
                # Synthesize and play the speech
                try:
                    # Log what we're about to synthesize
                    log.debug(f"Playing speech: {text[:50]}{'...' if len(text) > 50 else ''}")
                    
                    # Synthesize the waveform
                    waveform = self.tts.synthesize(text, speed=speed_factor)
                    
                    # Adjust volume
                    waveform = waveform * (volume / 100.0)
                    
                    # Convert to int16 for NVDA's audio system
                    waveform = np.clip(waveform, -1.0, 1.0)
                    waveform = (waveform * 32767).astype(np.int16)
                    
                    # Play using NVDA's audio system
                    if hasattr(self, '_player'):
                        self._player.stop()
                        self._player.feed(waveform.tobytes())
                except Exception as e:
                    log.error(f"Error synthesizing speech: {e}")
                    
                    # Try to speak an error message using NVDA's fallback synthesizer
                    try:
                        ui.message("Error in Kokoro TTS synthesis")
                    except:
                        pass
                
                # Mark the task as done
                self._speech_queue.task_done()
                
            except queue.Empty:
                # No items in the queue, just continue
                continue
            except Exception as e:
                log.error(f"Error in speech processing thread: {e}")
                
                # Try to speak an error message using NVDA's fallback synthesizer
                try:
                    ui.message("Error in Kokoro TTS processing")
                except:
                    pass
    
    def speak(self, speechSequence):
        """
        Speak the given sequence.
        
        Args:
            speechSequence: A list of speech sequences to speak
        """
        # Convert the speech sequence to plain text
        text = ""
        lang = None
        
        for item in speechSequence:
            if isinstance(item, str):
                text += item
            elif isinstance(item, speech.commands.LangChangeCommand):
                # Store the language but don't use it directly with eSpeak
                # This prevents NVDA from trying to use eSpeak-NG directly
                lang = item.lang
        
        # Log what we're speaking
        log.debug(f"Speaking: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Add the text to the speech queue
        if text:
            # Ensure a voice is set
            if not self._voice or not self.tts.current_voice:
                log.warning("No voice set, attempting to set default voice")
                voices = self.tts.list_voices()
                if voices:
                    self._voice = voices[0]
                    self.tts.set_voice(self._voice)
                    log.info(f"Set default voice to: {self._voice}")
                else:
                    log.error("No voices available, cannot speak")
                    return
            
            # Stop current speech before adding new speech
            if hasattr(self, '_player'):
                self._player.stop()
            
            # Clear the queue to ensure only the most recent speech request is processed
            try:
                while not self._speech_queue.empty():
                    try:
                        self._speech_queue.get_nowait()
                        self._speech_queue.task_done()
                    except queue.Empty:
                        break
            except Exception as e:
                log.error(f"Error clearing speech queue: {e}")
            
            # Add the new text to the queue
            self._speech_queue.put((text, self._rate, self._volume))
    
    def cancel(self):
        """Stop speaking."""
        # Stop the current playback immediately
        if hasattr(self, '_player'):
            self._player.stop()
            log.debug("Speech cancelled")
    
    def pause(self, switch):
        """Pause or resume speech."""
        if hasattr(self, '_player'):
            if switch:
                self._player.pause()
            else:
                self._player.resume()
    
    def _get_voice(self):
        """Get the current voice."""
        return self._voice
    
    def _set_voice(self, voice):
        """Set the voice."""
        if voice in self.tts.list_voices():
            self._voice = voice
            self.tts.set_voice(voice)
        else:
            log.error(f"Invalid voice: {voice}")
            # Fall back to the first available voice
            available_voices = self.tts.list_voices()
            if available_voices:
                self._voice = available_voices[0]
                self.tts.set_voice(self._voice)
                log.info(f"Falling back to voice: {self._voice}")
    
    def _get_rate(self):
        """Get the current rate."""
        return self._rate
    
    def _set_rate(self, rate):
        """Set the rate."""
        self._rate = max(0, min(100, rate))
    
    def _get_volume(self):
        """Get the current volume."""
        return self._volume
    
    def _set_volume(self, volume):
        """Set the volume."""
        self._volume = max(0, min(100, volume))
    
    def getVoiceNames(self):
        """Get a list of available voices."""
        return sorted(self.availableVoices.keys())


class GlobalPlugin(globalPluginHandler.GlobalPlugin):
    """
    Global plugin for Kokoro TTS integration with NVDA.
    """
    def __init__(self):
        super(GlobalPlugin, self).__init__()
        # Register the synthesizer
        synthDriverHandler.registerSynthDriver(SynthDriver)
        # Add the synthesizer to the list of available synthesizers
        log.info("Kokoro TTS plugin initialized")


# For testing outside of NVDA
if __name__ == "__main__":
    print("This module is designed to be imported by NVDA.")
    print("Testing Kokoro TTS standalone...")
    
    # Paths
    model_path = os.path.join("model", "kokoro.onnx")
    voice_dir = "voices"
    config_path = "config.json"
    tokenizer_path = "tokenizer.json"
    
    # Initialize TTS engine
    tts = KokoroTTS(model_path, voice_dir, config_path, tokenizer_path)
    
    # List available voices
    voices = tts.list_voices()
    print(f"Available voices: {', '.join(voices)}")
    
    if voices:
        # Set voice
        tts.set_voice(voices[0])
        
        # Synthesize and play speech
        text = "This is a test of the Kokoro TTS engine for NVDA."
        waveform = tts.synthesize(text)
        print("Waveform synthesized, but cannot play without NVDA's audio system.") 