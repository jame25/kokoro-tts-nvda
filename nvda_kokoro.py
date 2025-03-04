import os
import sys
import ctypes
import threading
import time
import json
import re
import wx
import numpy as np
from typing import Optional, Dict, Any
import globalVars
import speech
import synthDriverHandler
from synthDriverHandler import SynthDriver, VoiceInfo, synthIndexReached
from logHandler import log
import gui
import gui.settingsDialogs
from gui import nvdaControls
import config
import ui
import nvwave
import scriptHandler
import inputCore
import addonHandler
import globalPluginHandler
from languageHandler import getLanguage
from speech.commands import (
    IndexCommand,
    CharacterModeCommand,
    LangChangeCommand,
    BreakCommand,
    PitchCommand,
    RateCommand,
    VolumeCommand,
)
# Import the translation function
try:
    from languageHandler import gettext as _
except ImportError:
    # Fallback if not available
    def _(text):
        return text

# Fix queue import for Python 2/3 compatibility
try:
    import queue
except ImportError:
    # Python 2 compatibility
    import Queue as queue

# Add the current directory to the path so we can import our TTS module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the Kokoro TTS engine
from kokoro_tts import KokoroTTS

# Initialize the addon
addonHandler.initTranslation()

# We'll use a simpler approach instead of patching NVDA's speech settings dialog
# This will be more compatible with different NVDA versions

# Store the original setSynth function
original_setSynth = synthDriverHandler.setSynth

# Define a patched version that forces a refresh when Kokoro is selected
def patched_setSynth(synthName):
    """
    Patched version of setSynth that forces a refresh when Kokoro is selected.
    """
    result = original_setSynth(synthName)
    
    # If Kokoro was selected, force a refresh of the speech settings panel
    if synthName == "kokoro":
        log.info("Kokoro TTS selected, forcing refresh of speech settings panel")
        try:
            # Find the speech settings dialog
            for dialog in wx.GetTopLevelWindows():
                if dialog and hasattr(dialog, "GetTitle") and dialog.GetTitle() == _("Speech"):
                    log.info("Found speech settings dialog, forcing refresh")
                    
                    # Force a refresh by simulating a change event
                    wx.CallAfter(lambda: force_refresh_dialog(dialog))
                    break
        except Exception as e:
            log.error(f"Error forcing refresh after setSynth: {e}")
    
    return result

def force_refresh_dialog(dialog):
    """Force a refresh of the speech settings dialog."""
    try:
        # Different NVDA versions have different panel structures
        # Try different approaches
        
        # Approach 1: For newer NVDA versions
        if hasattr(dialog, "speechPanel"):
            panel = dialog.speechPanel
            
            # Force a refresh of the voice list
            if hasattr(panel, "_refreshVoiceSettings"):
                panel._refreshVoiceSettings()
                log.info("Refreshed voice settings using _refreshVoiceSettings")
                return True
            
            # Targeted approach similar to Sonata addon
            try:
                # Get the current synth driver
                synth = synthDriverHandler.getSynth()
                if synth.name == "kokoro":
                    log.info("Using targeted approach to refresh voice panel")
                    
                    # Get the voice panel
                    voice_panel = panel.voicePanel
                    
                    # Update the voice list
                    if hasattr(voice_panel, "voiceList") and hasattr(synth, "availableVoices"):
                        voices = list(synth.availableVoices.values())
                        voice_panel.voiceList.SetItems([v.displayName for v in voices])
                        
                        # Update based on config
                        if hasattr(voice_panel, "updateDriverSettings"):
                            voice_panel.updateDriverSettings("voice")
                            log.info("Successfully updated voice list using targeted approach")
                            return True
                        else:
                            # Select the current voice
                            current_voice = synth.voice
                            for i, voice in enumerate(synth.availableVoices):
                                if voice == current_voice:
                                    voice_panel.voiceList.SetSelection(i)
                                    break
                            log.info("Updated voice list selection")
                            return True
            except Exception as e:
                log.error(f"Error in targeted approach: {e}")
            
            # Alternative: Destroy and recreate the panel
            try:
                panel.Destroy()
                dialog.speechPanel = gui.settingsDialogs.SpeechPanel(dialog)
                dialog.speechPanel.SetSizerAndFit(dialog.speechPanel.sizer)
                dialog.Layout()
                log.info("Refreshed speech panel by recreating it")
                return True
            except Exception as e:
                log.error(f"Error recreating speech panel: {e}")
        
        # Approach 2: For older NVDA versions
        for child in dialog.GetChildren():
            if hasattr(child, "GetLabel") and child.GetLabel() == _("Speech"):
                try:
                    child.Destroy()
                    dialog.makeSettings(dialog)
                    dialog.postInit()
                    dialog.Fit()
                    log.info("Refreshed speech panel (older NVDA version)")
                    return True
                except Exception as e:
                    log.error(f"Error refreshing speech panel (older NVDA): {e}")
        
        # Approach 3: Last resort - try to find and trigger the onSynthesizerChange method
        for child in dialog.GetChildren():
            if hasattr(child, "onSynthesizerChange"):
                try:
                    # Create a dummy event
                    evt = wx.CommandEvent(wx.wxEVT_CHOICE)
                    child.onSynthesizerChange(evt)
                    log.info("Triggered onSynthesizerChange manually")
                    return True
                except Exception as e:
                    log.error(f"Error triggering onSynthesizerChange: {e}")
        
        log.warning("Could not find a way to refresh the speech panel")
        return False
    except Exception as e:
        log.error(f"Error in force_refresh_dialog: {e}")
        return False

# Apply the patch
synthDriverHandler.setSynth = patched_setSynth
log.info("Patched synthDriverHandler.setSynth to force refresh for Kokoro TTS")

class SynthDriver(synthDriverHandler.SynthDriver):
    """
    NVDA Synthesizer driver for Kokoro TTS.
    """
    name = "kokoro"
    description = "Kokoro TTS"
    
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
    
    # Convert supportedSettings from a static tuple to a dynamic property
    @property
    def supportedSettings(self):
        """
        Dynamic property that returns the supported settings.
        This allows us to refresh the UI when needed by changing the returned settings.
        """
        settings = [
            synthDriverHandler.SynthDriver.VoiceSetting(),
            synthDriverHandler.SynthDriver.RateSetting(),
            synthDriverHandler.SynthDriver.VolumeSetting(),
        ]
        
        # Store the settings to detect changes
        if not hasattr(self, "_cachedSupportedSettings") or self._cachedSupportedSettings != settings:
            self._cachedSupportedSettings = settings
            # If this is not the first time (initialization), trigger a refresh
            if hasattr(self, "_tts") and self._tts:
                log.info("Supported settings changed, triggering refresh")
                # Trigger a refresh by setting a property
                self._refreshUI()
        
        return settings
    
    def _refreshUI(self):
        """
        Force NVDA to refresh the UI by setting properties to their current values.
        This is a non-intrusive way to refresh the speech panel.
        """
        # Use a debounce mechanism to prevent too many refreshes
        if hasattr(self, "_refreshPending") and self._refreshPending:
            log.debug("Refresh already pending, skipping")
            return
            
        try:
            # Check if wx is available
            import wx
            if not wx.GetApp():
                log.debug("wx.App not ready, skipping UI refresh")
                return
                
            # Mark that a refresh is pending
            self._refreshPending = True
            
            # Use a single property change instead of multiple
            # Voice change is usually the most important for the UI
            current_voice = self.voice
            if current_voice:
                log.debug("Triggering UI refresh via voice property")
                self.voice = current_voice
            else:
                # Fallback to rate if voice isn't set
                log.debug("Triggering UI refresh via rate property")
                current_rate = self.rate
                self.rate = current_rate
                
            log.info("Triggered UI refresh by resetting properties")
        except Exception as e:
            if "quota" in str(e).lower():
                log.warning(f"Windows quota error during UI refresh: {e}")
            else:
                log.error(f"Error refreshing UI: {e}")
        finally:
            # Clear the pending flag after a short delay
            # This prevents rapid successive refreshes
            try:
                if wx.GetApp():
                    wx.CallLater(500, self._clearRefreshPending)
                else:
                    # If wx.App is not ready, clear the flag immediately
                    self._refreshPending = False
            except Exception as e:
                log.error(f"Error scheduling refresh pending clear: {e}")
                # Clear the flag immediately as a fallback
                self._refreshPending = False
    
    def _clearRefreshPending(self):
        """Clear the refresh pending flag."""
        self._refreshPending = False
    
    @classmethod
    def check(cls):
        """
        Check if this synthesizer is available.
        @return: C{True} if this synthesizer is available, C{False} otherwise.
        @rtype: bool
        """
        # Check if the required files exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "model", "kokoro.onnx")
        voice_dir = os.path.join(current_dir, "voices")
        config_path = os.path.join(current_dir, "config.json")
        tokenizer_path = os.path.join(current_dir, "tokenizer.json")
        
        # Log the paths we're checking
        log.info(f"Checking Kokoro TTS files: model={os.path.exists(model_path)}, "
                f"voices_dir={os.path.exists(voice_dir)}, "
                f"config={os.path.exists(config_path)}, "
                f"tokenizer={os.path.exists(tokenizer_path)}")
        
        # Check if all required files exist
        if not os.path.exists(model_path):
            log.error(f"Kokoro TTS model file not found: {model_path}")
            return False
        if not os.path.exists(voice_dir) or not os.listdir(voice_dir):
            log.error(f"Kokoro TTS voice directory not found or empty: {voice_dir}")
            return False
        if not os.path.exists(config_path):
            log.error(f"Kokoro TTS config file not found: {config_path}")
            return False
        if not os.path.exists(tokenizer_path):
            log.error(f"Kokoro TTS tokenizer file not found: {tokenizer_path}")
            return False
        
        # Check if we can import onnxruntime
        try:
            import onnxruntime
            log.info(f"ONNX Runtime version: {onnxruntime.__version__}")
        except ImportError:
            log.error("ONNX Runtime not installed")
            return False
        
        # All checks passed
        log.info("Kokoro TTS synthesizer is available")
        return True
    
    # Add the availableVoices property
    @property
    def availableVoices(self):
        """Return a dictionary of available voices."""
        try:
            voices = {}
            
            # Log that we're getting available voices
            log.debug("Getting available voices for Kokoro TTS")
            
            for voice in self.tts.list_voices():
                # Extract language information from the voice name
                # Voice names are expected to be in the format: language_name
                # e.g., af_bella, am_michael
                try:
                    parts = voice.split('_')
                    if len(parts) >= 2:
                        lang_code = parts[0]
                        name = parts[1]
                    else:
                        # Handle case where voice name doesn't follow expected format
                        log.warning(f"Voice name {voice} doesn't follow expected format (lang_name)")
                        lang_code = "unknown"
                        name = voice
                    
                    # Create voice info without gender
                    try:
                        # Try with positional arguments (most compatible approach)
                        voices[voice] = synthDriverHandler.VoiceInfo(voice, name.capitalize(), lang_code)
                    except TypeError:
                        # If that fails, try with just id and name
                        try:
                            voices[voice] = synthDriverHandler.VoiceInfo(voice, name.capitalize())
                        except Exception as e:
                            log.error(f"Error creating voice info for {voice}: {e}")
                            # Ultimate fallback - just use a dictionary if VoiceInfo fails completely
                            voices[voice] = {"id": voice, "name": voice}
                except Exception as e:
                    log.error(f"Error parsing voice info for {voice}: {e}")
                    # Fallback to basic voice info with minimal parameters
                    try:
                        voices[voice] = synthDriverHandler.VoiceInfo(voice, voice)
                    except Exception as e2:
                        log.error(f"Error creating basic voice info for {voice}: {e2}")
                        # Ultimate fallback - just use a dictionary if VoiceInfo fails completely
                        voices[voice] = {"id": voice, "name": voice}
            
            log.info(f"Available voices: {', '.join(voices.keys())}")
            return voices
        except Exception as e:
            log.error(f"Error getting available voices: {e}")
            return {}
    
    def __init__(self):
        """Initialize the Kokoro TTS driver."""
        super(SynthDriver, self).__init__()
        
        # Paths to model files
        self.model_path = os.path.join(current_dir, "model", "kokoro.onnx")
        self.voice_dir = os.path.join(current_dir, "voices")
        self.config_path = os.path.join(current_dir, "config.json")
        self.tokenizer_path = os.path.join(current_dir, "tokenizer.json")
        
        # Initialize instance variables for UI refresh detection
        self._inSpeechPanel = False
        self._refreshPending = False
        
        # Initialize all timers to None
        self._speechPanelDetectionTimer = None
        self._rateChangeTimer = None
        self._volumeChangeTimer = None
        self._voiceChangeTimer = None
        
        # Safely create a timer to detect when we're in the speech panel
        try:
            import wx
            if wx.GetApp():
                self._speechPanelDetectionTimer = wx.Timer()
                self._speechPanelDetectionTimer.Bind(wx.EVT_TIMER, self._checkForSpeechPanel)
                self._speechPanelDetectionTimer.Start(3000)  # Check every 3 seconds
                log.debug("Speech panel detection timer started")
            else:
                log.warning("wx.App not ready, will initialize timer later")
                # We'll initialize the timer later when the app is ready
                wx.CallAfter(self._initTimer)
        except Exception as e:
            log.error(f"Error initializing speech panel detection timer: {e}")
        
        # Initialize the TTS engine
        try:
            log.info("Initializing Kokoro TTS")
            self.tts = KokoroTTS(
                model_path=self.model_path,
                voice_dir=self.voice_dir,
                config_path=self.config_path,
                tokenizer_path=self.tokenizer_path
            )
            
            # Initialize the speech queue and processing thread
            self._speech_queue = queue.Queue()
            self._stop_event = threading.Event()
            self._processing_thread = threading.Thread(
                target=self._process_speech_queue,
                daemon=True
            )
            self._processing_thread.start()
            
            # Initialize the audio player
            self._player = None
            
            # Set default voice
            voices = self.tts.list_voices()
            if voices:
                self._voice = voices[0]
                self.tts.set_voice(self._voice)
                log.info(f"Default voice set to {self._voice}")
            else:
                log.error("No voices available")
                self._voice = None
            
            # Set default rate and volume
            # Note: KokoroTTS doesn't have direct set_rate and set_volume methods
            # We'll store these values and apply them during synthesis
            self._rate = 50  # NVDA uses 0-100 scale
            self._volume = 100  # NVDA uses 0-100 scale
            
            log.info("Kokoro TTS initialized successfully")
        except Exception as e:
            log.error(f"Error initializing Kokoro TTS: {e}")
            raise
    
    def terminate(self):
        """Clean up resources when the synth is terminated."""
        # Stop all timers
        for timer_attr in ["_speechPanelDetectionTimer", "_rateChangeTimer", "_volumeChangeTimer", "_voiceChangeTimer"]:
            try:
                if hasattr(self, timer_attr):
                    timer = getattr(self, timer_attr)
                    if timer:
                        timer.Stop()
                        setattr(self, timer_attr, None)
            except Exception as e:
                log.error(f"Error stopping timer {timer_attr}: {e}")
        
        # Clean up the original resources
        self._stop_event.set()
        if self._processing_thread.is_alive():
            self._processing_thread.join(timeout=0.5)
        
        # Close the audio player if it exists
        if hasattr(self, '_player') and self._player is not None:
            try:
                self._player.close()
                self._player = None
                log.debug("Closed audio player")
            except Exception as e:
                log.error(f"Error closing audio player: {e}")
                self._player = None
        
        super(SynthDriver, self).terminate()
    
    def _process_speech_queue(self):
        """Process speech requests from the queue."""
        while not self._stop_event.is_set():
            try:
                # Try to get an item without waiting
                try:
                    item = self._speech_queue.get_nowait()
                except queue.Empty:
                    # If the queue is empty, wait a bit and try again
                    time.sleep(0.05)
                    continue
                
                # None is a signal to stop
                if item is None:
                    break
                
                # Process the speech item
                text, index, rate, volume, voice = item
                
                try:
                    # Generate speech
                    if not text:
                        continue
                    
                    # Set parameters for this speech item
                    if voice and voice != self.tts.current_voice:
                        self.tts.set_voice(voice)
                    
                    # Convert NVDA rate (0-100) to speed factor for Kokoro
                    # Lower values in NVDA mean slower speech, but lower values in speed factor mean faster speech
                    # So we need to invert the relationship
                    # Map rate 0-100 to speed factor 0.5-1.5
                    speed_factor = 1.5 - (rate / 100.0)  # This maps 0->1.5, 100->0.5
                    
                    # Generate audio
                    audio_data = self.tts.synthesize(text, speed=speed_factor)
                    
                    # Adjust volume if needed (if KokoroTTS doesn't handle volume internally)
                    if volume != 100 and audio_data is not None:
                        # Convert to float32, adjust volume, then back to original format
                        audio_data = audio_data * (volume / 100.0)
                    
                    # If we got audio data, play it
                    if audio_data is not None and len(audio_data) > 0 and not self._stop_event.is_set():
                        # Validate audio data
                        valid_audio = True
                        if isinstance(audio_data, bytes) and len(audio_data) < 100:
                            log.warning(f"Audio data suspiciously small ({len(audio_data)} bytes), may be invalid")
                            if len(audio_data) < 10:  # Extremely small data is definitely invalid
                                log.error("Audio data too small to be valid, skipping playback")
                                valid_audio = False
                        
                        # Log the audio data type for debugging
                        log.debug(f"Audio data type: {type(audio_data)}")
                        if not isinstance(audio_data, (bytes, bytearray)) and not str(type(audio_data)).find('numpy') >= 0:
                            log.warning(f"Unexpected audio data type: {type(audio_data)}")
                            # We'll still try to play it, but it might fail
                        
                        if not valid_audio:
                            log.error("Skipping invalid audio data")
                            self._speech_queue.task_done()
                            continue
                            
                        # Create a wave player if needed
                        if not hasattr(self, '_player') or self._player is None:
                            # Get the output device from NVDA's configuration
                            outputDevice = None
                            try:
                                if hasattr(config.conf, "get") and "audio" in config.conf:
                                    outputDevice = config.conf["audio"].get("outputDevice")
                                if not outputDevice and "speech" in config.conf:
                                    outputDevice = config.conf["speech"].get("outputDevice")
                            except Exception as e:
                                log.warning(f"Could not get output device from NVDA config, using default: {e}")
                            
                            # Initialize the wave player with different approaches
                            player_created = False
                            
                            # Approach 1: Try with minimal parameters (most compatible)
                            if not player_created:
                                try:
                                    self._player = nvwave.WavePlayer(
                                        channels=1,
                                        samplesPerSec=self.tts.sample_rate,
                                        bitsPerSample=16,
                                        outputDevice=outputDevice
                                    )
                                    log.debug(f"Created WavePlayer with basic parameters, sample rate {self.tts.sample_rate}")
                                    player_created = True
                                except Exception as e:
                                    log.warning(f"Could not create WavePlayer with basic parameters: {e}")
                            
                            # Approach 2: Try with different sample rate
                            if not player_created:
                                try:
                                    # Try with standard sample rate
                                    self._player = nvwave.WavePlayer(
                                        channels=1,
                                        samplesPerSec=16000,  # Standard sample rate
                                        bitsPerSample=16,
                                        outputDevice=outputDevice
                                    )
                                    log.debug("Created WavePlayer with standard sample rate 16000")
                                    player_created = True
                                except Exception as e:
                                    log.warning(f"Could not create WavePlayer with standard sample rate: {e}")
                            
                            # Approach 3: Try with positional arguments only
                            if not player_created:
                                try:
                                    # Try with positional arguments
                                    self._player = nvwave.WavePlayer(1, self.tts.sample_rate, 16, outputDevice)
                                    log.debug("Created WavePlayer with positional arguments")
                                    player_created = True
                                except Exception as e:
                                    log.warning(f"Could not create WavePlayer with positional arguments: {e}")
                            
                            # If all approaches failed, we can't play audio
                            if not player_created:
                                log.error("All approaches to create WavePlayer failed")
                                continue
                        
                        # Play the audio
                        try:
                            # Make sure numpy is available
                            np_available = False
                            try:
                                import numpy as np
                                np_available = True
                                log.debug("Successfully imported numpy in _process_speech_queue")
                            except ImportError:
                                log.warning("Could not import numpy, audio conversion may be limited")
                            
                            # Convert audio_data to the right format if needed
                            # Some versions of NVDA expect bytes, others expect numpy arrays
                            if np_available and isinstance(audio_data, np.ndarray):
                                # If it's a numpy array, convert to bytes
                                log.debug(f"Converting numpy array with dtype {audio_data.dtype} to bytes")
                                if audio_data.dtype == np.float32:
                                    # Convert float32 to int16
                                    audio_data = (audio_data * 32767).astype(np.int16).tobytes()
                                elif audio_data.dtype == np.int16:
                                    # Just convert to bytes
                                    audio_data = audio_data.tobytes()
                                else:
                                    # Try to convert to int16 first
                                    audio_data = audio_data.astype(np.int16).tobytes()
                                log.debug(f"Converted audio data to bytes, length: {len(audio_data)}")
                            
                            # Try to feed the audio data
                            log.debug(f"Feeding audio data of type {type(audio_data)} to player")
                            self._player.feed(audio_data)
                        except TypeError as e:
                            log.error(f"TypeError playing audio: {e}")
                            # If there's a type error, try to convert the data differently
                            try:
                                # Make sure numpy is available
                                np_available = False
                                try:
                                    import numpy as np
                                    np_available = True
                                    log.debug("Successfully imported numpy in TypeError handler")
                                except ImportError:
                                    log.error("Could not import numpy for fallback conversion")
                                    self._player = None
                                    continue
                                
                                # Try different approaches based on the error message
                                error_msg = str(e).lower()
                                log.debug(f"Analyzing error message: {error_msg}")
                                
                                if "don't know how to convert parameter" in error_msg:
                                    # This is likely a format issue
                                    log.debug("Detected parameter conversion error, trying alternative formats")
                                    
                                    if isinstance(audio_data, bytes):
                                        # Try converting bytes to a simple array
                                        log.debug("Converting bytes to array")
                                        # Use ctypes to create a simple array
                                        byte_count = len(audio_data)
                                        buffer = (ctypes.c_byte * byte_count)()
                                        for i in range(byte_count):
                                            buffer[i] = audio_data[i]
                                        self._player.feed(buffer)
                                        log.debug("Successfully played audio using ctypes buffer")
                                    elif np_available and isinstance(audio_data, np.ndarray):
                                        # Try different numpy array formats
                                        log.debug(f"Converting numpy array with dtype {audio_data.dtype}")
                                        if audio_data.dtype != np.int16:
                                            audio_data = audio_data.astype(np.int16)
                                        
                                        # Try as raw buffer
                                        buffer = audio_data.tobytes()
                                        byte_count = len(buffer)
                                        c_buffer = (ctypes.c_byte * byte_count)()
                                        for i in range(byte_count):
                                            c_buffer[i] = buffer[i]
                                        self._player.feed(c_buffer)
                                        log.debug("Successfully played audio using ctypes buffer from numpy array")
                                    else:
                                        log.error(f"Unsupported audio data type: {type(audio_data)}")
                                        self._player = None
                                else:
                                    # Try the original numpy conversions
                                    if np_available:
                                        if isinstance(audio_data, bytes):
                                            # Convert bytes to numpy array
                                            audio_data_np = np.frombuffer(audio_data, dtype=np.int16)
                                            self._player.feed(audio_data_np)
                                            log.debug("Successfully played audio after converting bytes to numpy array")
                                        elif isinstance(audio_data, np.ndarray):
                                            # Try different numpy array formats
                                            if audio_data.dtype != np.int16:
                                                audio_data = audio_data.astype(np.int16)
                                            self._player.feed(audio_data)
                                            log.debug("Successfully played audio after converting numpy array format")
                                        else:
                                            log.error(f"Unsupported audio data type: {type(audio_data)}")
                                            self._player = None
                                    else:
                                        # Ultimate fallback: try to use a simple buffer approach
                                        log.debug("Trying ultimate fallback without numpy")
                                        try:
                                            if isinstance(audio_data, bytes):
                                                # Create a simple buffer from bytes
                                                byte_count = len(audio_data)
                                                buffer = (ctypes.c_byte * byte_count)()
                                                for i in range(byte_count):
                                                    buffer[i] = audio_data[i]
                                                self._player.feed(buffer)
                                                log.debug("Successfully played audio using simple buffer fallback")
                                            else:
                                                log.error(f"Cannot convert {type(audio_data)} without numpy")
                                                self._player = None
                                        except Exception as e3:
                                            log.error(f"Ultimate fallback failed: {e3}")
                                            self._player = None
                            except Exception as e2:
                                log.error(f"Error in audio conversion fallback: {e2}")
                                self._player = None
                        except Exception as e:
                            log.error(f"Error playing audio: {e}")
                            # If there was an error with the player, set it to None to force recreation next time
                            self._player = None
                        
                        # Signal when done speaking
                        if index is not None:
                            try:
                                # Try different ways to notify about the index
                                # Method 1: Use synthIndexReached if available
                                if hasattr(synthDriverHandler, 'synthIndexReached'):
                                    synthDriverHandler.synthIndexReached.notify(synth=self, index=index)
                                # Method 2: Use the index_reached method if available
                                elif hasattr(self, 'index_reached'):
                                    self.index_reached(index)
                                # Method 3: Use the onIndexReached method if available
                                elif hasattr(synthDriverHandler, 'onIndexReached'):
                                    synthDriverHandler.onIndexReached(index)
                                else:
                                    log.warning(f"Could not notify index {index} reached - no notification method available")
                            except Exception as e:
                                log.error(f"Error notifying index reached: {e}")
                    
                    # Signal that we're done with this item
                    self._speech_queue.task_done()
                
                except Exception as e:
                    log.error(f"Error processing speech: {e}")
                    self._speech_queue.task_done()
            
            except Exception as e:
                log.error(f"Error in speech queue processing: {e}")
                time.sleep(0.1)  # Avoid tight loop in case of persistent errors
    
    def speak(self, speechSequence):
        """
        Speak the given sequence.
        
        Args:
            speechSequence: A list of speech sequences to speak
        """
        # Convert the speech sequence to plain text
        text = ""
        lang = None
        index = None
        
        for item in speechSequence:
            if isinstance(item, str):
                text += item
            elif isinstance(item, speech.commands.IndexCommand):
                index = item.index
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
            if hasattr(self, '_player') and self._player is not None:
                try:
                    self._player.stop()
                except Exception as e:
                    log.error(f"Error stopping current speech: {e}")
                    # If there was an error with the player, set it to None to force recreation
                    self._player = None
            
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
            
            # Add the new text to the queue with the index
            self._speech_queue.put((text, index, self._rate, self._volume, self._voice))
    
    def cancel(self):
        """Stop speaking."""
        # Stop the current playback immediately
        if hasattr(self, '_player') and self._player is not None:
            try:
                self._player.stop()
                log.debug("Speech cancelled")
            except Exception as e:
                log.error(f"Error cancelling speech: {e}")
    
    def pause(self, switch):
        """Pause or resume speech."""
        if hasattr(self, '_player') and self._player is not None:
            try:
                if switch:
                    self._player.pause()
                    log.debug("Speech paused")
                else:
                    self._player.resume()
                    log.debug("Speech resumed")
            except Exception as e:
                log.error(f"Error {'pausing' if switch else 'resuming'} speech: {e}")
    
    def _get_voice(self):
        """Get the current voice."""
        return self._voice
    
    def _set_voice(self, voice):
        """Set the voice."""
        if voice in self.tts.list_voices():
            self._voice = voice
            self.tts.set_voice(voice)
            log.info(f"Voice set to: {voice}")
            
            # Trigger a UI refresh when voice changes
            # This ensures the speech panel is updated with the new voice
            try:
                import wx
                if wx.GetApp() and hasattr(self, "_inSpeechPanel") and self._inSpeechPanel:
                    # Use CallLater instead of CallAfter to reduce resource usage
                    # Also add a small delay to allow multiple property changes to complete
                    if not self._voiceChangeTimer:
                        self._voiceChangeTimer = wx.Timer()
                        self._voiceChangeTimer.Bind(wx.EVT_TIMER, lambda evt: self._refreshUI())
                    else:
                        self._voiceChangeTimer.Stop()
                    self._voiceChangeTimer.Start(300, oneShot=True)  # Refresh after 300ms
            except Exception as e:
                log.error(f"Error setting up voice change timer: {e}")
        else:
            log.error(f"Invalid voice: {voice}")
            # Fall back to the first available voice
            available_voices = self.tts.list_voices()
            if available_voices:
                self._voice = available_voices[0]
                self.tts.set_voice(self._voice)
                log.info(f"Falling back to voice: {self._voice}")
    
    def _get_rate(self):
        """Get the rate."""
        return self._rate
    
    def _set_rate(self, rate):
        """Set the rate."""
        self._rate = max(0, min(100, rate))
        # Note: We don't call self.tts.set_rate() because it doesn't exist
        # The rate will be applied during synthesis
        
        # If we're in the speech panel, this might be a good time to refresh the UI
        # This helps ensure all settings are properly displayed
        try:
            import wx
            if wx.GetApp() and hasattr(self, "_inSpeechPanel") and self._inSpeechPanel:
                # Use CallLater instead of CallAfter to reduce resource usage
                # Also add a small delay to allow multiple property changes to complete
                if not self._rateChangeTimer:
                    self._rateChangeTimer = wx.Timer()
                    self._rateChangeTimer.Bind(wx.EVT_TIMER, lambda evt: self._refreshUI())
                else:
                    self._rateChangeTimer.Stop()
                self._rateChangeTimer.Start(300, oneShot=True)  # Refresh after 300ms
        except Exception as e:
            log.error(f"Error setting up rate change timer: {e}")
    
    def _get_volume(self):
        """Get the current volume."""
        return self._volume
    
    def _set_volume(self, volume):
        """Set the volume."""
        self._volume = max(0, min(100, volume))
        # Note: We don't call self.tts.set_volume() because it doesn't exist
        # The volume will be applied during synthesis
        
        # If we're in the speech panel, this might be a good time to refresh the UI
        # This helps ensure all settings are properly displayed
        try:
            import wx
            if wx.GetApp() and hasattr(self, "_inSpeechPanel") and self._inSpeechPanel:
                # Use CallLater instead of CallAfter to reduce resource usage
                # Also add a small delay to allow multiple property changes to complete
                if not self._volumeChangeTimer:
                    self._volumeChangeTimer = wx.Timer()
                    self._volumeChangeTimer.Bind(wx.EVT_TIMER, lambda evt: self._refreshUI())
                else:
                    self._volumeChangeTimer.Stop()
                self._volumeChangeTimer.Start(300, oneShot=True)  # Refresh after 300ms
        except Exception as e:
            log.error(f"Error setting up volume change timer: {e}")
    
    def getVoiceNames(self):
        """Get a list of available voices."""
        return sorted(self.availableVoices.keys())

    def _checkForSpeechPanel(self, evt=None):
        """
        Check if the speech panel is currently open.
        This helps us know when to trigger UI refreshes.
        """
        # If the wx.App isn't ready yet, try to initialize the timer
        try:
            import wx
            if not self._speechPanelDetectionTimer and wx.GetApp():
                self._initTimer()
                return
        except Exception as e:
            log.debug(f"Error checking for wx.App in _checkForSpeechPanel: {e}")
            return
            
        # Skip checks if we've checked recently
        if hasattr(self, "_lastCheckTime"):
            current_time = time.time()
            if current_time - self._lastCheckTime < 2.0:  # Only check every 2 seconds
                return
            self._lastCheckTime = current_time
        else:
            self._lastCheckTime = time.time()
            
        try:
            # Look for the speech settings dialog
            inSpeechPanel = False
            
            # Limit the number of windows we check to avoid resource issues
            top_windows = wx.GetTopLevelWindows()
            if len(top_windows) > 10:  # If there are too many windows, be more selective
                top_windows = [w for w in top_windows if hasattr(w, "GetTitle") and w.GetTitle() == _("Speech")]
                
            for dialog in top_windows:
                if dialog and hasattr(dialog, "GetTitle") and dialog.GetTitle() == _("Speech"):
                    # Found the speech settings dialog
                    if hasattr(dialog, "speechPanel"):
                        # Check if our synth is selected
                        panel = dialog.speechPanel
                        if hasattr(panel, "synthList") and panel.synthList:
                            selection = panel.synthList.GetStringSelection()
                            if selection == self.description:
                                inSpeechPanel = True
                                # Store a reference to the dialog for potential future use
                                self._speechSettingsDialog = dialog
                                break
            
            # If the state changed, log it
            if inSpeechPanel != self._inSpeechPanel:
                self._inSpeechPanel = inSpeechPanel
                log.debug(f"In speech panel: {inSpeechPanel}")
                
                # If we just entered the speech panel, refresh the UI
                # But use CallLater with a slight delay to avoid resource issues
                if inSpeechPanel:
                    wx.CallLater(100, self._refreshUI)
        except Exception as e:
            if "quota" in str(e).lower():
                log.warning(f"Windows quota error during speech panel check: {e}")
            else:
                log.error(f"Error checking for speech panel: {e}")
                
            # If we get an error, slow down the checks even more
            self._lastCheckTime = time.time() + 5.0  # Wait longer before next check

    def _initTimer(self):
        """Initialize the speech panel detection timer when the wx.App is ready."""
        try:
            import wx
            if not self._speechPanelDetectionTimer and wx.GetApp():
                log.debug("Initializing speech panel detection timer")
                self._speechPanelDetectionTimer = wx.Timer()
                self._speechPanelDetectionTimer.Bind(wx.EVT_TIMER, self._checkForSpeechPanel)
                self._speechPanelDetectionTimer.Start(3000)  # Check every 3 seconds
                log.debug("Speech panel detection timer started")
            elif not wx.GetApp():
                log.warning("wx.App still not ready, will try again later")
                wx.CallAfter(self._initTimer)
        except Exception as e:
            log.error(f"Error in _initTimer: {e}")


class GlobalPlugin(globalPluginHandler.GlobalPlugin):
    """
    Global plugin for Kokoro TTS integration with NVDA.
    """
    
    # Define the scripts that this global plugin will provide
    __gestures = {
        "kb:NVDA+shift+r": "refreshSpeechPanel",
    }
    
    def __init__(self):
        """Initialize the global plugin."""
        super(GlobalPlugin, self).__init__()
        # Register the synthesizer
        try:
            synthDriverHandler.registerSynthDriver(SynthDriver)
            log.info("Kokoro TTS synthesizer registered successfully")
        except Exception as e:
            log.error(f"Failed to register Kokoro TTS synthesizer: {e}")
        
        # Add the synthesizer to the list of available synthesizers
        log.info("Kokoro TTS plugin initialized")
        
        # Register for the event when the synthesizer changes
        try:
            if hasattr(synthDriverHandler.synthDriverHandler, "post_synthDriverChanged"):
                synthDriverHandler.synthDriverHandler.post_synthDriverChanged.register(self.on_synth_changed)
                log.info("Registered for synthesizer change events")
        except Exception as e:
            log.error(f"Failed to register for synthesizer change events: {e}")
        
        # Add a script to force refresh of the speech settings panel
        self._speechSettingsDialog = None
    
    def terminate(self):
        """Clean up when the plugin is terminated."""
        # Unregister from the event
        try:
            if hasattr(synthDriverHandler.synthDriverHandler, "post_synthDriverChanged"):
                synthDriverHandler.synthDriverHandler.post_synthDriverChanged.unregister(self.on_synth_changed)
                log.info("Unregistered from synthesizer change events")
        except Exception as e:
            log.error(f"Failed to unregister from synthesizer change events: {e}")
        
        super(GlobalPlugin, self).terminate()
    
    def on_synth_changed(self):
        """
        Called when the synthesizer is changed.
        This ensures the speech settings panel is refreshed with the correct voice settings.
        """
        try:
            # Check if the current synth is Kokoro
            if synthDriverHandler.getSynth().name == "kokoro":
                log.info("Kokoro TTS selected, refreshing speech settings panel")
                # Force a refresh of the speech settings panel if it's open
                for dialog in wx.GetTopLevelWindows():
                    if dialog and hasattr(dialog, "GetTitle") and dialog.GetTitle() == _("Speech"):
                        # Found the speech settings dialog
                        log.info("Found speech settings dialog, refreshing")
                        # Store a reference to the dialog for the refresh script
                        self._speechSettingsDialog = dialog
                        # Try to refresh the panel
                        self.force_refresh_speech_panel(dialog)
        except Exception as e:
            log.error(f"Error in on_synth_changed: {e}")
    
    def force_refresh_speech_panel(self, dialog=None):
        """Force a refresh of the speech settings panel."""
        try:
            if not dialog:
                dialog = self._speechSettingsDialog
                if not dialog or not dialog.IsShown():
                    # No dialog or it's not shown, nothing to refresh
                    return
            
            log.info("Forcing refresh of speech settings panel")
            
            # Try the targeted approach first (similar to Sonata addon)
            try:
                # Get the current synth driver
                synth = synthDriverHandler.getSynth()
                if synth.name == "kokoro":
                    log.info("Using targeted approach to refresh voice panel")
                    
                    # Get the speech panel and voice panel
                    if hasattr(dialog, "speechPanel"):
                        panel = dialog.speechPanel
                        if hasattr(panel, "voicePanel"):
                            voice_panel = panel.voicePanel
                            
                            # Update the voice list
                            if hasattr(voice_panel, "voiceList") and hasattr(synth, "availableVoices"):
                                voices = list(synth.availableVoices.values())
                                voice_panel.voiceList.SetItems([v.displayName for v in voices])
                                
                                # Update based on config
                                if hasattr(voice_panel, "updateDriverSettings"):
                                    voice_panel.updateDriverSettings("voice")
                                    log.info("Successfully updated voice list using targeted approach")
                                    return
                                else:
                                    # Select the current voice
                                    current_voice = synth.voice
                                    for i, voice in enumerate(synth.availableVoices):
                                        if voice == current_voice:
                                            voice_panel.voiceList.SetSelection(i)
                                            break
                                    log.info("Updated voice list selection")
                                    return
            except Exception as e:
                log.error(f"Error in targeted approach: {e}")
            
            # Try different approaches to refresh the panel
            try:
                # Approach 1: Recreate the speech panel
                if hasattr(dialog, "speechPanel"):
                    panel = dialog.speechPanel
                    # Store the current synthesizer selection
                    currentSynth = None
                    if hasattr(panel, "synthList") and panel.synthList:
                        currentSynth = panel.synthList.GetStringSelection()
                    
                    # Destroy and recreate the panel
                    panel.Destroy()
                    dialog.speechPanel = gui.settingsDialogs.SpeechPanel(dialog)
                    dialog.speechPanel.SetSizerAndFit(dialog.speechPanel.sizer)
                    
                    # Restore the synthesizer selection if needed
                    if currentSynth and hasattr(dialog.speechPanel, "synthList"):
                        index = dialog.speechPanel.synthList.FindString(currentSynth)
                        if index != wx.NOT_FOUND:
                            dialog.speechPanel.synthList.SetSelection(index)
                            # Trigger the change event
                            evt = wx.CommandEvent(wx.wxEVT_CHOICE)
                            evt.SetInt(index)
                            dialog.speechPanel.onSynthesizerChange(evt)
                    
                    dialog.Layout()
                    dialog.Fit()
                    log.info("Speech panel refreshed (approach 1)")
                    return
            except Exception as e:
                log.error(f"Error in approach 1: {e}")
            
            # Approach 2: For newer NVDA versions
            try:
                # Find the speech category in the settings dialog
                for child in dialog.GetChildren():
                    if hasattr(child, "GetLabel") and child.GetLabel() == _("Speech"):
                        # Found the speech category
                        child.Destroy()
                        # Recreate the panel
                        dialog.makeSettings(dialog)
                        dialog.postInit()
                        dialog.Fit()
                        log.info("Speech panel refreshed (approach 2)")
                        return
            except Exception as e:
                log.error(f"Error in approach 2: {e}")
            
            # Approach 3: Last resort - close and reopen the dialog
            try:
                # Get the dialog position
                pos = dialog.GetPosition()
                # Close the dialog
                dialog.Close()
                # Open a new speech settings dialog
                wx.CallLater(100, lambda: gui.mainFrame._popupSettingsDialog(gui.settingsDialogs.SpeechSettingsDialog))
                # Try to position it at the same place
                wx.CallLater(200, lambda: self._position_dialog(pos))
                log.info("Speech panel refreshed (approach 3)")
                return
            except Exception as e:
                log.error(f"Error in approach 3: {e}")
            
            log.warning("All approaches to refresh the speech panel failed")
        except Exception as e:
            log.error(f"Error forcing refresh of speech panel: {e}")
    
    def _position_dialog(self, pos):
        """Position the speech settings dialog at the specified position."""
        try:
            for dialog in wx.GetTopLevelWindows():
                if dialog and hasattr(dialog, "GetTitle") and dialog.GetTitle() == _("Speech"):
                    dialog.SetPosition(pos)
                    break
        except:
            pass
    
    def script_refreshSpeechPanel(self, gesture):
        """Script to force a refresh of the speech settings panel."""
        # Check if Kokoro TTS is the current synthesizer
        if synthDriverHandler.getSynth().name == "kokoro":
            # Force a refresh of the speech settings panel
            self.force_refresh_speech_panel()
            ui.message(_("Refreshing speech settings panel"))
        else:
            ui.message(_("Kokoro TTS is not the active synthesizer"))
    # Add the script binding using the older style
    script_refreshSpeechPanel.__doc__ = _("Refresh the speech settings panel")
    script_refreshSpeechPanel.category = _("Kokoro TTS")


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
