from __future__ import annotations

import os
import queue
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import addonHandler
import config
import globalVars
import nvwave
import speech
import synthDriverHandler
from logHandler import log

addonHandler.initTranslation()

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEPS_DIR = os.path.join(_BASE_DIR, "deps")
if _DEPS_DIR not in sys.path:
    sys.path.insert(0, _DEPS_DIR)

from .kokoro_tts import KokoroTTS


@dataclass(frozen=True)
class _Utterance:
    segments: tuple[tuple[str, int | None], ...]
    rate: int
    volume: int
    voice: str
    generation: int


@dataclass(frozen=True)
class _AudioPacket:
    generation: int
    pcm: bytes | None = None
    index: int | None = None
    utteranceEnd: bool = False


class SynthDriver(synthDriverHandler.SynthDriver):
    name = "kokoro"
    description = "Kokoro TTS"
    supportedSettings = (
        synthDriverHandler.SynthDriver.VoiceSetting(),
        synthDriverHandler.SynthDriver.RateSetting(),
        synthDriverHandler.SynthDriver.VolumeSetting(),
    )
    supportedCommands = frozenset({speech.commands.IndexCommand})
    supportedNotifications = frozenset({
        synthDriverHandler.synthIndexReached,
        synthDriverHandler.synthDoneSpeaking,
    })

    @classmethod
    def check(cls) -> bool:
        dataDir = os.path.join(globalVars.appArgs.configPath, "kokoroTTS")
        modelPath = os.path.join(dataDir, "model", "kokoro.onnx")
        voiceDir = os.path.join(dataDir, "voices")
        required = (
            modelPath,
            os.path.join(_BASE_DIR, "config.json"),
            os.path.join(_BASE_DIR, "tokenizer.json"),
            os.path.join(_BASE_DIR, "espeak", "espeak-ng.exe"),
        )
        if not all(os.path.isfile(path) for path in required):
            return False
        if not os.path.isdir(voiceDir) or not any(name.lower().endswith((".npy", ".bin")) for name in os.listdir(voiceDir)):
            return False
        try:
            import numpy  # noqa: F401
            import onnxruntime  # noqa: F401
        except Exception:
            log.debugWarning("Kokoro dependencies could not be imported", exc_info=True)
            return False
        return True

    def __init__(self):
        super().__init__()
        self._rate = 50
        self._volume = 100
        self._generation = 0
        self._generationLock = threading.Lock()
        self._speechQueue: queue.Queue[_Utterance | None] = queue.Queue()
        self._audioQueue: queue.Queue[_AudioPacket | None] = queue.Queue()
        self._stopEvent = threading.Event()
        self._player = nvwave.WavePlayer(
            channels=1,
            samplesPerSec=24000,
            bitsPerSample=16,
            outputDevice=config.conf["audio"]["outputDevice"],
        )
        self._dataDir = os.path.join(globalVars.appArgs.configPath, "kokoroTTS")
        self._userVoiceDir = os.path.join(self._dataDir, "voices")
        self._tts = KokoroTTS(
            model_path=os.path.join(self._dataDir, "model", "kokoro.onnx"),
            voice_dir=self._userVoiceDir,
            config_path=os.path.join(_BASE_DIR, "config.json"),
            tokenizer_path=os.path.join(_BASE_DIR, "tokenizer.json"),
        )
        voices = self._tts.list_voices()
        if not voices:
            raise RuntimeError("Kokoro found no voice files")
        self._voice = voices[0]
        self._tts.set_voice(self._voice)
        self._synthesisWorker = threading.Thread(
            target=self._runSynthesis,
            name="KokoroSynthesis",
            daemon=True,
        )
        self._playbackWorker = threading.Thread(
            target=self._runPlayback,
            name="KokoroPlayback",
            daemon=True,
        )
        self._synthesisWorker.start()
        self._playbackWorker.start()

    def terminate(self):
        self.cancel()
        self._stopEvent.set()
        self._speechQueue.put(None)
        self._audioQueue.put(None)
        if self._synthesisWorker.is_alive():
            self._synthesisWorker.join(timeout=1.0)
        if self._playbackWorker.is_alive():
            self._playbackWorker.join(timeout=1.0)
        try:
            self._player.close()
        finally:
            super().terminate()

    def _currentGeneration(self) -> int:
        with self._generationLock:
            return self._generation

    def _nextGeneration(self) -> int:
        with self._generationLock:
            self._generation += 1
            return self._generation

    @staticmethod
    def _discardQueue(workQueue: queue.Queue) -> None:
        while True:
            try:
                workQueue.get_nowait()
            except queue.Empty:
                return
            else:
                workQueue.task_done()

    @staticmethod
    def _speedForRate(rate: int) -> float:
        return 0.5 + (max(0, min(100, rate)) * 1.5 / 100.0)

    @staticmethod
    def _toPcm16(audio, volume: int) -> bytes:
        import numpy as np

        data = np.asarray(audio, dtype=np.float32)
        if data.ndim != 1:
            data = data.reshape(-1)
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
        data = np.clip(data * (max(0, min(100, volume)) / 100.0), -1.0, 1.0)
        return (data * 32767.0).astype(np.int16).tobytes()

    def _runSynthesis(self) -> None:
        while not self._stopEvent.is_set():
            item = self._speechQueue.get()
            try:
                if item is None:
                    return
                if item.generation != self._currentGeneration():
                    continue
                if item.voice != self._tts.current_voice:
                    self._tts.set_voice(item.voice)

                completed = True
                for text, index in item.segments:
                    if item.generation != self._currentGeneration() or self._stopEvent.is_set():
                        completed = False
                        break
                    pcm = None
                    try:
                        if text and text.strip():
                            audio = self._tts.synthesize(text, speed=self._speedForRate(item.rate))
                            if item.generation != self._currentGeneration() or self._stopEvent.is_set():
                                completed = False
                                break
                            pcm = self._toPcm16(audio, item.volume)
                    except Exception:
                        log.debugWarning(
                            "Kokoro skipped an unsynthesizable speech fragment",
                            exc_info=True,
                        )
                    if item.generation != self._currentGeneration():
                        completed = False
                        break
                    # Emit each completed segment immediately. This lets playback of
                    # segment N overlap ONNX generation of segment N+1.
                    self._audioQueue.put(_AudioPacket(item.generation, pcm or None, index, False))

                if completed and item.generation == self._currentGeneration():
                    self._audioQueue.put(_AudioPacket(item.generation, utteranceEnd=True))
            except Exception:
                log.exception("Kokoro synthesis worker failed")
                if item is not None and item.generation == self._currentGeneration():
                    self._audioQueue.put(_AudioPacket(item.generation, utteranceEnd=True))
            finally:
                self._speechQueue.task_done()

    def _notifyIndexIfCurrent(self, generation: int, index: int) -> None:
        if generation == self._currentGeneration() and not self._stopEvent.is_set():
            synthDriverHandler.synthIndexReached.notify(synth=self, index=index)

    def _runPlayback(self) -> None:
        while not self._stopEvent.is_set():
            packet = self._audioQueue.get()
            try:
                if packet is None:
                    return
                if packet.generation != self._currentGeneration():
                    continue
                if packet.utteranceEnd:
                    # Only playback owns synchronization and completion reporting.
                    self._player.idle()
                    if packet.generation == self._currentGeneration():
                        synthDriverHandler.synthDoneSpeaking.notify(synth=self)
                    continue
                if packet.pcm:
                    if packet.index is not None:
                        self._player.feed(
                            packet.pcm,
                            onDone=lambda generation=packet.generation, index=packet.index: self._notifyIndexIfCurrent(
                                generation,
                                index,
                            ),
                        )
                    else:
                        self._player.feed(packet.pcm)
                elif packet.index is not None:
                    self._notifyIndexIfCurrent(packet.generation, packet.index)
            except Exception:
                log.exception("Kokoro playback worker failed")
                if packet is not None and packet.generation == self._currentGeneration():
                    try:
                        self._player.stop()
                    except Exception:
                        pass
                    synthDriverHandler.synthDoneSpeaking.notify(synth=self)
            finally:
                self._audioQueue.task_done()

    @staticmethod
    def _segmentsFromSequence(speechSequence: Iterable[object]) -> tuple[tuple[str, int | None], ...]:
        segments: list[tuple[str, int | None]] = []
        textParts: list[str] = []
        for item in speechSequence:
            if isinstance(item, str):
                textParts.append(item)
            elif isinstance(item, speech.commands.IndexCommand):
                segments.append(("".join(textParts), item.index))
                textParts.clear()
        if textParts:
            segments.append(("".join(textParts), None))
        return tuple(segment for segment in segments if segment[0] or segment[1] is not None)

    def speak(self, speechSequence):
        segments = self._segmentsFromSequence(speechSequence)
        if not segments:
            return
        generation = self._currentGeneration()
        self._speechQueue.put(_Utterance(segments, self._rate, self._volume, self._voice, generation))

    def cancel(self):
        self._nextGeneration()
        self._discardQueue(self._speechQueue)
        self._discardQueue(self._audioQueue)
        try:
            self._player.stop()
        except Exception:
            log.debugWarning("Kokoro could not stop audio", exc_info=True)

    def pause(self, switch):
        self._player.pause(bool(switch))

    def reloadVoices(self):
        voices = self._tts.reload_voices()
        if self._voice not in voices and voices:
            self.cancel()
            self._voice = voices[0]
            self._tts.set_voice(self._voice)
        return voices

    def _getAvailableVoices(self):
        voices = OrderedDict()
        for voiceId in sorted(self._tts.list_voices()):
            prefix, _, name = voiceId.partition("_")
            language = {
                "af": "en-US",
                "am": "en-US",
                "bf": "en-GB",
                "bm": "en-GB",
            }.get(prefix)
            displayName = name.replace("_", " ").title() if name else voiceId
            voices[voiceId] = synthDriverHandler.VoiceInfo(voiceId, displayName, language)
        return voices

    def _get_voice(self):
        return self._voice

    def _set_voice(self, value):
        if value not in self._tts.list_voices():
            raise LookupError(f"Unknown Kokoro voice: {value}")
        if value == self._voice:
            return
        self.cancel()
        self._voice = value

    def _get_rate(self):
        return self._rate

    def _set_rate(self, value):
        self._rate = max(0, min(100, int(value)))

    def _get_volume(self):
        return self._volume

    def _set_volume(self, value):
        self._volume = max(0, min(100, int(value)))
