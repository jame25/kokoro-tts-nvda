from __future__ import annotations

import ctypes
import os
import queue
import re
import sys
import threading
import time
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

if hasattr(os, "add_dll_directory"):
    for root, dirs, files in os.walk(_DEPS_DIR):
        if any(f.lower().endswith(".dll") for f in files):
            try:
                os.add_dll_directory(root)
            except Exception:
                pass
    espeak_dir = os.path.join(_BASE_DIR, "espeak")
    if os.path.isdir(espeak_dir):
        try:
            os.add_dll_directory(espeak_dir)
        except Exception:
            pass

# Explicitly preload C++ runtime DLLs and ONNX Runtime DLLs to guarantee symbol resolution
_preload_dlls = [
    os.path.join(_DEPS_DIR, "vcruntime140.dll"),
    os.path.join(_DEPS_DIR, "vcruntime140_1.dll"),
    os.path.join(_DEPS_DIR, "msvcp140.dll"),
    os.path.join(_DEPS_DIR, "msvcp140_1.dll"),
    os.path.join(_DEPS_DIR, "onnxruntime", "capi", "onnxruntime.dll"),
    os.path.join(_DEPS_DIR, "onnxruntime", "capi", "onnxruntime_providers_shared.dll"),
]
for _dll_path in _preload_dlls:
    if os.path.isfile(_dll_path):
        try:
            ctypes.CDLL(_dll_path)
        except Exception:
            pass

from .kokoro_tts import KokoroTTS


@dataclass(frozen=True)
class _Utterance:
    full_text: str
    spans: tuple[tuple[int, int, int | None], ...]
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
    isLastChunk: bool = False


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
        quantPath = os.path.join(dataDir, "model", "kokoro_quant.onnx")
        hasModel = os.path.isfile(quantPath) or os.path.isfile(modelPath)
        voiceDir = os.path.join(dataDir, "voices")
        required = (
            os.path.join(_BASE_DIR, "config.json"),
            os.path.join(_BASE_DIR, "tokenizer.json"),
            os.path.join(_BASE_DIR, "espeak", "espeak-ng.exe"),
        )
        if not hasModel or not all(os.path.isfile(path) for path in required):
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
        self._inferenceLock = threading.Lock()
        self._speechQueue: queue.Queue[_Utterance | None] = queue.Queue()
        self._audioQueue: queue.Queue[_AudioPacket | None] = queue.Queue()
        self._stopEvent = threading.Event()
        self._isSpeakingEvent = threading.Event()
        self._ready = threading.Event()
        self._player = nvwave.WavePlayer(
            channels=1,
            samplesPerSec=24000,
            bitsPerSample=16,
            outputDevice=config.conf["audio"]["outputDevice"],
        )
        self._dataDir = os.path.join(globalVars.appArgs.configPath, "kokoroTTS")
        self._userVoiceDir = os.path.join(self._dataDir, "voices")
        # Building the ONNX session and warming it up costs several seconds on a
        # low-power laptop (thread pool spin-up, kernel selection, arena
        # allocation) and previously blocked NVDA's main thread at startup.
        # Voice discovery is cheap, so do that here and load the model in the
        # background; speech queued before the model is ready simply waits in
        # the synthesis worker.
        self._knownVoices = self._scanVoiceFiles(self._userVoiceDir)
        if not self._knownVoices:
            raise RuntimeError("Kokoro found no voice files")
        self._voice = self._knownVoices[0]
        self._tts: KokoroTTS | None = None
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
        self._initThread = threading.Thread(
            target=self._initTTS,
            name="KokoroInit",
            daemon=True,
        )
        self._initThread.start()
        threading.Thread(
            target=self._precacheCommonLabels,
            name="KokoroPrecache",
            daemon=True,
        ).start()

    @staticmethod
    def _scanVoiceFiles(voiceDir: str) -> list[str]:
        """Return the sorted voice ids found on disk (fast, no ONNX session needed)."""
        names = []
        try:
            for name in os.listdir(voiceDir):
                lower = name.lower()
                if lower.endswith((".npy", ".bin")):
                    names.append(os.path.splitext(name)[0])
        except OSError:
            pass
        return sorted(names)

    @staticmethod
    def _setThreadBelowNormal() -> None:
        """Lower worker thread priority so the NVDA main thread and its
        watchdog are never starved of CPU while inference runs on a low
        core-count laptop."""
        try:
            k32 = ctypes.windll.kernel32
            k32.GetCurrentThread.restype = ctypes.c_void_p
            k32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
            # THREAD_PRIORITY_BELOW_NORMAL
            k32.SetThreadPriority(k32.GetCurrentThread(), -1)
        except Exception:
            pass

    def _initTTS(self) -> None:
        self._setThreadBelowNormal()
        try:
            for attempt in range(2):
                if self._stopEvent.is_set():
                    return
                try:
                    model_path = os.path.join(self._dataDir, "model", "kokoro_quant.onnx")
                    if not os.path.isfile(model_path):
                        model_path = os.path.join(self._dataDir, "model", "kokoro.onnx")
                    tts = KokoroTTS(
                        model_path=model_path,
                        voice_dir=self._userVoiceDir,
                        config_path=os.path.join(_BASE_DIR, "config.json"),
                        tokenizer_path=os.path.join(_BASE_DIR, "tokenizer.json"),
                    )
                    if self._stopEvent.is_set():
                        return
                    voices = tts.list_voices()
                    if self._voice not in voices and voices:
                        self._voice = voices[0]
                    if self._voice in voices:
                        tts.set_voice(self._voice)
                    self._tts = tts
                    return
                except Exception:
                    # Transient failures (antivirus locking the model file,
                    # a half-written download) are common on Windows; retry once.
                    log.exception(f"Kokoro model initialization attempt {attempt + 1} failed")
                    if attempt == 0 and not self._stopEvent.is_set():
                        time.sleep(0.5)
        finally:
            self._ready.set()

    def _precacheCommonLabels(self) -> None:
        # The ONNX session is created in the background; don't run inference
        # concurrently with its own warm-up.
        self._ready.wait()
        if self._stopEvent.is_set():
            return
        # Precache only the handful of labels NVDA speaks during normal startup
        # and menu navigation. A larger list saturates the CPU of a low-power
        # laptop for many minutes (each label is ~2s of inference here) and
        # triggers NVDA's watchdog freeze recovery. Everything else is cached
        # to disk on first use and is effectively instant afterwards.
        common_labels = (
            "NVDA", "Desktop", "Taskbar", "Start menu",
            "OK", "Cancel", "Close", "Apply",
            "Settings", "Preferences", "Exit", "About",
            "View log", "Tools", "Help", "Menu",
            "File", "Edit",
        )
        for label in common_labels:
            if self._stopEvent.is_set():
                break
            # Wait until the user has no queued synthesis work AND the tail of
            # any current playback has drained, so precaching never steals CPU
            # from speech that is still being heard.
            while (
                self._isSpeakingEvent.is_set()
                or not self._speechQueue.empty()
                or not self._audioQueue.empty()
            ) and not self._stopEvent.is_set():
                time.sleep(0.2)
            if self._stopEvent.is_set() or self._tts is None:
                break
            try:
                # Serialize with the synthesis worker so the two threads never
                # run ONNX inference simultaneously.
                with self._inferenceLock:
                    self._tts.synthesize(label, speed=self._speedForRate(self._rate))
            except Exception:
                pass
            time.sleep(0.1)  # Yield CPU to keep the system responsive

    def terminate(self):
        self.cancel()
        self._stopEvent.set()
        # Release any worker blocked waiting for the model to become ready.
        self._ready.set()
        self._speechQueue.put(None)
        self._audioQueue.put(None)
        if self._synthesisWorker.is_alive():
            self._synthesisWorker.join(timeout=1.0)
        if self._playbackWorker.is_alive():
            self._playbackWorker.join(timeout=1.0)
        initThread = getattr(self, "_initThread", None)
        if initThread is not None and initThread.is_alive():
            initThread.join(timeout=1.0)
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
    def _toPcm16(audio, volume: int, trim_tail: bool = False) -> bytes:
        import numpy as np

        data = np.asarray(audio, dtype=np.float32)
        if data.ndim != 1:
            data = data.reshape(-1)
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)

        if trim_tail and len(data) > 1000:
            abs_data = np.abs(data)
            last_sound = len(data) - 1 - np.argmax(abs_data[::-1] > 0.01)
            keep_len = min(len(data), last_sound + 720)  # 30ms natural soft fade
            data = data[:keep_len]

        data = np.clip(data * (max(0, min(100, volume)) / 100.0), -1.0, 1.0)
        return (data * 32767.0).astype(np.int16).tobytes()

    @staticmethod
    def _chunkText(text: str, max_first_chars: int = 15, max_chars: int = 45) -> list[tuple[str, int, int]]:
        if not text:
            return []

        # If the text has NVDA multi-space separators (\s{2,}), split on them first (UI control focus sequences)
        if '  ' in text:
            raw_parts = re.split(r'(\s{2,})', text)
            chunks = []
            curr_pos = 0
            for part in raw_parts:
                p_strip = part.strip()
                if p_strip:
                    c_start = text.find(p_strip, curr_pos)
                    if c_start == -1:
                        c_start = curr_pos
                    c_end = c_start + len(p_strip)
                    chunks.append((p_strip, c_start, c_end))
                    curr_pos = c_end
            return chunks if chunks else [(text, 0, len(text))]

        # Split continuous sentences by major punctuation marks
        clauses = [c.strip() for c in re.split(r'([,.;!?\:\—\–\n]+)', text) if c.strip()]
        merged_clauses = []
        i = 0
        while i < len(clauses):
            clause = clauses[i]
            if i + 1 < len(clauses) and clauses[i+1] in (',', '.', ';', ':', '!', '?', '—', '–', '-'):
                clause += clauses[i+1]
                i += 2
            else:
                i += 1
            merged_clauses.append(clause)

        final_chunks = []
        curr_str = ''
        sub_start = 0
        CONJUNCTION_PATTERN = re.compile(
            r'^\b(that|which|before|after|because|although|while|where|when|since|unless|until|and|but|or|so|with|to|in|on|at|by|for|from|he|she|it|they|of)\b$',
            re.IGNORECASE,
        )

        for clause in merged_clauses:
            target_limit = max_first_chars if len(final_chunks) == 0 and not curr_str else max_chars
            if len(curr_str) + len(clause) + (1 if curr_str else 0) <= target_limit:
                curr_str += (' ' if curr_str else '') + clause
            else:
                if curr_str:
                    c_start = text.find(curr_str, sub_start)
                    if c_start == -1:
                        c_start = sub_start
                    c_end = c_start + len(curr_str)
                    final_chunks.append((curr_str, c_start, c_end))
                    sub_start = c_end
                    curr_str = ''

                curr_limit = max_first_chars if len(final_chunks) == 0 else max_chars
                if len(clause) > curr_limit:
                    words = clause.split(' ')
                    sub_words = []
                    sub_len = 0
                    for w in words:
                        clean_w = re.sub(r'[\W_]+', '', w)
                        curr_limit = max_first_chars if len(final_chunks) == 0 else max_chars
                        space_len = 1 if sub_words else 0

                        if sub_len + len(w) + space_len > curr_limit and sub_words:
                            sub_str = ' '.join(sub_words)
                            c_start = text.find(sub_str, sub_start)
                            if c_start == -1:
                                c_start = sub_start
                            c_end = c_start + len(sub_str)
                            final_chunks.append((sub_str, c_start, c_end))
                            sub_start = c_end
                            sub_words = [w]
                            sub_len = len(w)
                        elif sub_len > 25 and CONJUNCTION_PATTERN.match(clean_w):
                            sub_str = ' '.join(sub_words)
                            c_start = text.find(sub_str, sub_start)
                            if c_start == -1:
                                c_start = sub_start
                            c_end = c_start + len(sub_str)
                            final_chunks.append((sub_str, c_start, c_end))
                            sub_start = c_end
                            sub_words = [w]
                            sub_len = len(w)
                        else:
                            sub_words.append(w)
                            sub_len += len(w) + space_len
                    if sub_words:
                        sub_str = ' '.join(sub_words)
                        curr_str = sub_str
                else:
                    curr_str = clause

        if curr_str:
            c_start = text.find(curr_str, sub_start)
            if c_start == -1:
                c_start = sub_start
            c_end = c_start + len(curr_str)
            final_chunks.append((curr_str, c_start, c_end))

        return final_chunks if final_chunks else [(text, 0, len(text))]

    def _runSynthesis(self) -> None:
        self._setThreadBelowNormal()
        while not self._stopEvent.is_set():
            item = self._speechQueue.get()
            try:
                if item is None:
                    return
                # Wait for the background model load/warm-up before touching
                # the ONNX session. Speech queued in the meantime is deferred,
                # not dropped, and NVDA's main thread stays responsive.
                self._ready.wait()
                if self._stopEvent.is_set():
                    return
                if self._tts is None:
                    continue
                if item.generation != self._currentGeneration():
                    continue
                if item.voice != self._tts.current_voice:
                    self._tts.set_voice(item.voice)

                full_text = item.full_text
                spans = item.spans
                if not full_text and not spans:
                    continue

                utterance_t0 = time.perf_counter()
                self._isSpeakingEvent.set()
                text_chunks = self._chunkText(full_text)
                num_chunks = len(text_chunks)
                log.debug(f"Kokoro chunked {len(text_chunks)} parts in {time.perf_counter() - utterance_t0:.3f}s")

                for chunk_idx, (chunk_str, c_start, c_end) in enumerate(text_chunks):
                    if item.generation != self._currentGeneration() or self._stopEvent.is_set():
                        break

                    pcm_bytes = b""
                    if chunk_str and chunk_str.strip():
                        chunk_t0 = time.perf_counter()
                        try:
                            # Never run inference concurrently with the background
                            # label precacher: on a 4-core/8-thread laptop, two
                            # simultaneous 2-thread ONNX runs compete for the same
                            # cores and each becomes several times slower.
                            with self._inferenceLock:
                                audio = self._tts.synthesize(chunk_str, speed=self._speedForRate(item.rate))
                            if item.generation != self._currentGeneration() or self._stopEvent.is_set():
                                break
                            is_last = (chunk_idx == num_chunks - 1)
                            pcm_bytes = self._toPcm16(audio, item.volume, trim_tail=not is_last)
                        except Exception:
                            log.debugWarning("Kokoro skipped unsynthesizable chunk", exc_info=True)
                        log.debug(
                            f"Kokoro chunk {chunk_idx} {chunk_str!r} took {time.perf_counter() - chunk_t0:.3f}s "
                            f"({len(pcm_bytes) // 2 / 24000:.2f}s audio)"
                        )

                    if item.generation != self._currentGeneration() or self._stopEvent.is_set():
                        break

                    chunk_spans = [
                        (s_start, s_end, idx)
                        for s_start, s_end, idx in spans
                        if s_start < c_end and s_end > c_start
                    ]

                    total_chunk_chars = max(1, len(chunk_str))
                    total_pcm_len = len(pcm_bytes)

                    if pcm_bytes:
                        first_idx = chunk_spans[0][2] if chunk_spans else None
                        is_last_chunk = (chunk_idx == num_chunks - 1)
                        self._audioQueue.put(_AudioPacket(item.generation, pcm_bytes, first_idx, False, is_last_chunk))
                        # Yield CPU slice to playback worker so WASAPI sound card output starts immediately
                        time.sleep(0.015)
            except Exception:
                log.exception("Kokoro synthesis worker failed")
            finally:
                if self._speechQueue.empty():
                    self._isSpeakingEvent.clear()
                self._speechQueue.task_done()

    def _notifyIndexIfCurrent(self, generation: int, index: int) -> None:
        if generation == self._currentGeneration() and not self._stopEvent.is_set():
            synthDriverHandler.synthIndexReached.notify(synth=self, index=index)

    def _notifyDoneIfCurrent(self, generation: int, index: Optional[int]) -> None:
        if generation == self._currentGeneration() and not self._stopEvent.is_set():
            if index is not None:
                synthDriverHandler.synthIndexReached.notify(synth=self, index=index)
            synthDriverHandler.synthDoneSpeaking.notify(synth=self)

    def _runPlayback(self) -> None:
        self._setThreadBelowNormal()
        while not self._stopEvent.is_set():
            packet = self._audioQueue.get()
            try:
                if packet is None:
                    return
                if packet.generation != self._currentGeneration():
                    continue
                if packet.pcm:
                    if packet.isLastChunk:
                        self._player.feed(
                            packet.pcm,
                            onDone=lambda generation=packet.generation, index=packet.index: self._notifyDoneIfCurrent(
                                generation,
                                index,
                            ),
                        )
                    elif packet.index is not None:
                        self._player.feed(
                            packet.pcm,
                            onDone=lambda generation=packet.generation, index=packet.index: self._notifyIndexIfCurrent(
                                generation,
                                index,
                            ),
                        )
                    else:
                        self._player.feed(packet.pcm)
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
    def _parseSequence(speechSequence: Iterable[object]) -> tuple[str, tuple[tuple[int, int, int | None], ...]]:
        text_parts: list[str] = []
        spans: list[tuple[int, int, int | None]] = []
        pending_index: int | None = None
        start_char: int = 0

        for item in speechSequence:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, speech.commands.IndexCommand):
                current_text = "".join(text_parts)
                end_char = len(current_text)
                if end_char > start_char or pending_index is not None:
                    spans.append((start_char, end_char, pending_index))
                    start_char = end_char
                pending_index = item.index

        current_text = "".join(text_parts)
        end_char = len(current_text)
        if end_char > start_char or pending_index is not None:
            spans.append((start_char, end_char, pending_index))

        return current_text, tuple(spans)

    def speak(self, speechSequence):
        full_text, spans = self._parseSequence(speechSequence)
        if not full_text and not spans:
            return
        # Signal real speech immediately so the background label precacher stops
        # starting new inferences instead of competing with the user's read.
        self._isSpeakingEvent.set()
        generation = self._currentGeneration()
        self._speechQueue.put(_Utterance(full_text, spans, self._rate, self._volume, self._voice, generation))

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
        self._knownVoices = self._scanVoiceFiles(self._userVoiceDir)
        if self._tts is not None:
            voices = self._tts.reload_voices()
            self._knownVoices = list(voices)
            if self._voice not in voices and voices:
                self.cancel()
                self._voice = voices[0]
                self._tts.set_voice(self._voice)
        elif self._voice not in self._knownVoices and self._knownVoices:
            self.cancel()
            self._voice = self._knownVoices[0]
        return self._knownVoices

    def _getAvailableVoices(self):
        voices = OrderedDict()
        for voiceId in self._knownVoices:
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
        if value not in self._knownVoices:
            raise LookupError(f"Unknown Kokoro voice: {value}")
        if value == self._voice:
            return
        self.cancel()
        self._voice = value
        if self._tts is not None:
            self._tts.set_voice(value)

    def _get_rate(self):
        return self._rate

    def _set_rate(self, value):
        self._rate = max(0, min(100, int(value)))

    def _get_volume(self):
        return self._volume

    def _set_volume(self, value):
        self._volume = max(0, min(100, int(value)))
