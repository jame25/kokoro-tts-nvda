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
        self._playerLock = threading.Lock()
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
        self._disk_cache_dir = os.path.join(self._dataDir, "model", "cache")
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

    def _getFromFastDiskCache(self, text: str, voice: str, volume: int) -> bytes | None:
        """Instant Tier-0 cache lookup directly from disk without requiring ONNX model initialization."""
        if not self._disk_cache_dir or not os.path.isdir(self._disk_cache_dir):
            return None
        clean_text = text.strip()
        norm_key = re.sub(r'[\.\…\:\,\;\!\?\/]+', ' ', clean_text.lower())
        norm_key = re.sub(r'\s+', ' ', norm_key).strip()
        if not norm_key:
            norm_key = clean_text.lower()
        if not norm_key or len(norm_key) > 300:
            return None
        disk_key = f"{norm_key}:{voice}"
        try:
            key_hash = hashlib.md5(disk_key.encode("utf-8")).hexdigest()
            file_path = os.path.join(self._disk_cache_dir, f"{key_hash}.bin")
            if os.path.isfile(file_path):
                with open(file_path, "rb") as f:
                    raw_pcm = f.read()
                if raw_pcm:
                    if volume == 100:
                        return raw_pcm
                    import numpy as np
                    data = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32)
                    data = np.clip(data * (max(0, min(100, volume)) / 100.0), -32767.0, 32767.0)
                    return data.astype(np.int16).tobytes()
        except Exception:
            pass
        return None

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
    def _setThreadPriority(priority: int) -> None:
        """Set thread priority for current thread (-1 = BELOW_NORMAL, 0 = NORMAL)."""
        try:
            k32 = ctypes.windll.kernel32
            k32.GetCurrentThread.restype = ctypes.c_void_p
            k32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
            k32.SetThreadPriority(k32.GetCurrentThread(), priority)
        except Exception:
            pass

    @classmethod
    def _setThreadBelowNormal(cls) -> None:
        cls._setThreadPriority(-1)

    @classmethod
    def _setThreadNormal(cls) -> None:
        cls._setThreadPriority(0)

    def _initTTS(self) -> None:
        self._setThreadBelowNormal()
        init_t0 = time.perf_counter()
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
                    # Warm up ONNX Runtime lazily inside the background init thread before ready signal
                    try:
                        tts._warmup()
                    except Exception:
                        log.debugWarning("Kokoro background warm-up failed", exc_info=True)
                    self._tts = tts
                    log.info(f"Kokoro model loaded and warmed in {time.perf_counter() - init_t0:.2f}s")
                    return
                except Exception:
                    # Transient failures (antivirus locking the model file,
                    # a half-written download) are common on Windows; retry once.
                    log.exception(f"Kokoro model initialization attempt {attempt + 1} failed")
                    if attempt == 0 and not self._stopEvent.is_set():
                        time.sleep(0.5)
        finally:
            self._ready.set()

    @staticmethod
    def _getDesktopShortcutLabels() -> list[str]:
        labels = []
        try:
            desktop_dirs = [
                os.path.join(os.path.expanduser("~"), "Desktop"),
                os.path.join(os.environ.get("PUBLIC", r"C:\Users\Public"), "Desktop"),
            ]
            for d in desktop_dirs:
                if os.path.isdir(d):
                    for f in os.listdir(d):
                        if f.lower().endswith(".lnk"):
                            name = os.path.splitext(f)[0].strip()
                            if name and name not in labels:
                                labels.append(name)
        except Exception:
            pass
        return labels

    def _precacheCommonLabels(self) -> None:
        # Wait for model ready, then wait 10s so NVDA startup completes completely
        self._ready.wait()
        for _ in range(100):
            if self._stopEvent.is_set():
                return
            time.sleep(0.1)

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
            # Wait until the user has no queued synthesis work AND playback is idle
            while (
                self._isSpeakingEvent.is_set()
                or not self._speechQueue.empty()
                or not self._audioQueue.empty()
            ) and not self._stopEvent.is_set():
                time.sleep(0.5)
            if self._stopEvent.is_set() or self._tts is None:
                break
            if self._isSpeakingEvent.is_set() or not self._speechQueue.empty():
                continue
            try:
                # Serialize with the synthesis worker so the two threads never
                # run ONNX inference simultaneously.
                with self._inferenceLock:
                    if not self._isSpeakingEvent.is_set() and self._speechQueue.empty():
                        self._tts.synthesize(label, speed=self._speedForRate(self._rate))
            except Exception:
                pass
            time.sleep(1.0)  # Generous yield so CPU stays cool and watchdog is never triggered

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
            with self._playerLock:
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
    def _toPcm16(audio, volume: int, trim_leading: bool = False, trim_tail: bool = False) -> bytes:
        import numpy as np

        data = np.asarray(audio, dtype=np.float32)
        if data.ndim != 1:
            data = data.reshape(-1)
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)

        if len(data) > 480:
            abs_data = np.abs(data)
            above_thresh = abs_data > 0.008

            if np.any(above_thresh):
                first_sound = int(np.argmax(above_thresh))
                last_sound = len(data) - 1 - int(np.argmax(above_thresh[::-1]))

                start_pos = max(0, first_sound - 240) if trim_leading else 0
                end_pos = min(len(data), last_sound + (240 if trim_tail else 720))

                if start_pos < end_pos:
                    data = data[start_pos:end_pos]

        data = np.clip(data * (max(0, min(100, volume)) / 100.0), -1.0, 1.0)
        return (data * 32767.0).astype(np.int16).tobytes()

    @staticmethod
    def _chunkText(text: str, max_subsequent_chars: int = 160) -> list[tuple[str, int, int]]:
        if not text:
            return []

        clean_stripped = text.strip()

        # Split by natural punctuation marks (, . ; : ! ? — – \n \t) or multi-space delimiters
        clauses = [c.strip() for c in re.split(r'([,.;!?:—–\n\t]+|\s{2,})', text) if c.strip()]
        merged_clauses = []
        i = 0
        while i < len(clauses):
            c = clauses[i]
            if i + 1 < len(clauses) and clauses[i+1] in (',', '.', ';', ':', '!', '?', '—', '–'):
                c += clauses[i+1]
                i += 2
            else:
                i += 1
            merged_clauses.append(c)

        if not merged_clauses:
            return [(clean_stripped, 0, len(text))]

        final_chunks = []
        sub_start = 0

        # Chunk 0: The first natural clause ending at the first comma (,) or full stop (.) / punctuation mark!
        first_clause = merged_clauses[0]
        CONJUNCTION_PATTERN = re.compile(
            r'^\b(that|which|before|after|because|although|while|where|when|since|unless|until|and|but|or|so|with|to|in|on|at|by|for|from|including|instead|without)\b$',
            re.IGNORECASE,
        )

        # If the first clause is extraordinarily long (> 160 chars with no punctuation), split at a conjunction
        if len(first_clause) > max_subsequent_chars:
            words = first_clause.split(' ')
            sub_words = []
            sub_len = 0
            for w in words:
                clean_w = re.sub(r'[\W_]+', '', w)
                space_len = 1 if sub_words else 0
                if sub_len >= 30 and CONJUNCTION_PATTERN.match(clean_w):
                    break
                sub_words.append(w)
                sub_len += len(w) + space_len
            c0_str = ' '.join(sub_words) if sub_words else first_clause
            rem_first = first_clause[len(c0_str):].strip()
            subsequent_clauses = ([rem_first] if rem_first else []) + merged_clauses[1:]
        else:
            c0_str = first_clause
            subsequent_clauses = merged_clauses[1:]

        c_start = text.find(c0_str, sub_start)
        if c_start == -1:
            c_start = sub_start
        c_end = c_start + len(c0_str)
        final_chunks.append((c0_str, c_start, c_end))
        sub_start = c_end

        # Group subsequent clauses into full, continuous compound sentences (up to 160 chars)
        curr_str = ''
        for cl in subsequent_clauses:
            if len(curr_str) + len(cl) + (1 if curr_str else 0) <= max_subsequent_chars:
                curr_str += (' ' if curr_str else '') + cl
            else:
                if curr_str:
                    c_start = text.find(curr_str, sub_start)
                    if c_start == -1:
                        c_start = sub_start
                    c_end = c_start + len(curr_str)
                    final_chunks.append((curr_str, c_start, c_end))
                    sub_start = c_end
                    curr_str = ''
                curr_str = cl

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
                if item.generation != self._currentGeneration():
                    continue

                full_text = item.full_text
                spans = item.spans
                if not full_text and not spans:
                    continue

                # Tier-0 Fast Disk Cache Check:
                # If this UI phrase is already cached on disk (e.g. tray menu, desktop icons),
                # emit audio IMMEDIATELY without waiting for the ONNX model to finish loading!
                fast_pcm = self._getFromFastDiskCache(full_text, item.voice, item.volume)
                if fast_pcm is not None:
                    if item.generation == self._currentGeneration():
                        self._isSpeakingEvent.set()
                        first_idx = spans[0][2] if spans else None
                        self._audioQueue.put(_AudioPacket(item.generation, fast_pcm, first_idx, False, True))
                    continue

                wait_t0 = time.perf_counter()
                self._ready.wait()
                log.debug(f"Kokoro synthesis waited {time.perf_counter() - wait_t0:.2f}s for model ready")
                if self._stopEvent.is_set():
                    return
                if self._tts is None:
                    continue
                if item.generation != self._currentGeneration():
                    continue
                if item.voice != self._tts.current_voice:
                    self._tts.set_voice(item.voice)

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
                        # Check fast disk cache for this chunk
                        chunk_fast_pcm = self._getFromFastDiskCache(chunk_str, item.voice, item.volume)
                        if chunk_fast_pcm is not None:
                            pcm_bytes = chunk_fast_pcm
                        else:
                            chunk_t0 = time.perf_counter()
                            try:
                                self._setThreadNormal()
                                with self._inferenceLock:
                                    if item.generation != self._currentGeneration() or self._stopEvent.is_set():
                                        break
                                    audio = self._tts.synthesize(chunk_str, speed=self._speedForRate(item.rate))
                                if item.generation != self._currentGeneration() or self._stopEvent.is_set():
                                    break
                                is_first = (chunk_idx == 0)
                                is_last = (chunk_idx == num_chunks - 1)
                                pcm_bytes = self._toPcm16(audio, item.volume, trim_leading=not is_first, trim_tail=not is_last)
                            except Exception:
                                log.debugWarning("Kokoro skipped unsynthesizable chunk", exc_info=True)
                            finally:
                                self._setThreadBelowNormal()
                            elapsed = time.perf_counter() - chunk_t0
                            log.debug(
                                f"Kokoro chunk {chunk_idx} {chunk_str!r} took {elapsed:.3f}s "
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
                    is_last_chunk = (chunk_idx == num_chunks - 1)

                    if (pcm_bytes or is_last_chunk) and item.generation == self._currentGeneration():
                        first_idx = chunk_spans[0][2] if chunk_spans else None
                        self._audioQueue.put(_AudioPacket(item.generation, pcm_bytes if pcm_bytes else None, first_idx, False, is_last_chunk))
                        # Yield CPU slice to playback worker so WASAPI sound card output starts immediately
                        time.sleep(0.015)
            except Exception:
                log.exception("Kokoro synthesis worker failed")
            finally:
                if self._speechQueue.empty():
                    self._isSpeakingEvent.clear()
                    self._setThreadBelowNormal()
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
        CHUNK_SIZE = 4800  # 100ms at 24kHz 16-bit mono
        while not self._stopEvent.is_set():
            packet = self._audioQueue.get()
            try:
                if packet is None:
                    return
                if packet.generation != self._currentGeneration():
                    if packet.isLastChunk:
                        self._notifyDoneIfCurrent(packet.generation, packet.index)
                    continue
                if packet.pcm:
                    pcm = packet.pcm
                    total_len = len(pcm)
                    offset = 0
                    while offset < total_len and not self._stopEvent.is_set():
                        if packet.generation != self._currentGeneration():
                            if packet.isLastChunk:
                                self._notifyDoneIfCurrent(packet.generation, packet.index)
                            break

                        chunk = pcm[offset : offset + CHUNK_SIZE]
                        offset += len(chunk)
                        is_final_slice = (offset >= total_len) and packet.isLastChunk

                        with self._playerLock:
                            if packet.generation != self._currentGeneration():
                                if packet.isLastChunk:
                                    self._notifyDoneIfCurrent(packet.generation, packet.index)
                                break
                            try:
                                if is_final_slice:
                                    self._player.feed(
                                        chunk,
                                        onDone=lambda generation=packet.generation, index=packet.index: self._notifyDoneIfCurrent(
                                            generation,
                                            index,
                                        ),
                                    )
                                elif packet.index is not None and is_final_slice:
                                    self._player.feed(
                                        chunk,
                                        onDone=lambda generation=packet.generation, index=packet.index: self._notifyIndexIfCurrent(
                                            generation,
                                            index,
                                        ),
                                    )
                                else:
                                    self._player.feed(chunk)
                            except Exception:
                                log.exception("Kokoro WavePlayer feed failed, recreating player handle")
                                try:
                                    self._player.close()
                                except Exception:
                                    pass
                                self._player = nvwave.WavePlayer(
                                    channels=1,
                                    samplesPerSec=24000,
                                    bitsPerSample=16,
                                    outputDevice=config.conf["audio"]["outputDevice"],
                                )
                                if packet.generation == self._currentGeneration():
                                    if is_final_slice:
                                        self._player.feed(
                                            chunk,
                                            onDone=lambda generation=packet.generation, index=packet.index: self._notifyDoneIfCurrent(
                                                generation,
                                                index,
                                            ),
                                        )
                                    else:
                                        self._player.feed(chunk)

                        # Pace audio feed so hardware buffer holds at most ~100ms ahead,
                        # allowing instant (<50ms) abrupt cancellation on cursor movement.
                        if offset < total_len:
                            target_sleep = len(chunk) / 48000.0 * 0.70
                            sleep_end = time.perf_counter() + target_sleep
                            while time.perf_counter() < sleep_end and not self._stopEvent.is_set():
                                if packet.generation != self._currentGeneration():
                                    break
                                time.sleep(0.015)
                else:
                    if packet.isLastChunk:
                        self._notifyDoneIfCurrent(packet.generation, packet.index)
                    elif packet.index is not None:
                        self._notifyIndexIfCurrent(packet.generation, packet.index)
            except Exception:
                log.exception("Kokoro playback worker failed")
                if packet is not None and packet.generation == self._currentGeneration():
                    try:
                        with self._playerLock:
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
        log.debug(f"Kokoro speak() queued {len(full_text)} chars: {full_text[:60]!r}")
        # Signal real speech immediately so the background label precacher stops
        # starting new inferences instead of competing with the user's read.
        self._isSpeakingEvent.set()
        generation = self._currentGeneration()
        self._speechQueue.put(_Utterance(full_text, spans, self._rate, self._volume, self._voice, generation))

    def cancel(self):
        with self._playerLock:
            self._nextGeneration()
            self._discardQueue(self._speechQueue)
            self._discardQueue(self._audioQueue)
            self._isSpeakingEvent.clear()
            try:
                self._player.stop()
            except Exception:
                log.debugWarning("Kokoro could not stop audio", exc_info=True)

    def pause(self, switch):
        with self._playerLock:
            try:
                self._player.pause(bool(switch))
            except Exception:
                pass

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
