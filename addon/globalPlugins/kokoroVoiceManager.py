from __future__ import annotations

import os
import threading
import urllib.request

import addonHandler
import globalPluginHandler
import globalVars
import gui
import wx
from logHandler import log

addonHandler.initTranslation()

DATA_DIR = os.path.join(globalVars.appArgs.configPath, "kokoroTTS")
MODEL_DIR = os.path.join(DATA_DIR, "model")
VOICE_DIR = os.path.join(DATA_DIR, "voices")
MODEL_PATH = os.path.join(MODEL_DIR, "kokoro.onnx")
QUANT_MODEL_PATH = os.path.join(MODEL_DIR, "kokoro_quant.onnx")

HF_BASE = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main"
QUANT_MODEL_URL = f"{HF_BASE}/onnx/model_quantized.onnx?download=true"
MODEL_URL = f"{HF_BASE}/onnx/model.onnx?download=true"

MIN_QUANT_MODEL_BYTES = 75 * 1024 * 1024
MIN_FULL_MODEL_BYTES = 250 * 1024 * 1024

# Approximate download sizes shown in the consent prompt (decimal megabytes).
# Each voice bank is 510 x 256 float32 values (522,240 bytes).
APPROX_QUANT_MODEL_MB = 92.0
APPROX_VOICE_MB = 0.52

VOICE_IDS = (
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
)


def _voice_path(voice_id: str) -> str:
    return os.path.join(VOICE_DIR, voice_id + ".bin")


def _valid_model() -> bool:
    try:
        return os.path.isfile(QUANT_MODEL_PATH) and os.path.getsize(QUANT_MODEL_PATH) >= MIN_QUANT_MODEL_BYTES
    except OSError:
        return False


def _valid_voice(voice_id: str) -> bool:
    path = _voice_path(voice_id)
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    return size >= (256 * 4) and size % (256 * 4) == 0


def _missing_voices() -> list[str]:
    return [voice for voice in VOICE_IDS if not _valid_voice(voice)]


def _format_mb(megabytes: float) -> str:
    return f"about {max(1, round(megabytes))} MB"


def _consent_message(need_model: bool, missing_voices: list[str]) -> str:
    items = []
    total_mb = 0.0
    if need_model:
        items.append(f"- The Kokoro speech model ({_format_mb(APPROX_QUANT_MODEL_MB)})")
        total_mb += APPROX_QUANT_MODEL_MB
    if missing_voices:
        voices_mb = len(missing_voices) * APPROX_VOICE_MB
        noun = "voice" if len(missing_voices) == 1 else "voices"
        items.append(f"- {len(missing_voices)} {noun} ({_format_mb(voices_mb)})")
        total_mb += voices_mb
    return (
        "Kokoro TTS needs to download the following files before it can speak:\n\n"
        + "\n".join(items)
        + f"\n\nTotal download: {_format_mb(total_mb)} from Hugging Face (huggingface.co). "
        "The files are saved in your NVDA configuration folder.\n\n"
        "Do you want to download them now?\n\n"
        "If you choose No, nothing will be downloaded. "
        "You will be asked again the next time NVDA starts."
    )


def _download(url: str, target: str, progress_callback=None) -> None:
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temporary = target + ".download"
    try:
        def report_hook(block_count, block_size, total_size):
            if progress_callback is not None and total_size > 0:
                completed = min(block_count * block_size, total_size)
                progress_callback(completed, total_size)

        urllib.request.urlretrieve(url, temporary, reporthook=report_hook)
        if not os.path.isfile(temporary) or os.path.getsize(temporary) <= 0:
            raise RuntimeError("The downloaded file is empty")
        os.replace(temporary, target)
    finally:
        try:
            if os.path.isfile(temporary):
                os.remove(temporary)
        except OSError:
            pass


class _InstallProgressDialog(wx.Dialog):
    def __init__(self, parent):
        super().__init__(parent, title="Installing Kokoro TTS", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.status = wx.StaticText(
            panel,
            label="Preparing the Kokoro model and voices.",
        )
        self.status.SetName("Kokoro installation status")
        sizer.Add(self.status, 0, wx.EXPAND | wx.ALL, 12)

        self.gauge = wx.Gauge(panel, range=100)
        self.gauge.SetName("Kokoro installation progress")
        sizer.Add(self.gauge, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        explanation = wx.StaticText(
            panel,
            label=(
                "The high-performance Kokoro model and voice banks are installed "
                "once for low-latency screen-reader speech."
            ),
        )
        explanation.Wrap(520)
        sizer.Add(explanation, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        panel.SetSizer(sizer)
        sizer.Fit(self)
        self.SetMinSize((580, 220))
        self.Fit()
        self.CentreOnScreen()

    def update_status(self, message: str, percent: int | None = None) -> None:
        self.status.SetLabel(message)
        if percent is None:
            self.gauge.Pulse()
        else:
            self.gauge.SetValue(max(0, min(100, int(percent))))


class GlobalPlugin(globalPluginHandler.GlobalPlugin):
    def __init__(self):
        super().__init__()
        self._dialog = None
        self._worker = None
        self._terminating = False
        wx.CallAfter(self._start_if_needed)

    def _start_if_needed(self) -> None:
        if self._terminating:
            return
        need_model = not _valid_model()
        missing_voices = _missing_voices()
        if not need_model and not missing_voices:
            log.info("Kokoro one-time installation is complete")
            return
        if globalVars.appArgs.secure:
            # Never prompt for, or start, a download on a secure screen.
            log.info("Kokoro model or voices missing; skipping download prompt on secure screen")
            return

        # Ask before downloading anything (GitHub issue #2).
        answer = gui.messageBox(
            _consent_message(need_model, missing_voices),
            "Kokoro TTS: download model and voices?",
            wx.YES_NO | wx.ICON_QUESTION,
        )
        if self._terminating:
            return
        if answer != wx.YES:
            log.info("Kokoro model/voice download declined by the user; will ask again on next NVDA start")
            return

        self._dialog = _InstallProgressDialog(gui.mainFrame)
        self._dialog.Show()
        self._dialog.Raise()
        self._worker = threading.Thread(
            target=self._install_worker,
            name="KokoroAutomaticInstaller",
            daemon=True,
        )
        self._worker.start()

    def _set_status(self, message: str, percent: int | None = None) -> None:
        if self._dialog is not None and not self._terminating:
            self._dialog.update_status(message, percent)

    def _install_worker(self) -> None:
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            os.makedirs(VOICE_DIR, exist_ok=True)

            if not _valid_model():
                wx.CallAfter(
                    self._set_status,
                    "Downloading the high-performance Kokoro model.",
                    0,
                )

                def model_progress(done: int, total: int) -> None:
                    percent = int(done * 100 / total)
                    wx.CallAfter(
                        self._set_status,
                        f"Downloading Kokoro model: {percent} percent.",
                        percent,
                    )

                _download(QUANT_MODEL_URL, QUANT_MODEL_PATH, model_progress)
                if not _valid_model():
                    raise RuntimeError(
                        "The downloaded Kokoro model file is invalid or incomplete."
                    )

            missing_voices = _missing_voices()
            total_voices = len(missing_voices)
            for number, voice_id in enumerate(missing_voices, 1):
                display_name = voice_id.partition("_")[2].replace("_", " ").title()
                wx.CallAfter(
                    self._set_status,
                    f"Downloading voice {number} of {total_voices}: {display_name}.",
                    int((number - 1) * 100 / max(1, total_voices)),
                )
                target = _voice_path(voice_id)
                _download(
                    f"{HF_BASE}/voices/{voice_id}.bin?download=true",
                    target,
                )
                if not _valid_voice(voice_id):
                    try:
                        os.remove(target)
                    except OSError:
                        pass
                    raise RuntimeError(f"The downloaded voice {display_name} is invalid.")

            wx.CallAfter(self._installation_complete)
        except Exception as error:
            log.exception("Automatic Kokoro installation failed")
            wx.CallAfter(self._installation_failed, str(error))

    def _close_dialog(self) -> None:
        if self._dialog is not None:
            try:
                self._dialog.Destroy()
            except Exception:
                pass
            self._dialog = None

    def _installation_complete(self) -> None:
        self._close_dialog()
        gui.messageBox(
            (
                "The supported Kokoro model and all voices were installed successfully.\n\n"
                "Restart NVDA once, then select Kokoro TTS from the synthesizer list."
            ),
            "Kokoro TTS",
            wx.OK | wx.ICON_INFORMATION,
        )

    def _installation_failed(self, error: str) -> None:
        self._close_dialog()
        gui.messageBox(
            (
                "Kokoro could not finish installing its model and voices.\n\n"
                f"{error}\n\n"
                "You will be asked again the next time NVDA starts."
            ),
            "Kokoro TTS",
            wx.OK | wx.ICON_ERROR,
        )

    def terminate(self):
        self._terminating = True
        self._close_dialog()
        super().terminate()
