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

HF_BASE = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main"
MODEL_URL = f"{HF_BASE}/onnx/model.onnx?download=true"

# The supported full-precision model is approximately 326 MB. A generous
# lower bound rejects every smaller model while tolerating minor upstream
# packaging changes.
MIN_FULL_MODEL_BYTES = 250 * 1024 * 1024

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
        return os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) >= MIN_FULL_MODEL_BYTES
    except OSError:
        return False


def _valid_voice(voice_id: str) -> bool:
    path = _voice_path(voice_id)
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    return size >= (256 * 4) and size % (256 * 4) == 0


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
        super().__init__(parent, title="Installing Kokoro TTS", size=(560, 190))
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.status = wx.StaticText(
            panel,
            label="Preparing the supported Kokoro model and voices.",
        )
        self.status.SetName("Kokoro installation status")
        sizer.Add(self.status, 0, wx.EXPAND | wx.ALL, 12)

        self.gauge = wx.Gauge(panel, range=100)
        self.gauge.SetName("Kokoro installation progress")
        sizer.Add(self.gauge, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        explanation = wx.StaticText(
            panel,
            label=(
                "The full-precision model is installed once because smaller "
                "models are not suitable for reliable screen-reader speech."
            ),
        )
        explanation.Wrap(520)
        sizer.Add(explanation, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        panel.SetSizer(sizer)
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
        missing_voices = [voice for voice in VOICE_IDS if not _valid_voice(voice)]
        if _valid_model() and not missing_voices:
            log.info("Kokoro one-time installation is complete")
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
                    "Downloading the supported full-precision Kokoro model.",
                    0,
                )

                def model_progress(done: int, total: int) -> None:
                    percent = int(done * 100 / total)
                    wx.CallAfter(
                        self._set_status,
                        f"Downloading Kokoro model: {percent} percent.",
                        percent,
                    )

                _download(MODEL_URL, MODEL_PATH, model_progress)
                if not _valid_model():
                    raise RuntimeError(
                        "The downloaded Kokoro model is not the supported full-precision model."
                    )

            missing_voices = [voice for voice in VOICE_IDS if not _valid_voice(voice)]
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
                "NVDA will try again the next time it starts."
            ),
            "Kokoro TTS",
            wx.OK | wx.ICON_ERROR,
        )

    def terminate(self):
        self._terminating = True
        self._close_dialog()
        super().terminate()
