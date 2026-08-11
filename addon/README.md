# NVDA add-on source

This directory contains the NVDA 2026-compatible driver and the automatic
model/voice installer. It deliberately excludes generated and third-party
binary files.

A distributable package must also place these runtime components under
`synthDrivers/kokoro`:

- Python dependencies in `deps`, including NumPy and ONNX Runtime builds
  compatible with NVDA's embedded Python;
- `espeak/espeak-ng.exe`, `espeak/libespeak-ng.dll`, and the matching
  `espeak/espeak-ng-data` directory.

The installer downloads the full-precision Kokoro ONNX model and voice banks
to the user's NVDA configuration directory on first use. Those large model
assets are not part of the add-on package.

The driver uses separate synthesis and playback workers, supports NVDA index
and completion notifications, and invalidates queued work during cancellation.
It has been exercised with NVDA 2026, including speech, cancellation, voice
changes, and synthesizer switching.
