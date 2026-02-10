# Raspberry Pi 5 TUI Plan (CripIt Headless)

This document describes a practical, Pi-friendly plan to run CripIt as a terminal UI (TUI) on Raspberry Pi OS 64-bit, CPU-only, using the existing pipeline and `pywhispercpp` (which builds whisper.cpp under the hood).

Goals
- Run on Raspberry Pi 5 without a desktop session.
- Keep changes minimal: reuse `core/` modules and add a new headless entrypoint.
- Provide a curses-based terminal UI with live status and streaming transcript output.
- Prefer WebRTC VAD; guarantee segmentation with a lightweight energy VAD fallback.
- English-first defaults for performance (`base.en` / `small.en`).

Non-goals
- GPU/CUDA support (Pi is CPU-only for this project).
- Replacing `pywhispercpp` with direct whisper.cpp CLI (can be added later as an alternate backend).

Overview
- New entrypoint: `terminal_app.py` (no PyQt imports).
- Audio input: `sounddevice` backend (PortAudio).
- VAD/segmentation: WebRTC VAD when available; else energy VAD fallback.
- Durability: keep the disk-backed spool/pipeline so capture never silently drops segments.
- UI: curses main loop in the main thread; worker threads communicate via an event queue.

Architecture

Data flow (same as GUI, headless)
1. `AudioCapture` listens on the microphone.
2. VAD segments audio into finalized recordings (`FinalizedRecording`).
3. Each finalized recording is written to the spool (`RecordingSpool.enqueue`).
4. The pipeline (`TranscriptionPipeline`) consumes spool jobs in order and calls `Transcriber.transcribe`.
5. On success, jobs are deleted; on failure, jobs are moved to `output/spool/failed/` with an error file.
6. TUI displays status + prints each completed segment.

Threading model
- Main thread: curses UI loop (rendering and key handling).
- Audio callback thread (PortAudio / sounddevice): VAD decisions + buffering.
- Pipeline worker thread: sequential transcription.

Important rule: curses is not thread-safe.
- No curses calls from audio/pipeline threads.
- Use a `queue.Queue()` of events (state updates, VAD state, job started/done/failed, backlog updates) consumed by the main thread.

Planned repository changes

1) Add `terminal_app.py` (new)
- Implements CLI + curses TUI.
- Reuses existing `core/` modules.

2) Add `EnergyVAD` fallback (small patch)
- Location: `core/audio_capture.py`.
- Reason: if WebRTC VAD is not installable on the Pi, CripIt currently has no reliable segmentation.
- Behavior:
  - Prefer WebRTC VAD when installed.
  - Else use EnergyVAD (pure numpy RMS + hysteresis).

3) Add `requirements-cli.txt` (new)
- Minimal dependencies for headless use (no `PyQt6`).

4) Optional docs
- Add a short Pi section to `README.md` or link to this plan.

Terminal UI (curses)

Display layout
- Header (one line):
  - Model: `base.en` (or chosen)
  - Device: `CPU`
  - State: `LISTENING/RECORDING/PROCESSING`
  - VAD: `speech/silence`
  - Backlog: `queued=<n> processing=<0|1>`
  - Perf: last segment duration + average RTF (from `Transcriber.get_stats()`)

- Transcript window (scrolling):
  - Appends each completed transcription segment.
  - Optionally prefixes with job seq and timestamp.

- Footer (one line):
  - Controls: `q quit | r toggle rec | c clear screen | s save toggle (optional)`
  - Note: application starts recording immediately by default.

Input handling
- Non-blocking getch with periodic redraw (10-20 Hz).
- Keybinds (minimum viable):
  - `q`: quit cleanly (stop audio, stop pipeline)
  - `r`: toggle recording on/off
  - `c`: clear transcript display (does not delete spool)

Logging considerations
- Avoid writing logs to stdout while curses is active.
- Suggested: log to a file under `output/terminal.log` or lower log level.
- Option: `--verbose` enables debug logging to a file.

CLI (terminal_app.py)

Basic usage
- Start (auto-record):
  - `python terminal_app.py`
- Choose model:
  - `python terminal_app.py --model base.en`
  - `python terminal_app.py --model small.en`
- Set mic device:
  - `python terminal_app.py --list-devices`
  - `python terminal_app.py --device-index 2`

Model management (no GUI)
- List models:
  - `python terminal_app.py --list-models`
- Download:
  - `python terminal_app.py --download base.en`

Planned flags
- `--model` (default: `base.en`)
- `--language` (default: `en`)
- `--threads` (default: `4`)
- `--vad` (`webrtc` or `energy`; default `webrtc` with fallback)
- `--device-index` (default: system default)
- `--gain-db` (default: 0)
- `--spool-dir` (default: `output/spool`)
- `--output-file` (optional append-only transcript file)
- `--no-autostart` (optional; if present, start idle and require `r`)
- `--list-devices`
- `--list-models`
- `--download MODEL`

Defaults for Pi performance
- `--model base.en` (fast and workable on Pi)
- `--threads 4` (Pi 5 has 4 performance cores; start here)
- `--language en` (avoid auto-detect overhead)

Voice Activity Detection (VAD)

WebRTC VAD (preferred)
- Package: `webrtcvad` or `webrtcvad-wheels`.
- Pros: robust speech detection, low CPU.
- Cons: dependency availability can be annoying on ARM in some setups.

Energy VAD (fallback)
- Pure numpy.
- Uses RMS/energy thresholding with hysteresis.
- Pros: always installable, simple.
- Cons: less robust in noisy environments; thresholds may need tuning.

Energy VAD tuning
- Parameters to expose (later if needed):
  - `start_threshold` (RMS)
  - `stop_threshold` (RMS)
  - `hangover_frames`
- Initial thresholds should be conservative; allow quick adjustments via CLI.

Raspberry Pi OS setup

System packages
On Raspberry Pi OS (Debian-based) 64-bit:

```bash
sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-dev \
  build-essential cmake git \
  portaudio19-dev \
  libopenblas-dev

# If curses is missing (rare on full installs)
sudo apt-get install -y python3-curses
```

Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
```

Install CLI dependencies

```bash
pip install -r requirements-cli.txt
```

Building `pywhispercpp` from source on Pi
- On ARM, `pywhispercpp` may compile native code during install.
- If you need a source build explicitly:

```bash
pip uninstall -y pywhispercpp

# build from PyPI source (or replace with a git URL if desired)
pip install --no-binary=:all: --no-cache-dir pywhispercpp
```

Notes
- Keep CUDA-related env vars unset on Pi (no CUDA).
- If builds fail, capture the output and check missing system libs.

Model selection on Pi 5

Recommended English models
- `base.en`: best default for speed.
- `small.en`: better accuracy, heavier CPU; test and decide.

Download example

```bash
python terminal_app.py --download base.en
python terminal_app.py --model base.en
```

Spool behavior on Pi
- Default spool dir: `output/spool`.
- Jobs are deleted on success; failures are kept under `output/spool/failed`.
- If disk gets low, recording should stop to avoid silent drops.

Testing and validation

Quick checks
1) Verify microphone devices:

```bash
python terminal_app.py --list-devices
```

2) Download model:

```bash
python terminal_app.py --download base.en
```

3) Run TUI:

```bash
python terminal_app.py --model base.en --threads 4
```

Functional acceptance criteria
- Starts and begins listening immediately.
- When speaking, the VAD transitions to RECORDING.
- On silence timeout, a segment finalizes; a job appears in spool.
- Pipeline processes jobs sequentially; transcript appears in the TUI.
- Ctrl+C or `q` quits cleanly (no corrupt terminal state).

Performance measurements
- Observe average RTF (real-time factor) from `Transcriber.get_stats()`.
- Target: RTF near or below ~1.0 for interactive use (depends heavily on model and audio segment sizes).

Failure modes and handling

Common problems
- No audio devices:
  - Ensure `portaudio19-dev` is installed and `sounddevice` imports.
- WebRTC VAD missing:
  - Should automatically fall back to EnergyVAD.
- Model too slow:
  - Use `base.en` or `tiny.en`, reduce `--threads`, shorten segment duration.
- Build failures for `pywhispercpp`:
  - Install build tools and try source build; consider pinning versions.

Terminal corruption after crash
- Always restore terminal state in a `finally` block:
  - `curses.nocbreak()`, `curses.echo()`, `curses.endwin()`

Milestones

Milestone 1: Minimal TUI runner
- `terminal_app.py` starts immediately, prints transcript, basic status line.

Milestone 2: Robust VAD fallback
- Add `EnergyVAD` and ensure segmentation without WebRTC.

Milestone 3: Quality-of-life
- Device listing, model download/list, output file append, better status, resize handling.

Milestone 4: Pi tuning
- Document best-known settings and basic troubleshooting.
