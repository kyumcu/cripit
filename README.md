# Real-Time Speech-to-Text Application
## Project: CripIt - Voice to Text with Whisper.cpp

---

## 📋 Overview

A PyQt6-based real-time speech-to-text application using whisper.cpp via `pywhispercpp` bindings for maximum speed and offline capability.

**Key Features:**
- Real-time transcription as you speak
- Whisper Large V3 Turbo (809M) as primary model
- Modular architecture supporting multiple models
- Voice Activity Detection (VAD) for efficient processing
- No-drop recording pipeline (disk-backed FIFO spool)
- Copy-to-clipboard functionality
- Cross-platform support (Windows, macOS, Linux)

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────┐
│                         PyQt6 GUI                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Text Area  │  │   Controls  │  │ Status (VAD/Queue)  │ │
│  │  (Output)   │  │(Start/Stop) │  │ + Settings dialog   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└───────────────────────────┬───────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────┐
│                    Audio Capture Thread                    │
│             (sounddevice/PyAudio + WebRTC VAD)             │
│  - Finalizes speech segments (silence timeout)             │
│  - Hard-splits long speech (max segment duration)          │
└───────────────────────────┬───────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────┐
│               Disk-Backed Recording Spool (FIFO)           │
│              output/spool/{queued,processing,failed}       │
│  - Each segment becomes a timestamped sequential WAV job   │
│  - If disk is low: stop recording (never silently drop)    │
└───────────────────────────┬───────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────┐
│               Sequential Transcription Pipeline            │
│                 (single worker thread, FIFO)               │
│  - Deletes job WAV on success                               │
│  - Moves to failed/ on error and continues                  │
└───────────────────────────┬───────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────┐
│                 whisper.cpp (pywhispercpp)                 │
│    - CPU by default; CUDA supported when built with CUDA   │
└───────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
cripit/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── README.md              # This file
├── config/
│   ├── __init__.py
│   └── settings.py        # App configuration
├── core/
│   ├── __init__.py
│   ├── audio_capture.py           # Audio capture + VAD, emits finalized recordings
│   ├── recording_spool.py         # Disk-backed FIFO spool (timestamped WAV jobs)
│   ├── transcription_pipeline.py  # Sequential transcription worker (FIFO)
│   ├── transcriber.py             # whisper.cpp wrapper
│   └── model_manager.py           # Model loading/switching
├── gui/
│   ├── __init__.py
│   ├── main_window.py     # Main PyQt window
│   ├── text_display.py    # Scrolling text area
│   └── controls.py        # Buttons, settings
├── models/                # GGML model files (gitignored)
│   └── README.md         # Model download instructions
└── utils/
    ├── __init__.py
    └── helpers.py         # Audio processing, etc.
```

---

## 🔧 Core Components

### 1. Audio Capture (`core/audio_capture.py`)
- **PyAudio** for microphone input
- **Silero VAD** (Voice Activity Detection) for speech detection
- Ring buffer for continuous audio (30-second chunks)
- Callback-driven for real-time processing
- Configurable sample rate: 16kHz (Whisper requirement)

### 2. Transcription Engine (`core/transcriber.py`)
- Uses **pywhispercpp** for whisper.cpp bindings
- Supported models:
  - Whisper Large V3 Turbo (809M) - **PRIMARY**
  - Whisper Large V3 (1.55B) - High quality
  - Distil-Whisper (756M) - Speed
  - Tiny/Base/Small (for testing)
- Auto language detection
- Thread-safe transcription
- Real-time callback for partial results

### 3. Model Manager (`core/model_manager.py`)
- Download/manage GGML models
- Auto-download missing models
- Model switching without restart
- Memory management (unload unused models)

### 4. GUI (`gui/main_window.py`)
**Main Window Components:**
- Large text display (scrollable, copyable)
- Start/Stop recording button
- Model selector dropdown
- Language selector (auto-detect or specific)
- Status bar (recording/processing/idle)
- Settings panel (audio device, VAD sensitivity)
- System tray icon (optional)
- Global hotkey support (Ctrl+Shift+R)

---

## ⚡ Real-Time Pipeline

### Audio Flow
1. Microphone → PyAudio
2. VAD detection (silence vs speech)
3. Audio chunks accumulate while speech detected
4. When speech ends (VAD silence) → Finalize a recording segment
5. Segment is spooled to disk as a sequential job (WAV + metadata)

### Transcription Flow
1. Pipeline reads the next spooled job (FIFO)
2. Job WAV → whisper.cpp
3. Text result generated
4. Results appended to text display
5. Job WAV is deleted on success (or moved to `output/spool/failed/` on error)

### Threading Model
- **Main thread**: PyQt GUI
- **Audio thread**: Continuous capture
- **Pipeline thread**: Sequential transcription worker (FIFO)

---

## 📦 Dependencies

```txt
PyQt6>=6.4.0
pywhispercpp>=1.2.0
PyAudio>=0.2.13
torch>=2.0.0        # For Silero VAD
numpy>=1.24.0
requests>=2.28.0    # For model downloading
```

---

## 🎛️ Features

### Core Features
- ✅ Real-time speech-to-text
- ✅ Whisper V3 Turbo (fast, accurate)
- ✅ Multi-model support (switchable)
- ✅ Auto language detection
- ✅ Copy-to-clipboard
- ✅ Audio device selection

### Advanced Features
- ✅ Voice Activity Detection (no silence transcription)
- ✅ Adjustable VAD sensitivity
- ✅ Partial result preview
- ✅ Keyboard shortcuts
- ✅ System tray mode

---

## 🧠 No-Drop Recording Spool

CripIt uses a disk-backed FIFO spool to ensure finalized recordings are not dropped when transcription falls behind.

- Spool root: `output/spool`
- States:
  - `output/spool/queued/` (backlog)
  - `output/spool/processing/` (in-flight)
  - `output/spool/failed/` (kept on error)

Operational rules:
- Jobs are processed strictly in order.
- On success: job WAV+JSON is deleted (no long-term archive).
- On failure: job is moved to `failed/` and the pipeline continues.
- On low disk: CripIt stops recording and shows an error (never silently drops).

If you are changing settings while the app is running:
- Most settings apply immediately.
- Microphone device changes generally require stopping and starting recording.

## 🎯 Supported Models

| Model | Params | Speed | WER | Use Case |
|-------|--------|-------|-----|----------|
| **Whisper Large V3 Turbo** | 809M | ⭐⭐⭐⭐⭐ | 7.75% | **PRIMARY** - Best balance |
| Whisper Large V3 | 1.55B | ⭐⭐⭐ | 7.4% | High quality, multilingual |
| Distil-Whisper | 756M | ⭐⭐⭐⭐⭐ | ~7.5% | English-only, fastest |
| Whisper Small | 466M | ⭐⭐⭐⭐ | ~10% | Testing, low resource |
| Whisper Base | 142M | ⭐⭐⭐⭐⭐ | ~15% | Testing only |

---

## 📝 Notes

### Model Files
Models are stored in `models/` directory as GGML binary files (.bin). These are NOT included in git.

**Download Models:**
```bash
# Using whisper.cpp's download script
./download-ggml-model.sh large-v3-turbo

# Or manually from:
# https://huggingface.co/ggerganov/whisper.cpp
```

### Audio Requirements
- Sample rate: 16kHz (Whisper requirement)
- Format: 16-bit PCM
- Channels: Mono

### Platform-Specific Notes
- **macOS**: May need to grant microphone permissions
- **Linux**: Requires PortAudio development libraries
- **Windows**: Works with default PyAudio wheels

---

## 🔮 Future Enhancements

- [ ] Export transcription to file
- [ ] Speaker diarization (who is speaking)
- [ ] Integration with OpenCode (the original use case!)
- [ ] Cloud sync for transcriptions
- [ ] Mobile app companion

---

## 📄 License

MIT License - Open source, free to use and modify.

---

**Created:** January 2026  
**Purpose:** Real-time STT for development workflows  
**Engine:** whisper.cpp + PyQt6
