#!/usr/bin/env python3
"""CripIt headless terminal UI (TUI).

This is a curses-based runner intended for Raspberry Pi / headless use.
It reuses the existing durable spool + sequential transcription pipeline.
"""

from __future__ import annotations

import argparse
import contextlib
import curses
import logging
import os
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Event:
    type: str
    data: Dict[str, Any]


def _configure_logging(*, verbose: bool) -> None:
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "terminal.log"
    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if invoked multiple times.
    root.handlers = []

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def _extract_text(result: object) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if hasattr(result, "text"):
        try:
            return str(getattr(result, "text", ""))
        except Exception:
            return ""
    if isinstance(result, dict):
        return str(result.get("text", ""))
    try:
        return str(result)
    except Exception:
        return ""


def _wrap_lines(text: str, width: int) -> List[str]:
    if width <= 4:
        return [text[: max(0, width)]]
    out: List[str] = []
    for raw in text.splitlines() or [""]:
        s = raw.rstrip("\n")
        while len(s) > width:
            cut = s.rfind(" ", 0, width + 1)
            if cut <= 0:
                cut = width
            out.append(s[:cut].rstrip())
            s = s[cut:].lstrip()
        out.append(s)
    return out


def _configure_third_party_console_noise() -> None:
    """Reduce third-party stdout/stderr noise that can corrupt curses output."""
    # Hugging Face / transformers can emit progress bars and warnings to stderr.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _silence_noisy_loggers() -> None:
    """Best-effort: force common third-party loggers to stay off the terminal."""
    try:
        import logging as _logging

        prefixes = (
            "whisperx",
            "faster_whisper",
            "transformers",
            "huggingface_hub",
            "pyannote",
            "speechbrain",
            "torchaudio",
            "numba",
        )

        for name in list(getattr(_logging.root.manager, "loggerDict", {}).keys()):
            if not isinstance(name, str):
                continue
            if not name.startswith(prefixes):
                continue
            lg = _logging.getLogger(name)
            lg.setLevel(_logging.ERROR)
            # If a library installed its own StreamHandler, drop it so logs go
            # to our root handlers (file) only.
            lg.handlers = []
            lg.propagate = True
    except Exception:
        pass


@contextlib.contextmanager
def _redirect_stdout_to_file(path: Path):
    """Temporarily redirect Python-level stdout to a file."""
    f = None
    old_out = sys.stdout
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        f = open(path, "a", encoding="utf-8")
        sys.stdout = f  # type: ignore[assignment]
        yield
    finally:
        try:
            if f:
                f.flush()
        except Exception:
            pass
        sys.stdout = old_out
        try:
            if f:
                f.close()
        except Exception:
            pass


@contextlib.contextmanager
def _redirect_stderr_to_file(path: Path):
    """Temporarily redirect Python-level stderr to a file."""
    f = None
    old_err = sys.stderr
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        f = open(path, "a", encoding="utf-8")
        sys.stderr = f  # type: ignore[assignment]
        yield
    finally:
        try:
            if f:
                f.flush()
        except Exception:
            pass
        sys.stderr = old_err
        try:
            if f:
                f.close()
        except Exception:
            pass


@contextlib.contextmanager
def _redirect_stderr_fd_to_file(path: Path):
    """Redirect OS-level stderr (fd 2) to a file.

    IMPORTANT: Do not redirect stdout (fd 1) while curses runs, or curses may
    fail to initialize (cbreak()/nocbreak() ERR) because stdout is no longer a TTY.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    f = None
    old2 = None
    try:
        f = open(path, "a", encoding="utf-8")
        old2 = os.dup(2)
        os.dup2(f.fileno(), 2)
        yield
    finally:
        try:
            if old2 is not None:
                os.dup2(old2, 2)
        except Exception:
            pass
        try:
            if old2 is not None:
                os.close(old2)
        except Exception:
            pass
        try:
            if f:
                f.close()
        except Exception:
            pass


class TerminalUI:
    def __init__(
        self,
        *,
        stdscr,
        event_q: "queue.Queue[Event]",
        audio,
        spool,
        pipeline,
        transcriber,
        engine: str,
        model_name: str,
        recording_device: str,
        output_file: Optional[Path],
        autostart: bool,
        whisperx_options: Optional[Dict[str, Any]] = None,
    ):
        self.stdscr = stdscr
        self.event_q = event_q
        self.audio = audio
        self.spool = spool
        self.pipeline = pipeline
        self.transcriber = transcriber
        self.engine = engine
        self.model_name = model_name
        self.recording_device = recording_device
        self.whisperx_options = whisperx_options or {}

        self.output_file = output_file
        self.saving_enabled = bool(output_file)

        self.running = True
        self.recording_enabled = False
        self.last_vad_speech = False
        self.last_error: Optional[str] = None
        self.last_segment_s: float = 0.0
        self.last_rtf: float = 0.0

        self.lines: List[str] = []
        self.max_lines = 4000

        self._init_curses()

        if autostart:
            # Start recording after curses is initialized.
            self._toggle_recording(force_on=True)

    def _init_curses(self) -> None:
        # Some terminals (or TERM settings) don't support cursor visibility
        # changes; avoid crashing on startup.
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        except Exception:
            pass

        self.stdscr.nodelay(True)
        self.stdscr.keypad(True)
        curses.noecho()
        curses.cbreak()

        # Make escape sequences feel less "laggy" (best-effort).
        try:
            curses.set_escdelay(25)
        except Exception:
            pass

    def _safe_addstr(self, y: int, x: int, s: str, attr: int = 0) -> None:
        try:
            self.stdscr.addnstr(y, x, s, max(0, self.stdscr.getmaxyx()[1] - x - 1), attr)
        except curses.error:
            # Some terminals can throw on wide/unprintable chars; try a
            # conservative ASCII fallback so the UI still renders.
            try:
                safe = s.encode("ascii", "replace").decode("ascii")
                self.stdscr.addnstr(y, x, safe, max(0, self.stdscr.getmaxyx()[1] - x - 1), attr)
            except Exception:
                pass

    def _append_text(self, text: str) -> None:
        if not text:
            return
        w = max(10, int(self.stdscr.getmaxyx()[1]) - 1)
        for ln in _wrap_lines(text, w):
            self.lines.append(ln)
        if len(self.lines) > self.max_lines:
            self.lines = self.lines[-self.max_lines :]

    def _toggle_recording(self, *, force_on: Optional[bool] = None) -> None:
        target = (not self.recording_enabled) if force_on is None else bool(force_on)
        if target == self.recording_enabled:
            return

        if target:
            if not self.transcriber.is_ready():
                self._append_text("[error] model is not loaded; cannot start recording")
                self.last_error = "model not loaded"
                return
            ok = bool(self.audio.start())
            if ok:
                self.recording_enabled = True
                self._append_text("[info] recording started")
            else:
                self._append_text("[error] failed to start audio capture")
                self.last_error = "audio start failed"
        else:
            try:
                self.audio.stop()
            except Exception:
                pass
            self.recording_enabled = False
            self._append_text("[info] recording stopped")

    def _maybe_save(self, text: str) -> None:
        if not self.output_file or not self.saving_enabled:
            return
        try:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(text.rstrip() + "\n")
        except Exception as e:
            self.last_error = f"failed to write output file: {e}"
            self._append_text(f"[error] {self.last_error}")

    def _handle_event(self, ev: Event) -> None:
        t = ev.type
        d = ev.data

        if t == "vad":
            self.last_vad_speech = bool(d.get("speech", False))
            return

        if t == "state":
            return

        if t == "backlog":
            return

        if t == "job_started":
            seq = int(d.get("seq", 0))
            self._append_text(f"[job {seq:08d}] transcribing...")
            return

        if t == "job_done":
            seq = int(d.get("seq", 0))
            text = str(d.get("text", ""))
            seg_s = float(d.get("duration_s", 0.0))
            rtf = float(d.get("rtf", 0.0))
            self.last_segment_s = seg_s
            self.last_rtf = rtf
            if text.strip():
                self._append_text("")
                self._append_text(f"[{seq:08d}] {text.strip()}")
                self._maybe_save(text)
            return

        if t == "job_failed":
            seq = int(d.get("seq", 0))
            err = str(d.get("error", ""))
            path = str(d.get("path", ""))
            self.last_error = err
            self._append_text(f"[job {seq:08d}] failed: {err} ({path})")
            return

        if t == "spool_error":
            msg = str(d.get("error", "spool error"))
            self.last_error = msg
            self._append_text(f"[error] {msg}")
            # Stop recording on spool failure.
            self._toggle_recording(force_on=False)
            return

        if t == "low_disk":
            self._append_text("[warn] low disk space; stopping recording")
            self._toggle_recording(force_on=False)
            return

    def _render(self) -> None:
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()
        w = max(10, w)
        h = max(3, h)

        # Header
        try:
            backlog = self.pipeline.backlog() if self.pipeline else None
            queued = int(getattr(backlog, "queued", 0))
            processing = int(getattr(backlog, "processing", 0))
        except Exception:
            queued, processing = 0, 0

        try:
            stats = self.transcriber.get_stats()
            avg_rtf = float(stats.get("avg_realtime_factor", 0.0))
            device = str(stats.get("device", "CPU"))
        except Exception:
            avg_rtf = 0.0
            device = "CPU"

        try:
            state = self.audio.get_state().name
        except Exception:
            state = "IDLE"

        vad_s = "speech" if self.last_vad_speech else "silence"
        
        # Build header with engine info
        engine_str = f"{self.engine}"
        if self.engine == "whisperx" and self.whisperx_options:
            compute = self.whisperx_options.get('compute_type', '')
            if compute:
                engine_str += f"({compute})"
        
        header = (
            f"Engine: {engine_str} | Model: {self.model_name} | Device: {device} | State: {state} | "
            f"VAD: {vad_s}"
        )
        self._safe_addstr(0, 0, header[: w - 1])
        
        # Second line with performance and backlog
        header2 = (
            f"Mic: {self.recording_device} | "
            f"Backlog: queued={queued} processing={processing} | "
            f"Perf: last={self.last_segment_s:.1f}s rtf={self.last_rtf:.2f} avg={avg_rtf:.2f}"
        )
        self._safe_addstr(1, 0, header2[: w - 1])

        # Transcript area
        top = 2
        bottom = h - 2
        avail = max(0, bottom - top + 1)
        view = self.lines[-avail:] if avail > 0 else []
        for i, ln in enumerate(view):
            self._safe_addstr(top + i, 0, ln[: w - 1])

        # Footer
        footer = "q quit | r toggle rec | c clear"
        if self.output_file:
            footer += f" | s save={'on' if self.saving_enabled else 'off'} ({self.output_file})"
        if self.last_error:
            footer += f" | err: {self.last_error}"
        self._safe_addstr(h - 1, 0, footer[: w - 1])

        self.stdscr.refresh()

    def loop(self) -> None:
        while self.running:
            # Events
            for _ in range(200):
                try:
                    ev = self.event_q.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._handle_event(ev)
                except Exception:
                    pass

            # Input
            try:
                ch = self.stdscr.getch()
            except Exception:
                ch = -1

            if ch != -1:
                if ch in (ord("q"), ord("Q")):
                    self.running = False
                elif ch == curses.KEY_RESIZE:
                    # Next render will redraw to the new size.
                    pass
                elif ch in (ord("r"), ord("R")):
                    self._toggle_recording()
                elif ch in (ord("c"), ord("C")):
                    self.lines = []
                elif ch in (ord("s"), ord("S")) and self.output_file:
                    self.saving_enabled = not self.saving_enabled

            try:
                self._render()
            except Exception as e:
                # Avoid leaving the terminal in a broken state due to render errors.
                self.last_error = f"render failed: {type(e).__name__}: {e}"
                time.sleep(0.1)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CripIt - headless terminal UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Choose engine:\n"
            "  # - whispercpp: uses local GGML .bin models in models/\n"
            "  # - whisperx: auto-downloads models from Hugging Face on first use\n"
            "  #   (or pre-download with --download-whisperx)\n\n"
            "  # Basic usage with whisper.cpp (default)\n"
            "  python terminal_app.py --engine whispercpp --model base.en --threads 4\n\n"
            "  # Use WhisperX for faster transcription\n"
            "  python terminal_app.py --engine whisperx --language en --whisperx-model base\n\n"
            "  # WhisperX on multicore CPU\n"
            "  python terminal_app.py --engine whisperx --language en --whisperx-device cpu --whisperx-model tiny --whisperx-compute-type int8 --whisperx-cpu-threads 4\n\n"
            "  # WhisperX with speaker diarization\n"
            "  python terminal_app.py --engine whisperx --whisperx-diarize --whisperx-hf-token <token>\n\n"
            "  # Start idle (press 'r' to record)\n"
            "  python terminal_app.py --no-autostart\n\n"
            "  # List microphone devices\n"
            "  python terminal_app.py --list-devices\n\n"
            "  # List models for specific engine\n"
            "  python terminal_app.py --list-models --engine whisperx\n\n"
            "  # Download whisper.cpp model (only used by whispercpp)\n"
            "  python terminal_app.py --download tiny\n"
            "\n"
            "  # Pre-download WhisperX model into Hugging Face cache\n"
            "  python terminal_app.py --download-whisperx tiny\n"
        ),
    )
    
    # ASR Engine selection
    p.add_argument(
        "--engine",
        choices=["whispercpp", "whisperx"],
        default=None,
        help=(
            "ASR engine to use (default: from config). "
            "whispercpp uses local .bin models in models/; whisperx auto-downloads from Hugging Face."
        ),
    )
    
    # whisper.cpp options
    p.add_argument("--model", default=None, help="Whisper.cpp model name (default: from config)")
    p.add_argument("--threads", type=int, default=None, help="Inference threads (default: from config)")
    
    # WhisperX options
    p.add_argument("--whisperx-model", default=None, 
                   choices=["tiny", "base", "small", "medium", "large-v3"],
                   help="WhisperX model size (default: from config)")
    p.add_argument("--whisperx-compute-type", default=None,
                   choices=["int8", "float16", "float32"],
                   help="WhisperX compute precision (default: from config)")
    p.add_argument("--whisperx-device", default=None,
                   choices=["cuda", "cpu"],
                   help="WhisperX device (default: from config)")
    p.add_argument(
        "--whisperx-cpu-threads",
        type=int,
        default=None,
        help="WhisperX CPU threads (0=auto; default: from config)",
    )
    p.add_argument(
        "--whisperx-num-workers",
        type=int,
        default=None,
        help="WhisperX worker processes/threads (default: from config)",
    )
    p.add_argument("--whisperx-diarize", action="store_true",
                   help="Enable speaker diarization (requires --whisperx-hf-token)")
    p.add_argument("--whisperx-hf-token", default=None,
                   help="HuggingFace token for diarization (get at https://huggingface.co/settings/tokens)")
    
    # Common options
    p.add_argument("--language", default=None, help="Language code (e.g., 'en'; default: from config)")
    p.add_argument("--vad", choices=["webrtc", "energy", "silero"], default="webrtc", help="VAD type")
    p.add_argument("--device-index", type=int, default=None, help="Audio input device index")
    p.add_argument("--gain-db", type=float, default=0.0, help="Input gain in dB")
    p.add_argument("--spool-dir", default=str(Path("output") / "spool"), help="Spool directory")
    p.add_argument("--output-file", default=None, help="Append-only transcript output file")
    p.add_argument("--no-autostart", action="store_true", help="Start idle; press r to record")
    p.add_argument("--list-devices", action="store_true", help="List audio input devices and exit")
    p.add_argument("--list-models", action="store_true", help="List available models and exit")
    p.add_argument(
        "--download",
        metavar="MODEL",
        help="Download a whisper.cpp model into models/ and exit (whispercpp only)",
    )
    p.add_argument(
        "--download-whisperx",
        metavar="SIZE",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help=(
            "Pre-download WhisperX model into Hugging Face cache and exit "
            "(requires huggingface_hub; models still load via --engine whisperx)"
        ),
    )
    p.add_argument("--check-engines", action="store_true", help="Check which ASR engines are available and exit")
    p.add_argument("--verbose", action="store_true", help="Verbose logging to output/terminal.log")
    return p.parse_args(argv)


def _print_devices() -> int:
    from core.audio_capture import AudioCapture

    devices = AudioCapture.list_devices()
    if not devices:
        print("No input devices found")
        return 1

    print("Input devices:")
    for d in devices:
        mark = "*" if d.get("default") else " "
        idx = d.get("index")
        name = d.get("name")
        ch = d.get("channels")
        sr = d.get("sample_rate")
        print(f"{mark} {idx:>3} | ch={ch} | sr={sr} | {name}")
    return 0


def _describe_recording_device(audio_cls, *, device_index: Optional[int], channels: int, sample_rate: int) -> str:
    try:
        devices = audio_cls.list_devices()
    except Exception:
        devices = []

    selected = None
    if device_index is not None:
        for d in devices:
            if int(d.get("index", -9999)) == int(device_index):
                selected = d
                break
        if selected is None:
            return f"#{device_index} (unavailable) | ch={channels} | sr={sample_rate}"
    else:
        selected = next((d for d in devices if d.get("default")), None)
        if selected is None and devices:
            selected = devices[0]

    if not selected:
        return f"default | ch={channels} | sr={sample_rate}"

    idx = selected.get("index")
    name = str(selected.get("name", "unknown"))
    return f"#{idx} | ch={channels} | sr={sample_rate} | {name}"


def _check_engines() -> int:
    """Check which ASR engines are available."""
    from core.transcriber_factory import check_engine_availability
    
    engines = check_engine_availability()
    
    print("\nASR Engine Availability:")
    print("=" * 40)
    for engine, available in engines.items():
        status = "✓ Available" if available else "✗ Not installed"
        print(f"  {engine:15s} {status}")
    print("=" * 40)
    
    # Return 0 if at least one engine is available
    return 0 if any(engines.values()) else 1


def _print_models(engine: Optional[str] = None) -> int:
    from core.model_manager import get_model_manager
    from core.transcriber_factory import check_engine_availability
    
    mm = get_model_manager()
    engines = check_engine_availability()
    
    # If engine not specified, try to determine from config
    if engine is None:
        try:
            from config.settings import config
            engine = config.model.asr_engine
        except:
            engine = "whispercpp"
    
    print(f"\nAvailable models for engine: {engine}")
    print("=" * 60)
    
    if engine == "whisperx":
        # WhisperX models (auto-downloaded from HF)
        print("Note: WhisperX models are downloaded automatically from Hugging Face on first use")
        print("      (or pre-download with: python terminal_app.py --download-whisperx tiny)\n")
        whisperx_models = {
            "tiny": "39M parameters, ~150MB",
            "base": "74M parameters, ~290MB", 
            "small": "244M parameters, ~900MB",
            "medium": "769M parameters, ~3GB",
            "large-v3": "1.55B parameters, ~6GB",
        }
        for name, desc in whisperx_models.items():
            print(f"  {name:12s} {desc}")
    else:
        # whisper.cpp models (need manual download)
        avail = set(mm.get_available_models())
        print("Note: Use --download <model> to download whisper.cpp models into models/\n")
        for name in mm.list_models():
            status = "✓ downloaded" if name in avail else "  not downloaded"
            info = mm.get_model_info(name)
            if info:
                print(f"  {name:20s} {status:18s} ({info.params}, ~{info.size_mb}MB)")
            else:
                print(f"  {name:20s} {status}")
    
    print("=" * 60)
    return 0


def _download_model(model: str) -> int:
    from core.model_manager import get_model_manager

    mm = get_model_manager()
    if model not in mm.list_models():
        print(f"Unknown model: {model}")
        return 2

    info = mm.get_model_info(model)
    size = getattr(info, "size_mb", None)
    if size:
        print(f"Downloading {model} ({size} MB)")
    else:
        print(f"Downloading {model}")

    def on_progress(name: str, percent: int, total: int) -> None:
        print(f"\r{name}: {percent}%", end="", flush=True)

    def on_complete(name: str, success: bool) -> None:
        print("")
        print("ok" if success else "failed")

    mm.on_progress = on_progress
    mm.on_complete = on_complete
    ok = bool(mm.download_model(model, blocking=True))
    return 0 if ok else 1


def _download_whisperx_model(size: str) -> int:
    """Pre-download WhisperX model weights into Hugging Face cache."""
    # Map CLI sizes to HF repos.
    repo_id = f"openai/whisper-{size}" if size != "large-v3" else "openai/whisper-large-v3"

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception:
        print("ERROR: huggingface_hub is not installed")
        print("Install with: pip install -U huggingface_hub")
        return 1

    print(f"Pre-downloading WhisperX model '{size}' into Hugging Face cache")
    print(f"Repo: {repo_id}")

    try:
        # Download into default HF cache (no local_dir) so WhisperX can pick it up.
        snapshot_download(repo_id=repo_id)
    except Exception as e:
        print(f"failed: {type(e).__name__}: {e}")
        return 1

    print("ok")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(list(argv) if argv is not None else sys.argv[1:])

    _configure_third_party_console_noise()

    # Non-curses actions
    if args.list_devices:
        return _print_devices()
    if args.check_engines:
        return _check_engines()
    if args.list_models:
        return _print_models(args.engine)
    if args.download:
        return _download_model(args.download)
    if getattr(args, "download_whisperx", None):
        return _download_whisperx_model(args.download_whisperx)

    _configure_logging(verbose=bool(args.verbose))
    log = logging.getLogger("terminal_app")
    log.info("Starting terminal_app")

    from core.audio_capture import AudioCapture
    from core.model_manager import get_model_manager
    from core.recording_spool import RecordingSpool
    from core.transcriber_factory import create_transcriber, check_engine_availability
    from core.transcription_pipeline import TranscriptionPipeline
    from config.settings import config

    # Check engine availability
    engines = check_engine_availability()
    
    # Determine which engine to use
    engine = args.engine or config.model.asr_engine
    
    # Validate engine availability
    if engine == "whisperx" and not engines["whisperx"]:
        print("ERROR: WhisperX engine selected but not installed")
        print("Install with: pip install whisperx torch torchaudio")
        return 1
    if engine == "whispercpp" and not engines["whispercpp"]:
        print("ERROR: whisper.cpp engine selected but not installed")
        print("Install with: pip install pywhispercpp")
        return 1
    
    # Update config with CLI arguments
    if args.engine:
        config.model.asr_engine = args.engine
    if args.language:
        config.model.language = args.language
    if args.threads:
        config.model.n_threads = args.threads
    
    # Update WhisperX-specific config
    if engine == "whisperx":
        if args.whisperx_model:
            config.model.whisperx_model = args.whisperx_model
        if args.whisperx_compute_type:
            config.model.whisperx_compute_type = args.whisperx_compute_type
        if args.whisperx_device:
            config.model.whisperx_device = args.whisperx_device
        if getattr(args, "whisperx_cpu_threads", None) is not None:
            config.model.whisperx_cpu_threads = int(args.whisperx_cpu_threads)
        if getattr(args, "whisperx_num_workers", None) is not None:
            config.model.whisperx_num_workers = int(args.whisperx_num_workers)
        if args.whisperx_diarize:
            config.model.whisperx_diarize = True
        if args.whisperx_hf_token:
            config.model.whisperx_hf_token = args.whisperx_hf_token
        
        whisperx_options = {
            'compute_type': config.model.whisperx_compute_type,
            'diarize': config.model.whisperx_diarize,
            'cpu_threads': getattr(config.model, 'whisperx_cpu_threads', 0),
            'num_workers': getattr(config.model, 'whisperx_num_workers', 1),
        }
        
        # Determine model name to display
        model_name = args.whisperx_model or config.model.whisperx_model
    else:
        # whisper.cpp
        if args.model:
            config.model.default_model = args.model
        model_name = args.model or config.model.default_model
        whisperx_options = {}
    
    # Save config changes
    config.save_config()
    
    # Create transcriber
    log.info(f"Creating transcriber with engine: {engine}")
    transcriber = create_transcriber(engine)
    if not transcriber:
        print("ERROR: Failed to create transcriber")
        print("Make sure you have either pywhispercpp or whisperx installed")
        return 1
    
    # Load model
    log.info(f"Loading model for {engine}...")
    if not transcriber.load_model():
        print(f"Failed to load model for {engine}")
        if engine == "whisperx":
            print("\nTroubleshooting:")
            print("  - Check GPU memory (WhisperX large needs ~6GB VRAM)")
            print("  - Try a smaller model: --whisperx-model base")
            print("  - Try CPU mode: --whisperx-device cpu")
            print("  - Check internet connection (models auto-download from HuggingFace)")
        print("\nSee output/terminal.log for details")
        return 1
    
    log.info(f"Model loaded successfully: {model_name}")

    # Durable spool + pipeline
    spool_dir = Path(args.spool_dir)
    spool_dir.mkdir(parents=True, exist_ok=True)
    spool = RecordingSpool(spool_dir)
    pipeline = TranscriptionPipeline(spool_root=spool_dir, transcriber=transcriber)

    # Audio capture
    audio = AudioCapture(
        sample_rate=16000,
        channels=1,
        vad_type=str(args.vad),
        vad_aggressiveness=2,
        silence_timeout=1.5,
        max_recording_duration=30.0,
        device_index=args.device_index,
        gain_db=float(args.gain_db),
    )
    recording_device = _describe_recording_device(
        AudioCapture,
        device_index=args.device_index,
        channels=1,
        sample_rate=16000,
    )

    event_q: "queue.Queue[Event]" = queue.Queue()

    def emit(ev_type: str, **data: Any) -> None:
        try:
            event_q.put_nowait(Event(ev_type, dict(data)))
        except Exception:
            pass

    def on_recording_ready(recording) -> None:
        try:
            job = spool.enqueue(recording)
        except Exception as e:
            emit("spool_error", error=f"failed to spool recording: {type(e).__name__}: {e}")
            return

        pipeline.enqueue(job)
        if spool.should_stop_for_low_disk():
            emit("low_disk")

    audio.on_recording_ready = on_recording_ready
    audio.on_state_change = lambda st: emit("state", state=str(getattr(st, "name", st)))
    audio.on_speech_detected = lambda s: emit("vad", speech=bool(s))

    pipeline.on_job_started = lambda job: emit("job_started", seq=int(getattr(job, "seq", 0)))

    def _job_done(job, result) -> None:
        text = _extract_text(result)
        dur_s = float(getattr(result, "duration", 0.0) or 0.0)
        proc_s = float(getattr(result, "processing_time", 0.0) or 0.0)
        rtf = (proc_s / dur_s) if dur_s > 0 else 0.0
        emit("job_done", seq=int(getattr(job, "seq", 0)), text=text, duration_s=dur_s, rtf=rtf)

    pipeline.on_job_done = _job_done
    pipeline.on_job_failed = lambda job, err, path: emit(
        "job_failed",
        seq=int(getattr(job, "seq", 0)),
        error=str(err),
        path=str(path),
    )
    pipeline.on_backlog_changed = lambda b: emit("backlog", queued=int(getattr(b, "queued", 0)), processing=int(getattr(b, "processing", 0)))

    pipeline.start()
    try:
        recovered = pipeline.recover()
        if recovered:
            log.info("Recovered %d queued job(s) from spool", recovered)
    except Exception:
        pass

    out_file = Path(args.output_file) if args.output_file else None

    def _curses_main(stdscr) -> None:
        ui = TerminalUI(
            stdscr=stdscr,
            event_q=event_q,
            audio=audio,
            spool=spool,
            pipeline=pipeline,
            transcriber=transcriber,
            engine=engine,
            model_name=model_name,
            recording_device=recording_device,
            output_file=out_file,
            autostart=(not bool(args.no_autostart)),
            whisperx_options=whisperx_options if engine == "whisperx" else None,
        )
        ui.loop()

    try:
        # Redirect stderr during curses so that libraries like WhisperX/tqdm
        # don't corrupt the screen with progress bars/warnings.
        ext_log = Path("output") / "terminal_external.log"
        _silence_noisy_loggers()
        with (
            _redirect_stderr_fd_to_file(ext_log),
            _redirect_stdout_to_file(ext_log),
            _redirect_stderr_to_file(ext_log),
        ):
            curses.wrapper(_curses_main)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            audio.stop()
        except Exception:
            pass
        try:
            pipeline.stop(timeout=2.0)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
