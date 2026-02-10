#!/usr/bin/env python3
"""CripIt headless terminal UI (TUI).

This is a curses-based runner intended for Raspberry Pi / headless use.
It reuses the existing durable spool + sequential transcription pipeline.
"""

from __future__ import annotations

import argparse
import curses
import logging
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
        model_name: str,
        recording_device: str,
        output_file: Optional[Path],
        autostart: bool,
    ):
        self.stdscr = stdscr
        self.event_q = event_q
        self.audio = audio
        self.spool = spool
        self.pipeline = pipeline
        self.transcriber = transcriber
        self.model_name = model_name
        self.recording_device = recording_device

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
        curses.curs_set(0)
        self.stdscr.nodelay(True)
        self.stdscr.keypad(True)
        curses.noecho()
        curses.cbreak()

    def _safe_addstr(self, y: int, x: int, s: str, attr: int = 0) -> None:
        try:
            self.stdscr.addnstr(y, x, s, max(0, self.stdscr.getmaxyx()[1] - x - 1), attr)
        except curses.error:
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
        header = (
            f"Model: {self.model_name} | Device: {device} | State: {state} | "
            f"VAD: {vad_s} | Backlog: queued={queued} processing={processing} | "
            f"Perf: last={self.last_segment_s:.1f}s rtf={self.last_rtf:.2f} avg={avg_rtf:.2f}"
        )
        self._safe_addstr(0, 0, header[: w - 1])
        self._safe_addstr(1, 0, f"Mic: {self.recording_device}"[: w - 1])

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
                elif ch in (ord("r"), ord("R")):
                    self._toggle_recording()
                elif ch in (ord("c"), ord("C")):
                    self.lines = []
                elif ch in (ord("s"), ord("S")) and self.output_file:
                    self.saving_enabled = not self.saving_enabled

            self._render()
            time.sleep(0.05)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CripIt - headless terminal UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Basic usage (starts recording immediately)\n"
            "  python terminal_app.py --model base.en --threads 4\n\n"
            "  # Start idle (press 'r' to record)\n"
            "  python terminal_app.py --no-autostart\n\n"
            "  # List microphone devices\n"
            "  python terminal_app.py --list-devices\n\n"
            "  # List models and download one\n"
            "  python terminal_app.py --list-models\n"
            "  python terminal_app.py --download base.en\n"
        ),
    )
    p.add_argument("--model", default="base.en", help="Whisper model name (default: base.en)")
    p.add_argument("--language", default="en", help="Language code (default: en; use 'auto' for auto-detect)")
    p.add_argument("--threads", type=int, default=4, help="Inference threads (default: 4)")
    p.add_argument("--vad", choices=["webrtc", "energy", "silero"], default="webrtc", help="VAD type")
    p.add_argument("--device-index", type=int, default=None, help="Audio input device index")
    p.add_argument("--gain-db", type=float, default=0.0, help="Input gain in dB")
    p.add_argument("--spool-dir", default=str(Path("output") / "spool"), help="Spool directory")
    p.add_argument("--output-file", default=None, help="Append-only transcript output file")
    p.add_argument("--no-autostart", action="store_true", help="Start idle; press r to record")
    p.add_argument("--list-devices", action="store_true", help="List audio input devices and exit")
    p.add_argument("--list-models", action="store_true", help="List available models and exit")
    p.add_argument("--download", metavar="MODEL", help="Download a model and exit")
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


def _print_models() -> int:
    from core.model_manager import get_model_manager

    mm = get_model_manager()
    avail = set(mm.get_available_models())
    for name in mm.list_models():
        status = "downloaded" if name in avail else "missing"
        print(f"{name:20s} {status}")
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


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(list(argv) if argv is not None else sys.argv[1:])

    # Non-curses actions
    if args.list_devices:
        return _print_devices()
    if args.list_models:
        return _print_models()
    if args.download:
        return _download_model(args.download)

    _configure_logging(verbose=bool(args.verbose))
    log = logging.getLogger("terminal_app")
    log.info("Starting terminal_app")

    from core.audio_capture import AudioCapture
    from core.model_manager import get_model_manager
    from core.recording_spool import RecordingSpool
    from core.transcriber_factory import create_transcriber, check_engine_availability
    from core.transcription_pipeline import TranscriptionPipeline

    # Check engine availability
    engines = check_engine_availability()
    
    # Prepare model + transcriber
    mm = get_model_manager()
    
    # Get model path only if using whispercpp
    transcriber = create_transcriber()
    if not transcriber:
        print("ERROR: Failed to create transcriber")
        print("Make sure you have either pywhispercpp or whisperx installed")
        return 1
    
    # Load model based on engine type
    if not transcriber.load_model():
        print("Failed to load model")
        print("See output/terminal.log for details")
        return 1

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
            model_name=str(args.model),
            recording_device=recording_device,
            output_file=out_file,
            autostart=(not bool(args.no_autostart)),
        )
        ui.loop()

    try:
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
