"""Sequential transcription pipeline.

Consumes RecordingJob files from a RecordingSpool in strict order and runs
transcription using a single worker thread.

On success: deletes job audio + metadata (no long-term archive).
On failure: moves job into failed/ and keeps it for debugging.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import traceback
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from core.recording_spool import RecordingJob


def _read_wav_int16_mono(path: Path) -> Tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        if sampwidth != 2:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth}")
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    audio = np.frombuffer(raw, dtype=np.int16)
    if channels == 1:
        return audio, sample_rate, channels
    # Flatten to mono by selecting the first channel.
    audio = audio.reshape(-1, channels)[:, 0].copy()
    return audio, sample_rate, channels


@dataclass
class PipelineBacklog:
    queued: int
    processing: int


class TranscriptionPipeline:
    def __init__(
        self,
        *,
        spool_root: Path,
        transcriber,
    ):
        self.spool_root = Path(spool_root)
        self.transcriber = transcriber

        self._q: queue.Queue[RecordingJob] = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._processing_seq: Optional[int] = None

        # Callbacks
        self.on_job_started: Optional[Callable[[RecordingJob], None]] = None
        self.on_job_done: Optional[Callable[[RecordingJob, object], None]] = None
        self.on_job_failed: Optional[Callable[[RecordingJob, str, Path], None]] = None
        self.on_backlog_changed: Optional[Callable[[PipelineBacklog], None]] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="TranscriptionPipeline", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def enqueue(self, job: RecordingJob) -> None:
        self._q.put(job)
        self._emit_backlog()

    def backlog(self) -> PipelineBacklog:
        return PipelineBacklog(queued=int(self._q.qsize()), processing=1 if self._processing_seq is not None else 0)

    def _emit_backlog(self) -> None:
        if self.on_backlog_changed:
            try:
                self.on_backlog_changed(self.backlog())
            except Exception:
                pass

    def recover(self) -> int:
        """Recover queued + processing jobs from disk.

        Returns number of jobs enqueued.
        """
        enqueued = 0

        queued_dir = self.spool_root / "queued"
        processing_dir = self.spool_root / "processing"

        # Move processing back to queued (best effort)
        if processing_dir.exists():
            for wav in sorted(processing_dir.glob("**/*.wav")):
                rel = wav.relative_to(processing_dir)
                dst_wav = queued_dir / rel
                dst_wav.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.replace(wav, dst_wav)
                except Exception:
                    continue

                meta = wav.with_suffix(".json")
                if meta.exists():
                    dst_meta = dst_wav.with_suffix(".json")
                    try:
                        os.replace(meta, dst_meta)
                    except Exception:
                        pass

        # Enqueue queued jobs in filename order
        if queued_dir.exists():
            for wav in sorted(queued_dir.glob("**/*.wav")):
                meta = wav.with_suffix(".json")
                # Parse seq from filename
                name = wav.name
                if len(name) < 8 or not name[:8].isdigit():
                    continue
                try:
                    seq = int(name[:8])
                except Exception:
                    continue

                # Infer session_id from parent directory name
                session_id = wav.parent.name
                job = RecordingJob(
                    seq=seq,
                    session_id=session_id,
                    start_ts=0.0,
                    end_ts=0.0,
                    duration_ms=0,
                    wav_path=wav,
                    meta_path=meta,
                )
                self.enqueue(job)
                enqueued += 1

        return enqueued

    def _move_state(self, job: RecordingJob, from_state: str, to_state: str) -> RecordingJob:
        src_root = self.spool_root / from_state
        dst_root = self.spool_root / to_state

        try:
            rel = job.wav_path.relative_to(src_root)
        except Exception:
            # job already moved or is outside expected layout; keep path stable
            rel = Path(job.wav_path.name)

        dst_wav = dst_root / rel
        dst_wav.parent.mkdir(parents=True, exist_ok=True)
        dst_meta = dst_wav.with_suffix(".json")

        try:
            os.replace(job.wav_path, dst_wav)
        except Exception:
            # If move fails, keep original
            dst_wav = job.wav_path

        if job.meta_path.exists():
            try:
                os.replace(job.meta_path, dst_meta)
            except Exception:
                dst_meta = job.meta_path
        else:
            dst_meta = job.meta_path

        return RecordingJob(
            seq=job.seq,
            session_id=job.session_id,
            start_ts=job.start_ts,
            end_ts=job.end_ts,
            duration_ms=job.duration_ms,
            wav_path=dst_wav,
            meta_path=dst_meta,
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.25)
            except queue.Empty:
                continue

            self._processing_seq = job.seq
            self._emit_backlog()

            try:
                # Move to processing
                job = self._move_state(job, "queued", "processing")

                if self.on_job_started:
                    try:
                        self.on_job_started(job)
                    except Exception:
                        pass

                audio, sr, _ = _read_wav_int16_mono(job.wav_path)

                # Safety: trust job metadata but do not require it
                if sr and hasattr(self.transcriber, "language"):
                    pass

                result = self.transcriber.transcribe(audio)
                if result is None:
                    raise RuntimeError("transcribe() returned None")

                # Delete on success
                for p in (job.wav_path, job.meta_path):
                    try:
                        if p.exists():
                            p.unlink()
                    except Exception:
                        pass

                if self.on_job_done:
                    try:
                        self.on_job_done(job, result)
                    except Exception:
                        pass

            except Exception as e:
                # Move to failed and write an error file
                failed_job = self._move_state(job, "processing", "failed")
                err_path = failed_job.wav_path.with_suffix(failed_job.wav_path.suffix + ".error.txt")
                try:
                    with open(err_path, "w", encoding="utf-8") as f:
                        f.write(f"ts={time.time()}\n")
                        f.write(f"seq={failed_job.seq}\n")
                        f.write(f"error={type(e).__name__}: {e}\n\n")
                        f.write(traceback.format_exc())
                except Exception:
                    pass

                if self.on_job_failed:
                    try:
                        self.on_job_failed(failed_job, f"{type(e).__name__}: {e}", failed_job.wav_path)
                    except Exception:
                        pass

            finally:
                self._processing_seq = None
                self._emit_backlog()
