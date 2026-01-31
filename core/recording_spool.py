"""Recording spool (disk-backed FIFO).

This module provides a durable, sequential, timestamped spool for finalized
audio recordings so that the capture side never needs to drop segments when
transcription falls behind.

Spool layout (under root):
  output/spool/
    queued/YYYYMMDD/<session_id>/*.wav + *.json
    processing/YYYYMMDD/<session_id>/*.wav + *.json
    failed/YYYYMMDD/<session_id>/*.wav + *.json + *.error.txt
"""

from __future__ import annotations

import json
import os
import shutil
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


class DiskLowError(RuntimeError):
    pass


class SpoolWriteError(RuntimeError):
    pass


@dataclass(frozen=True)
class FinalizedRecording:
    audio_int16: np.ndarray
    sample_rate: int
    channels: int
    start_ts: float
    end_ts: float
    chunks: int = 0

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.end_ts - self.start_ts))

    @property
    def num_samples(self) -> int:
        return int(self.audio_int16.size)


@dataclass(frozen=True)
class RecordingJob:
    seq: int
    session_id: str
    start_ts: float
    end_ts: float
    duration_ms: int
    wav_path: Path
    meta_path: Path


def _ts_iso_compact(ts: float) -> str:
    # Example: 20260131T013344.123
    t = time.localtime(ts)
    ms = int(round((ts - int(ts)) * 1000.0))
    return time.strftime("%Y%m%dT%H%M%S", t) + f".{ms:03d}"


def _date_dir(ts: Optional[float] = None) -> str:
    t = time.localtime(ts if ts is not None else time.time())
    return time.strftime("%Y%m%d", t)


def _ensure_int16_mono(audio_int16: np.ndarray) -> np.ndarray:
    if not isinstance(audio_int16, np.ndarray):
        raise TypeError("audio_int16 must be a numpy array")
    if audio_int16.dtype != np.int16:
        raise TypeError(f"audio_int16 must be int16, got {audio_int16.dtype}")
    if audio_int16.ndim != 1:
        # CripIt currently operates mono; keep 1D invariant for safety.
        raise ValueError(f"audio_int16 must be 1D mono array, got shape {audio_int16.shape}")
    return audio_int16


class RecordingSpool:
    def __init__(
        self,
        root_dir: Path,
        *,
        session_id: Optional[str] = None,
        soft_min_free_mb: int = 1024,
        hard_reserve_mb: int = 256,
    ):
        self.root_dir = Path(root_dir)
        self.session_id = session_id or time.strftime("%Y%m%d-%H%M%S") + f"-pid{os.getpid()}"
        self.soft_min_free_bytes = int(soft_min_free_mb) * 1024 * 1024
        self.hard_reserve_bytes = int(hard_reserve_mb) * 1024 * 1024

        self._seq = 0
        self._queued_warned_low_disk = False

        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _state_dir(self, state: str, *, ts: Optional[float] = None) -> Path:
        return self.root_dir / state / _date_dir(ts) / self.session_id

    def ensure_dirs(self, *, ts: Optional[float] = None) -> None:
        for state in ("queued", "processing", "failed"):
            self._state_dir(state, ts=ts).mkdir(parents=True, exist_ok=True)

    def _scan_next_seq(self) -> int:
        max_seq = 0
        for state in ("queued", "processing", "failed"):
            base = self._state_dir(state)
            if not base.exists():
                continue
            for p in base.glob("*.wav"):
                name = p.name
                if len(name) < 8 or not name[:8].isdigit():
                    continue
                try:
                    max_seq = max(max_seq, int(name[:8]))
                except Exception:
                    continue
        return max_seq + 1

    def next_seq(self) -> int:
        if self._seq <= 0:
            self._seq = self._scan_next_seq()
        else:
            self._seq += 1
        return self._seq

    def _estimate_job_bytes(self, recording: FinalizedRecording) -> int:
        # PCM16 WAV is approximately header (44 bytes) + raw audio bytes.
        return 44 + int(recording.audio_int16.size) * 2

    def _check_disk_or_raise(self, needed_bytes: int) -> Tuple[int, int, int]:
        usage = shutil.disk_usage(str(self.root_dir))
        free = int(usage.free)
        total = int(usage.total)
        used = int(usage.used)

        if free < needed_bytes + self.hard_reserve_bytes:
            raise DiskLowError(
                f"Low disk space: free={free} bytes, needed~={needed_bytes} bytes, "
                f"hard_reserve={self.hard_reserve_bytes} bytes"
            )
        return free, used, total

    def enqueue(self, recording: FinalizedRecording, *, extra_meta: Optional[Dict[str, Any]] = None) -> RecordingJob:
        audio = _ensure_int16_mono(recording.audio_int16)
        self.ensure_dirs(ts=recording.start_ts)

        seq = self.next_seq()
        start_iso = _ts_iso_compact(recording.start_ts)
        duration_ms = int(round(max(0.0, (recording.end_ts - recording.start_ts) * 1000.0)))

        filename_base = f"{seq:08d}_{start_iso}_{duration_ms}ms"
        out_dir = self._state_dir("queued", ts=recording.start_ts)
        wav_path = out_dir / f"{filename_base}.wav"
        meta_path = out_dir / f"{filename_base}.json"

        needed = self._estimate_job_bytes(recording)
        free, _, _ = self._check_disk_or_raise(needed)

        wav_tmp = wav_path.with_suffix(wav_path.suffix + ".tmp")
        meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")

        meta: Dict[str, Any] = {
            "seq": seq,
            "session_id": self.session_id,
            "start_ts": float(recording.start_ts),
            "end_ts": float(recording.end_ts),
            "duration_ms": duration_ms,
            "sample_rate": int(recording.sample_rate),
            "channels": int(recording.channels),
            "num_samples": int(audio.size),
            "chunks": int(recording.chunks),
            "created_ts": time.time(),
        }
        if extra_meta:
            meta.update(extra_meta)

        try:
            # Write WAV
            with wave.open(str(wav_tmp), "wb") as wf:
                wf.setnchannels(int(recording.channels))
                wf.setsampwidth(2)
                wf.setframerate(int(recording.sample_rate))
                wf.writeframes(audio.tobytes())

            # Write metadata
            with open(meta_tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True)

            os.replace(wav_tmp, wav_path)
            os.replace(meta_tmp, meta_path)
        except Exception as e:
            # Best-effort cleanup
            for p in (wav_tmp, meta_tmp):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            raise SpoolWriteError(f"Failed to write job to spool: {e}") from e

        # Soft threshold handling: mark that we should stop soon.
        if free < self.soft_min_free_bytes:
            self._queued_warned_low_disk = True

        return RecordingJob(
            seq=seq,
            session_id=self.session_id,
            start_ts=float(recording.start_ts),
            end_ts=float(recording.end_ts),
            duration_ms=duration_ms,
            wav_path=wav_path,
            meta_path=meta_path,
        )

    def should_stop_for_low_disk(self) -> bool:
        return bool(self._queued_warned_low_disk)
