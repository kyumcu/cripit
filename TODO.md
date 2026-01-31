# CripIt TODO (No-Drop Recording Pipeline)

This file tracks the work to change CripIt's architecture so that finalized recordings are never dropped when transcription falls behind.

Target behavior:
- Every finalized VAD segment becomes a timestamped, sequential job on disk.
- Jobs are processed strictly in order (single consumer).
- On success: delete the audio job file (no long-term archive).
- On failure: move job audio/metadata to `output/spool/failed/` and continue with the next job.
- On low disk: stop recording and surface a clear error (never silently drop).

Non-goals (for this change):
- Multi-worker parallel transcription (whisper.cpp is not thread-safe).
- Long-term audio archival or rotation.
- Automatic retry of failed jobs (manual requeue can be added later).

---

## Phase 0: Ground Rules / Definitions

Definitions:
- Recording: an in-memory finalized audio segment (int16 PCM at 16kHz, mono).
- Job: a persisted recording on disk with metadata and an assigned sequence number.
- Spool: disk-backed FIFO queue used to decouple capture from transcription.

Job ordering rules:
- Order is defined by a monotonically increasing `seq` within a session.
- Filename includes `seq` (zero-padded) and start timestamp.
- Lexicographic sort MUST equal processing order.

Disk rules:
- If we cannot safely write the next job (free space < job size + hard reserve), we stop recording and show an error.
- If we can write but free space is below the soft threshold, we warn and stop recording before accepting the next job.

Failure rules:
- If transcription fails for a job, move it to `failed/` and keep it.
- The pipeline continues with the next job.

---

## Phase 1: Spool + Pipeline (Core)

### 1.1 Create a durable spool writer
- Add `core/recording_spool.py`
  - `FinalizedRecording` dataclass:
    - `audio_int16: np.ndarray`
    - `sample_rate: int`
    - `channels: int`
    - `start_ts: float`
    - `end_ts: float`
    - `chunks: int`
  - `RecordingJob` dataclass:
    - `seq: int`
    - `session_id: str`
    - `start_ts/end_ts/duration_ms`
    - `wav_path: Path`
    - `meta_path: Path`
  - `RecordingSpool`:
    - root dir: `output/spool`
    - subdirs: `queued/`, `processing/`, `failed/`
    - `enqueue(recording) -> RecordingJob`:
      - disk checks (soft + hard)
      - atomic write WAV + JSON sidecar
      - returns job only after final rename
    - `next_seq()`:
      - monotonic sequence per session
      - on startup, scan existing filenames to avoid collisions
  - Exceptions:
    - `DiskLowError`
    - `SpoolWriteError`

Acceptance checks:
- Creating 3 jobs results in filenames `00000001_...`, `00000002_...`, `00000003_...`.
- Jobs written with `*.tmp` then atomically renamed.
- Metadata JSON contains at least: seq, timestamps, sample_rate, num_samples.

### 1.2 Create sequential transcription pipeline
- Add `core/transcription_pipeline.py`
  - `TranscriptionPipeline`:
    - single worker thread
    - internal queue of `RecordingJob`
    - `start()`, `stop()`, `enqueue(job)`
    - `recover()` on startup:
      - move `processing/` back to `queued/` (or requeue in-place)
      - scan `queued/` and enqueue in filename order
    - processing lifecycle:
      1) move job to `processing/`
      2) load WAV -> `np.int16`
      3) call `Transcriber.transcribe(...)`
      4) on success: delete WAV+JSON
      5) on failure: move to `failed/` and write `*.error.txt`
    - emits callbacks for GUI:
      - `on_job_started(job)`
      - `on_job_done(job, result)`
      - `on_job_failed(job, error_str, failed_path)`
      - `on_backlog_changed(queued_count, processing_count)`

Acceptance checks:
- Pipeline processes jobs strictly in filename order.
- On error, job ends up in `failed/` and next job is processed.
- On restart, leftover jobs in `queued/` are picked up automatically.

---

## Phase 2: AudioCapture Emits Timestamped Recordings

### 2.1 Add timestamped callback
- Update `core/audio_capture.py`
  - keep existing `on_audio_ready(np.ndarray)` for compatibility
  - add `on_recording_ready(FinalizedRecording)` (new)
  - compute timestamps:
    - `start_ts = first_chunk.timestamp`
    - `end_ts = last_chunk.timestamp + (len(last_chunk.data)/sample_rate)`

Acceptance checks:
- FinalizedRecording includes correct timestamps and sample counts.

### 2.2 Enforce max segment length during continuous speech
- Use `AudioSettings.max_recording_duration` as hard segment cap.
- While in `RECORDING`, if buffered duration exceeds max:
  - finalize current buffer
  - continue in `RECORDING` state (no drop, no waiting for silence)

Acceptance checks:
- A long continuous speech stream yields multiple sequential recordings.
- No unbounded buffer growth.

---

## Phase 3: GUI Switch to Spool + Pipeline

### 3.1 Remove “skip if busy” logic
- Update `gui/main_window.py`
  - remove `TranscriptionWorker` usage for per-chunk transcription
  - wire `AudioCapture.on_recording_ready` to `RecordingSpool.enqueue` then `TranscriptionPipeline.enqueue`
  - update status bar with queue metrics:
    - `Queue: <queued>` and `Processing: <seq>`

Acceptance checks:
- No code path logs "skipping this chunk".
- Backlog grows when GPU/CPU is slow; recordings are still enqueued.

### 3.2 Disk-low UX
- If spool refuses enqueue:
  - stop recording immediately
  - show a modal error explaining:
    - disk is low
    - where spool lives (`output/spool/...`)
    - failed jobs are kept in `output/spool/failed/`

---

## Phase 4: Thread Safety / Model Hot-Swap

### 4.1 Prevent unload during transcription
- Update `core/transcriber.py`
  - ensure `unload_model()` cannot run concurrently with `transcribe()`
  - ensure `load_model()` cannot run concurrently with `transcribe()`

Acceptance checks:
- Rapid model switching while jobs are processing does not crash.

---

## Phase 5: Config + Tests

### 5.1 Config
- Update `config/settings.py` + `app_config.json`:
  - `spool.dir` default: `output/spool`
  - `spool.soft_min_free_mb` (stop soon)
  - `spool.hard_reserve_mb` (refuse enqueue)

### 5.2 Tests
- Add unit tests for:
  - spool ordering + naming
  - pipeline delete-on-success
  - pipeline move-to-failed-on-error
  - pipeline recovery scan
  - disk-low refusal behavior
- Update existing tests that assume immediate transcription.

---

## Progress Log

- [x] Implement spool writer (`core/recording_spool.py`)
- [x] Implement pipeline worker (`core/transcription_pipeline.py`)
- [x] Update `core/audio_capture.py` to emit timestamped recordings and segment long speech
- [x] Update `gui/main_window.py` to use spool+pipeline and remove skip logic
- [x] Update `core/transcriber.py` load/unload thread safety
- [x] Add config fields for spool + disk thresholds
- [ ] Add dedicated spool/pipeline unit tests (ordering, recovery, delete-on-success, move-to-failed)
