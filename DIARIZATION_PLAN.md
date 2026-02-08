# Speaker Diarization Implementation Plan

## Overview

Implementation of speaker diarization for CripIt using whisper.cpp's `small.en-tdrz` model. This provides fully offline, local speaker segmentation for recordings.

**Use Case:** Non-real-time diarization of recorded/transcribed audio files  
**Speakers:** 2 people (configurable)  
**Display Format:** Option A - Inline speaker labels  
**Control:** Manual button in Converter tab  
**Storage:** Permanent with UI delete capability  
**Offline:** 100% offline after initial model download

---

## Architecture

### Two-Stage Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Audio File (WAV)                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
           ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│ Stage 1: Diarization │         │ Stage 2: High-Quality│
│   small.en-tdrz     │         │   Transcription      │
│   (466 MB)          │         │   large-v3-turbo     │
│                     │         │   (809 MB)           │
│ Output:             │         │                      │
│ - Speaker turns     │         │ Output:              │
│ - Timestamps        │         │ - Accurate text      │
│ - Rough text        │         │ - Word-level timing  │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           └───────────────┬───────────────┘
                           │
                           ▼
            ┌─────────────────────────────┐
            │   Alignment & Combination   │
            │                             │
            │ Match timestamps from       │
            │ Stage 1 with accurate text  │
            │ from Stage 2                │
            └──────────────┬──────────────┘
                           │
                           ▼
            ┌─────────────────────────────┐
            │  Final Output with Speakers │
            │                             │
            │ [Speaker 1]: Hello...       │
            │ [Speaker 2]: Hi there...    │
            └─────────────────────────────┘
```

---

## Components to Implement

### 1. Model Definition

**File:** `core/model_manager.py`

Add to MODELS dictionary:

```python
"small.en-tdrz": ModelInfo(
    name="small.en-tdrz",
    file="ggml-small.en-tdrz.bin",
    size_mb=466,
    params="244M",
    url="https://huggingface.co/akashmjn/tinydiarize-whisper.cpp/resolve/main/ggml-small.en-tdrz.bin",
    description="Speaker diarization model - detects speaker turns"
),
```

**Model Details:**
- Size: 466 MB
- Parameters: 244M
- Type: Fine-tuned small.en with special tokens
- Repository: https://huggingface.co/akashmjn/tinydiarize-whisper.cpp
- License: MIT
- Language: English only

**Download Command:**
```bash
./models/download-ggml-model.sh small.en-tdrz
```

---

### 2. Diarization Core Module

**File:** `core/diarization.py`

**Class:** `DiarizationProcessor`

**Responsibilities:**
- Load and manage small.en-tdrz model
- Process audio to detect speaker turns
- Extract speaker segments with timestamps
- Store results in structured format

**Key Methods:**

```python
class DiarizationProcessor:
    def __init__(self, model_manager: ModelManager):
        """Initialize with model manager for loading small.en-tdrz."""
        
    def load_model(self) -> bool:
        """Load small.en-tdrz model via pywhispercpp with tdrz_enable."""
        
    def process_audio(self, audio_path: Path) -> DiarizationResult:
        """
        Process audio file and return speaker segments.
        
        Args:
            audio_path: Path to WAV file
            
        Returns:
            DiarizationResult containing:
            - speakers: int (detected speaker count)
            - segments: List[SpeakerSegment]
            - raw_transcription: str
            
        Uses small.en-tdrz with tdrz_enable=True parameter.
        """
        
    def save_result(self, audio_path: Path, result: DiarizationResult) -> Path:
        """Save diarization result to JSON file in data/diarization/."""
        
    def load_result(self, audio_path: Path) -> Optional[DiarizationResult]:
        """Load previously saved diarization result."""
```

**Data Structures:**

```python
@dataclass
class SpeakerSegment:
    speaker: int  # 1, 2, 3, etc.
    start_time: float  # seconds
    end_time: float  # seconds
    text: str  # transcription from small.en-tdrz
    
@dataclass
class DiarizationResult:
    audio_file: Path
    speakers: int
    segments: List[SpeakerSegment]
    processed_at: datetime
    model: str  # "small.en-tdrz"
```

---

### 3. Diarization + Transcription Combiner

**File:** `core/diarization_transcriber.py`

**Class:** `DiarizationTranscriber`

**Responsibilities:**
- Combine diarization output with high-quality transcription
- Align speaker labels from small.en-tdrz with accurate text from large-v3-turbo
- Handle edge cases (overlapping speech, gaps)

**Key Methods:**

```python
class DiarizationTranscriber:
    def __init__(self, 
                 diarization_processor: DiarizationProcessor,
                 transcription_pipeline: TranscriptionPipeline):
        """Initialize with both processors."""
        
    def transcribe_with_diarization(
        self, 
        audio_path: Path,
        high_quality_model: str = "large-v3-turbo",
        progress_callback: Optional[Callable] = None
    ) -> CombinedResult:
        """
        Two-stage transcription with speaker labels.
        
        Stage 1: Run diarization with small.en-tdrz
        Stage 2: Run high-quality transcription
        Stage 3: Align and combine
        
        Returns:
            CombinedResult with speaker labels and accurate text
        """
        
    def align_transcriptions(
        self,
        diarization_result: DiarizationResult,
        hq_transcription: TranscriptionResult
    ) -> List[AlignedSegment]:
        """
        Align diarization segments with high-quality transcription.
        
        Algorithm:
        1. For each diarization segment, find overlapping HQ segments
        2. Use timestamp matching (>50% overlap threshold)
        3. Assign speaker label to HQ text
        4. Handle gaps and overlaps
        """
```

**Alignment Algorithm Details:**

```
For each diarization segment D:
    For each HQ transcription segment H:
        Calculate overlap = intersection(D.time, H.time)
        If overlap > 50% of min(D.duration, H.duration):
            Assign D.speaker to H
            
Result: HQ segments now have speaker labels
```

**Data Structures:**

```python
@dataclass
class AlignedSegment:
    speaker: int
    start_time: float
    end_time: float
    text: str  # from high-quality model
    confidence: float
    
@dataclass
class CombinedResult:
    audio_file: Path
    speakers: int
    segments: List[AlignedSegment]
    diarization_model: str
    transcription_model: str
    processed_at: datetime
```

---

### 4. Storage System

**Storage Location:** `data/diarization/`

**File Format:** JSON sidecar files

**Naming Convention:** `{audio_filename}.diarization.json`

**Example:**
```
data/
├── converted/
│   └── meeting_20260208_120000.wav
└── diarization/
    └── meeting_20260208_120000.wav.diarization.json
```

**JSON Schema:**

```json
{
  "version": "1.0",
  "audio_file": "meeting_20260208_120000.wav",
  "audio_hash": "sha256:abc123...",
  "processed_at": "2026-02-08T12:05:30",
  "diarization": {
    "model": "small.en-tdrz",
    "speakers": 2,
    "segments": [
      {
        "speaker": 1,
        "start": 0.0,
        "end": 3.8,
        "text": "Hello how are you?"
      },
      {
        "speaker": 2,
        "start": 3.8,
        "end": 6.2,
        "text": "I'm good thanks."
      }
    ]
  },
  "transcription": {
    "model": "large-v3-turbo",
    "segments": [
      {
        "speaker": 1,
        "start": 0.0,
        "end": 3.8,
        "text": "Hello, how are you doing today?"
      },
      {
        "speaker": 2,
        "start": 3.8,
        "end": 6.2,
        "text": "I'm doing well, thank you for asking."
      }
    ]
  }
}
```

**Storage Manager:**

```python
class DiarizationStorage:
    def __init__(self, storage_dir: Path = Path("data/diarization")):
        
    def save(self, audio_path: Path, result: CombinedResult) -> Path:
        """Save combined result to JSON."""
        
    def load(self, audio_path: Path) -> Optional[CombinedResult]:
        """Load result if exists."""
        
    def exists(self, audio_path: Path) -> bool:
        """Check if diarization exists for audio file."""
        
    def delete(self, audio_path: Path) -> bool:
        """Delete diarization file."""
        
    def get_audio_hash(self, audio_path: Path) -> str:
        """Calculate SHA256 hash of audio file for integrity checking."""
```

---

### 5. UI Integration - Converter Tab

**File:** `gui/converter_tab.py`

**New Elements:**

1. **Speaker Detection Column** in file table
   - Shows speaker count (e.g., "2 speakers")
   - Or "Not analyzed" if no diarization yet

2. **"Analyze Speakers" Button** per file
   - Only shown if diarization doesn't exist
   - Click to run diarization process
   - Shows progress dialog

3. **"Show Diarization" Button** per file
   - Only shown if diarization exists
   - Opens dialog with formatted output

4. **"Delete Diarization" Button** per file
   - Removes diarization data
   - Keeps audio file

**Updated File Table Columns:**
1. Filename
2. Date
3. Size
4. Duration
5. **Speakers** ← New
6. Actions ← Updated with diarization buttons

**Diarization Dialog:**

```
┌─────────────────────────────────────────┐
│  Speaker Analysis - meeting.wav         │
├─────────────────────────────────────────┤
│                                         │
│  Detected Speakers: 2                   │
│  Processed: 2026-02-08 12:05            │
│  Models: small.en-tdrz + large-v3-turbo │
│                                         │
│  ─────────────────────────────────────  │
│                                         │
│  [00:00:00] [Speaker 1]: Hello, how    │
│  are you doing today?                   │
│                                         │
│  [00:00:04] [Speaker 2]: I'm doing     │
│  well, thank you.                       │
│                                         │
│  [00:00:08] [Speaker 1]: Great to      │
│  hear that.                             │
│                                         │
│  ─────────────────────────────────────  │
│                                         │
│  [Copy to Clipboard]  [Close]           │
│                                         │
└─────────────────────────────────────────┘
```

**Progress Dialog:**

```
Analyzing Speakers...

Step 1/3: Running diarization model...  [████████░░] 80%
Step 2/3: Running high-quality transcription...  [░░░░░░░░░░] 0%
Step 3/3: Aligning speakers...  [░░░░░░░░░░] 0%

Estimated time remaining: 45 seconds
[Cancel]
```

---

### 6. Configuration

**File:** `config/settings.py`

**New Settings Section:**

```python
@dataclass
class DiarizationConfig:
    enabled: bool = True
    default_speakers: int = 2  # Assumed speaker count
    auto_analyze: bool = False  # Auto-run on new files?
    storage_format: str = "json"  # json, sqlite
    models_dir: Path = Path("models")
    
    # Model settings
    diarization_model: str = "small.en-tdrz"
    transcription_model: str = "large-v3-turbo"
    
    # Processing settings
    align_threshold: float = 0.5  # 50% overlap for alignment
    min_segment_duration: float = 1.0  # seconds
```

---

## Implementation Steps

### Phase 1: Core Infrastructure

1. **Add Model Definition**
   - Update `core/model_manager.py` with small.en-tdrz
   - Test model download

2. **Create Diarization Module**
   - Implement `core/diarization.py`
   - Test with sample audio
   - Verify JSON output format

3. **Create Storage System**
   - Implement `core/diarization_storage.py`
   - Add to config
   - Test save/load/delete

### Phase 2: Transcription Integration

4. **Create Combiner Module**
   - Implement `core/diarization_transcriber.py`
   - Implement alignment algorithm
   - Test two-stage pipeline

5. **Update Pipeline**
   - Add diarization support to transcription pipeline
   - Add single-file processing with diarization

### Phase 3: UI Integration

6. **Update Converter Tab**
   - Add speakers column to table
   - Add diarization buttons
   - Implement progress dialog

7. **Create Dialogs**
   - Diarization display dialog
   - Progress dialog
   - Confirmation dialogs

### Phase 4: Testing & Polish

8. **Testing**
   - Test with 2-speaker audio
   - Test edge cases (overlapping speech, silence)
   - Test storage persistence
   - Test delete functionality

9. **Documentation**
   - Update README
   - Add diarization section
   - Document limitations

---

## Technical Details

### pywhispercpp Integration

The small.en-tdrz model requires special handling in pywhispercpp:

```python
from pywhispercpp.model import Model

# Load with diarization enabled
model = Model(
    "models/ggml-small.en-tdrz.bin",
    params={
        "language": "en",
        "tdrz_enable": True,  # Enable speaker turn detection
    }
)

# Transcribe
segments = model.transcribe(audio_data)

# Each segment has speaker_turn_next field
for segment in segments:
    text = segment.text
    is_speaker_change = segment.speaker_turn_next
```

### Timestamp Alignment

Challenge: small.en-tdrz and large-v3-turbo may have slightly different timestamps

Solution: Use fuzzy matching with overlap threshold

```python
def align_segments(diarization_segments, transcription_segments, threshold=0.5):
    aligned = []
    
    for d_seg in diarization_segments:
        best_match = None
        best_overlap = 0
        
        for t_seg in transcription_segments:
            # Calculate time overlap
            overlap_start = max(d_seg.start, t_seg.start)
            overlap_end = min(d_seg.end, t_seg.end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # Calculate overlap ratio
            min_duration = min(d_seg.duration, t_seg.duration)
            overlap_ratio = overlap_duration / min_duration
            
            if overlap_ratio > threshold and overlap_ratio > best_overlap:
                best_match = t_seg
                best_overlap = overlap_ratio
        
        if best_match:
            aligned.append(AlignedSegment(
                speaker=d_seg.speaker,
                start_time=best_match.start,
                end_time=best_match.end,
                text=best_match.text,
                confidence=best_overlap
            ))
    
    return aligned
```

---

## Limitations & Considerations

### Current Limitations

1. **English Only**
   - small.en-tdrz only supports English
   - Multilingual support would need different approach

2. **Speaker Count**
   - Currently optimized for 2 speakers
   - Can detect more but accuracy decreases
   - No global speaker clustering (just local turn detection)

3. **Model Size**
   - Requires 466MB for diarization model
   - Plus 809MB for high-quality transcription
   - Total ~1.3GB for both models

4. **Processing Time**
   - Two-stage process = 2x processing time
   - small.en-tdrz is fast but adds overhead
   - Not suitable for real-time

5. **Accuracy**
   - Speaker turn detection: ~97% precision, ~70% recall
   - May miss rapid speaker switches
   - Struggles with overlapping speech

### Future Improvements

1. **Caching**
   - Cache diarization results to avoid reprocessing
   - Invalidate if audio file changes (hash check)

2. **Batch Processing**
   - Process multiple files in background
   - Queue system for large batches

3. **Export Formats**
   - Support SRT, VTT subtitles with speaker labels
   - Export to JSON, CSV, TXT formats

4. **Speaker Recognition**
   - Add speaker identification (name tags)
   - Learn speaker voices from samples

5. **Visualization**
   - Waveform with speaker color coding
   - Timeline view of speaker segments

---

## File Structure

```
cripit/
├── core/
│   ├── __init__.py
│   ├── diarization.py              # NEW: DiarizationProcessor
│   ├── diarization_transcriber.py  # NEW: DiarizationTranscriber
│   ├── diarization_storage.py      # NEW: Storage manager
│   ├── model_manager.py            # MODIFIED: Add small.en-tdrz
│   └── transcription_pipeline.py   # MODIFIED: Add diarization support
├── gui/
│   ├── __init__.py
│   ├── converter_tab.py            # MODIFIED: Add diarization UI
│   └── diarization_dialog.py       # NEW: Display dialog
├── config/
│   └── settings.py                 # MODIFIED: Add diarization config
├── data/
│   ├── converted/                  # Converted audio files
│   └── diarization/                # NEW: Diarization JSON files
└── models/
    └── ggml-small.en-tdrz.bin      # NEW: Diarization model
```

---

## Dependencies

**New Dependencies:**
- None! Uses existing pywhispercpp

**Model Downloads:**
- `small.en-tdrz`: 466 MB
- Already have: `large-v3-turbo`: 809 MB

**Disk Space Required:**
- ~500MB for diarization model
- Variable for diarization data storage

---

## Success Criteria

1. ✓ Can detect 2 speakers in test recordings
2. ✓ Speaker labels appear inline with transcription
3. ✓ Data persists across app restarts
4. ✓ Can delete diarization data from UI
5. ✓ Processing shows progress dialog
6. ✓ Results are stored in JSON format
7. ✓ No internet required after model download
8. ✓ Accurate transcription from large-v3-turbo preserved

---

## Testing Checklist

- [ ] Download small.en-tdrz model
- [ ] Process 2-speaker audio file
- [ ] Verify speaker detection accuracy
- [ ] Check alignment with high-quality transcription
- [ ] Test JSON storage and loading
- [ ] Test delete functionality
- [ ] Test UI buttons and dialogs
- [ ] Test with overlapping speech
- [ ] Test with single speaker (should detect 1 speaker)
- [ ] Test with 3+ speakers
- [ ] Test error handling (corrupt audio, missing model)
- [ ] Verify offline operation
- [ ] Test progress dialog cancellation

---

## References

1. **whisper.cpp tinydiarize PR:** https://github.com/ggerganov/whisper.cpp/pull/1058
2. **tinydiarize repository:** https://github.com/akashmjn/tinydiarize
3. **Model download:** https://huggingface.co/akashmjn/tinydiarize-whisper.cpp
4. **whisper.cpp documentation:** https://github.com/ggerganov/whisper.cpp
5. **pywhispercpp:** https://github.com/abdeladim-s/pywhispercpp

---

## Notes for Implementation

1. **Model Loading:** Ensure small.en-tdrz is loaded with `tdrz_enable=True` parameter
2. **Memory Management:** Unload model after processing to free RAM
3. **Error Handling:** Handle cases where diarization fails gracefully
4. **Threading:** Run diarization in background thread to avoid UI freezing
5. **Progress Updates:** Provide detailed progress (Stage 1/3, etc.)
6. **Integrity Checks:** Verify audio file hasn't changed before loading cached results
7. **Speaker Count:** Default to 2, but allow system to detect actual count
8. **Cleanup:** Provide "Clear All Diarization Data" option in settings

---

**Status:** Ready for implementation  
**Priority:** High  
**Estimated Implementation Time:** 2-3 days  
**Complexity:** Medium
