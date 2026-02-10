# CripIt WhisperX Integration Plan

## Overview
This document outlines the step-by-step implementation plan to convert CripIt from using only whisper.cpp to supporting both whisper.cpp and WhisperX as ASR engines, with runtime switching capability.

## Configuration Summary

### User Requirements
- ✅ **Runtime engine switching** - Hot-swap engines without app restart
- ✅ **Coexistence** - Both whisper.cpp and WhisperX available simultaneously
- ✅ **Memory warning** - UI warnings for high memory usage
- ✅ **Diarization = opt-in** - Disabled by default, no HuggingFace token required initially

### Default Behavior
| Feature | Default | Notes |
|---------|---------|-------|
| ASR Engine | `whispercpp` | Existing users unaffected |
| WhisperX Diarization | Disabled | Users can enable later if desired |
| Memory Warning | Enabled | Shows if < 4GB VRAM available |
| Runtime Switching | Enabled | Brief downtime during engine swap |
| Model Caching | Per-engine | whisper.cpp uses GGML files, WhisperX uses HF cache |

## Prerequisites

### System Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 8GB+ RAM (16GB+ recommended for WhisperX)
- 4GB+ VRAM (8GB+ recommended for WhisperX)

### Dependencies to Add
```txt
whisperx>=3.1.0
torch>=2.0.0
torchaudio>=2.0.0
# pyannote.audio>=3.1.0  # Optional: for speaker diarization
```

## Implementation Phases

### Phase 1: Foundation & Abstract Interface

#### Step 1.1: Create Abstract Base Class
**File:** `core/base_transcriber.py` (NEW)

Create an abstract base class that both transcribers will implement:

```python
from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    text: str
    language: Optional[str] = None
    confidence: float = 0.0
    duration: float = 0.0
    processing_time: float = 0.0
    segments: Optional[List[Dict]] = None
    is_partial: bool = False
    speakers: Optional[List[Dict]] = None  # For diarization

class BaseTranscriber(ABC):
    """Abstract base class for ASR transcriber implementations."""
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the ASR model. Returns True if successful."""
        pass
    
    @abstractmethod
    def transcribe(self, audio_data: np.ndarray) -> Optional[TranscriptionResult]:
        """Transcribe audio data."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload model to free memory."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if transcriber is ready (model loaded)."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get transcription statistics."""
        pass
    
    @abstractmethod
    def get_device_name(self) -> str:
        """Return device name (CPU/GPU)."""
        pass
```

#### Step 1.2: Update Configuration Settings
**File:** `config/settings.py` (MODIFY)

Add new settings to `ModelSettings` class:

```python
@dataclass
class ModelSettings:
    """Model configuration settings."""
    # Existing settings
    default_model: str = "large-v3-turbo"
    available_models: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {...})
    language: Optional[str] = None
    translate: bool = False
    n_threads: int = 4
    max_context: int = -1
    use_cuda: bool = True
    cuda_device: int = 0
    cuda_fallback_to_cpu: bool = True
    cuda_warn_on_fallback: bool = True
    
    # NEW: WhisperX settings
    asr_engine: str = "whispercpp"  # Options: "whispercpp", "whisperx"
    whisperx_model: str = "large-v3"  # Options: tiny, base, small, medium, large-v3
    whisperx_compute_type: str = "int8"  # Options: int8, float16, float32
    whisperx_diarize: bool = False  # Disabled by default (opt-in)
    whisperx_hf_token: Optional[str] = None  # Required only if diarization enabled
    whisperx_device: str = "cuda"  # Options: "cuda", "cpu"
```

Update `save_config()` method to include new settings.

#### Step 1.3: Update Requirements
**File:** `requirements.txt` (MODIFY)

Add WhisperX dependencies:

```txt
# ASR Engines (choose one or both)
pywhispercpp>=1.4.0  # whisper.cpp backend
whisperx>=3.1.0      # WhisperX backend (requires torch)
torch>=2.0.0
torchaudio>=2.0.0

# Optional: For WhisperX speaker diarization
# pyannote.audio>=3.1.0  # Uncomment and accept HF license to enable
```

---

### Phase 2: Refactor whisper.cpp Transcriber

#### Step 2.1: Rename and Refactor Existing Transcriber
**File:** `core/transcriber.py` → `core/whispercpp_transcriber.py` (RENAME + MODIFY)

1. Rename file from `transcriber.py` to `whispercpp_transcriber.py`
2. Rename class from `Transcriber` to `WhisperCppTranscriber`
3. Inherit from `BaseTranscriber`
4. Update imports at top of file

**Key changes:**
```python
from core.base_transcriber import BaseTranscriber, TranscriptionResult

class WhisperCppTranscriber(BaseTranscriber):
    """Speech-to-text transcriber using whisper.cpp via pywhispercpp."""
    
    def transcribe(self, audio_data: np.ndarray, 
                   is_partial: bool = False) -> Optional[TranscriptionResult]:
        # Existing implementation
        # Add speakers field (empty list since whisper.cpp doesn't support diarization)
        result = TranscriptionResult(
            text=text,
            language=detected_language,
            confidence=0.0,
            duration=duration,
            processing_time=processing_time,
            segments=[...],
            is_partial=is_partial,
            speakers=[]  # Empty list for compatibility
        )
        return result
```

5. Update factory function at bottom:
```python
def create_whispercpp_transcriber(config=None, model_path: Optional[str] = None) -> WhisperCppTranscriber:
    """Create WhisperCppTranscriber instance from config or defaults."""
```

6. Update `MultiModelTranscriber` to use `WhisperCppTranscriber`

---

### Phase 3: Implement WhisperX Transcriber

#### Step 3.1: Create WhisperX Transcriber
**File:** `core/whisperx_transcriber.py` (NEW)

Full implementation:

```python
"""
WhisperX Transcriber Module
Speech-to-text using WhisperX with optional speaker diarization.
"""

import logging
import threading
import time
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from core.base_transcriber import BaseTranscriber, TranscriptionResult

logger = logging.getLogger(__name__)

# Lazy import to avoid import errors if whisperx not installed
WHISPERX_AVAILABLE = False
try:
    import whisperx
    import torch
    WHISPERX_AVAILABLE = True
    logger.info("WhisperX available")
except ImportError as e:
    logger.warning(f"WhisperX not available: {e}")


class WhisperXTranscriber(BaseTranscriber):
    """
    Speech-to-text transcriber using WhisperX.
    Supports word-level timestamps and optional speaker diarization.
    """
    
    def __init__(self,
                 model_name: str = "large-v3",
                 device: str = "cuda",
                 compute_type: str = "int8",
                 language: Optional[str] = None,
                 enable_diarization: bool = False,
                 hf_token: Optional[str] = None):
        """
        Initialize WhisperX transcriber.
        
        Args:
            model_name: WhisperX model (tiny, base, small, medium, large-v3)
            device: Device to use ("cuda" or "cpu")
            compute_type: Compute precision ("int8", "float16", "float32")
            language: Language code (e.g., 'en') or None for auto-detect
            enable_diarization: Whether to enable speaker diarization
            hf_token: HuggingFace token (required for diarization)
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.enable_diarization = enable_diarization
        self.hf_token = hf_token
        
        # Model instances
        self._model = None
        self._align_model = None
        self._diarize_model = None
        self._model_loaded = False
        
        # Threading lock
        self._transcription_lock = threading.Lock()
        
        # Stats
        self._total_transcriptions = 0
        self._total_audio_seconds = 0.0
        self._total_processing_seconds = 0.0
        
        logger.info(f"WhisperXTranscriber initialized: {model_name} on {device}")
    
    def load_model(self) -> bool:
        """Load WhisperX model."""
        if not WHISPERX_AVAILABLE:
            logger.error("Cannot load model - WhisperX not available")
            return False
        
        if self._model_loaded:
            logger.info("Model already loaded")
            return True
        
        try:
            logger.info(f"Loading WhisperX model: {self.model_name}")
            
            # Load main model
            self._model = whisperx.load_model(
                self.model_name,
                self.device,
                compute_type=self.compute_type,
                language=self.language
            )
            
            # Load alignment model for word-level timestamps
            if self.language:
                self._align_model, self._metadata = whisperx.load_align_model(
                    language_code=self.language,
                    device=self.device
                )
            
            # Load diarization model if enabled
            if self.enable_diarization:
                if not self.hf_token:
                    logger.warning("Diarization enabled but no HF token provided. "
                                 "Get token at https://huggingface.co/settings/tokens")
                else:
                    logger.info("Loading diarization model...")
                    self._diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=self.hf_token,
                        device=self.device
                    )
            
            self._model_loaded = True
            logger.info("✓ WhisperX model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load WhisperX model: {e}")
            return False
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        logger.info("Unloading WhisperX model...")
        self._model = None
        self._align_model = None
        self._diarize_model = None
        self._model_loaded = False
        
        # Clear GPU cache if using CUDA
        if self.device == "cuda" and WHISPERX_AVAILABLE:
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
        
        logger.info("Model unloaded")
    
    def transcribe(self, audio_data: np.ndarray, 
                   is_partial: bool = False) -> Optional[TranscriptionResult]:
        """
        Transcribe audio data using WhisperX.
        
        Args:
            audio_data: Numpy array of audio samples (16kHz, int16 or float32)
            is_partial: Whether this is a partial result
            
        Returns:
            TranscriptionResult or None on error
        """
        if not self._model_loaded or not self._model:
            logger.error("Cannot transcribe - model not loaded")
            return None
        
        if not WHISPERX_AVAILABLE:
            logger.error("Cannot transcribe - WhisperX not available")
            return None
        
        # Acquire lock
        if not self._transcription_lock.acquire(timeout=30.0):
            logger.error("Timed out waiting for transcription lock")
            return None
        
        try:
            start_time = time.time()
            
            # Normalize audio
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            duration = len(audio_data) / 16000  # Assuming 16kHz
            logger.info(f"Transcribing {duration:.2f}s with WhisperX...")
            
            # Transcribe
            result = self._model.transcribe(audio_data, batch_size=16)
            
            # Align for word-level timestamps
            if self._align_model and result["segments"]:
                result = whisperx.align(
                    result["segments"],
                    self._align_model,
                    self._metadata,
                    audio_data,
                    self.device,
                    return_char_alignments=False
                )
            
            speakers = []
            # Diarize if enabled
            if self.enable_diarization and self._diarize_model:
                diarize_segments = self._diarize_model(audio_data)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                
                # Extract speaker info
                for seg in result["segments"]:
                    if "speaker" in seg:
                        speakers.append({
                            "speaker": seg["speaker"],
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["text"]
                        })
            
            # Build text
            text = " ".join([seg["text"] for seg in result["segments"]]).strip()
            
            processing_time = time.time() - start_time
            rtf = processing_time / duration if duration > 0 else 0
            
            logger.info(f"Transcription complete: {len(text)} chars, RTF={rtf:.2f}x")
            
            # Update stats
            self._total_transcriptions += 1
            self._total_audio_seconds += duration
            self._total_processing_seconds += processing_time
            
            return TranscriptionResult(
                text=text,
                language=result.get("language"),
                confidence=0.0,  # WhisperX doesn't provide confidence scores
                duration=duration,
                processing_time=processing_time,
                segments=result["segments"],
                is_partial=is_partial,
                speakers=speakers if speakers else []
            )
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self._transcription_lock.release()
    
    def is_ready(self) -> bool:
        """Check if transcriber is ready."""
        return self._model_loaded and self._model is not None
    
    def get_device_name(self) -> str:
        """Return device name."""
        if self.device == "cuda":
            try:
                import torch
                return f"GPU ({torch.cuda.get_device_name(0)})"
            except:
                return "GPU"
        return "CPU"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transcription statistics."""
        avg_rtf = (self._total_processing_seconds / self._total_audio_seconds 
                   if self._total_audio_seconds > 0 else 0)
        
        return {
            'total_transcriptions': self._total_transcriptions,
            'total_audio_seconds': self._total_audio_seconds,
            'total_processing_seconds': self._total_processing_seconds,
            'avg_realtime_factor': avg_rtf,
            'model_loaded': self._model_loaded,
            'model_name': self.model_name,
            'using_gpu': self.device == "cuda",
            'device': self.get_device_name(),
            'diarization_enabled': self.enable_diarization,
        }
```

---

### Phase 4: Create Transcriber Factory

#### Step 4.1: Create Factory Module
**File:** `core/transcriber_factory.py` (NEW)

```python
"""
Transcriber Factory Module
Creates appropriate transcriber based on configuration.
"""

import logging
from typing import Optional
from config.settings import config
from core.base_transcriber import BaseTranscriber

logger = logging.getLogger(__name__)


def check_engine_availability() -> dict:
    """
    Check which ASR engines are available.
    
    Returns:
        Dict with engine availability status
    """
    engines = {
        "whispercpp": False,
        "whisperx": False
    }
    
    # Check whisper.cpp
    try:
        from pywhispercpp.model import Model
        engines["whispercpp"] = True
        logger.info("✓ whisper.cpp (pywhispercpp) available")
    except ImportError:
        logger.warning("✗ whisper.cpp (pywhispercpp) not available")
    
    # Check WhisperX
    try:
        import whisperx
        import torch
        engines["whisperx"] = True
        logger.info("✓ WhisperX available")
    except ImportError:
        logger.warning("✗ WhisperX not available")
    
    return engines


def create_transcriber(engine_type: Optional[str] = None) -> Optional[BaseTranscriber]:
    """
    Factory function to create transcriber based on engine type.
    
    Args:
        engine_type: "whispercpp", "whisperx", or None (uses config)
        
    Returns:
        BaseTranscriber instance or None if creation fails
    """
    engine = engine_type or config.model.asr_engine
    logger.info(f"Creating transcriber: {engine}")
    
    if engine == "whisperx":
        try:
            from core.whisperx_transcriber import WhisperXTranscriber
            
            transcriber = WhisperXTranscriber(
                model_name=config.model.whisperx_model,
                device=config.model.whisperx_device,
                compute_type=config.model.whisperx_compute_type,
                language=config.model.language,
                enable_diarization=config.model.whisperx_diarize,
                hf_token=config.model.whisperx_hf_token
            )
            
            logger.info(f"Created WhisperX transcriber: {config.model.whisperx_model}")
            return transcriber
            
        except Exception as e:
            logger.error(f"Failed to create WhisperX transcriber: {e}")
            return None
    
    else:  # whispercpp (default)
        try:
            from core.whispercpp_transcriber import WhisperCppTranscriber
            
            transcriber = WhisperCppTranscriber(
                model_path=config.get_model_path(),
                model_name=config.model.default_model,
                language=config.model.language,
                n_threads=config.model.n_threads,
                translate=config.model.translate,
                use_cuda=config.model.use_cuda,
                cuda_device=config.model.cuda_device,
            )
            
            logger.info(f"Created whisper.cpp transcriber: {config.model.default_model}")
            return transcriber
            
        except Exception as e:
            logger.error(f"Failed to create whisper.cpp transcriber: {e}")
            return None


def reload_transcriber(current_transcriber: Optional[BaseTranscriber], 
                       new_engine: str) -> Optional[BaseTranscriber]:
    """
    Hot-swap transcriber to a different engine.
    
    Args:
        current_transcriber: Currently active transcriber (will be unloaded)
        new_engine: New engine type to load ("whispercpp" or "whisperx")
        
    Returns:
        New transcriber instance or None on failure
    """
    logger.info(f"Reloading transcriber: {new_engine}")
    
    # Save current stats if available
    saved_stats = None
    if current_transcriber:
        saved_stats = current_transcriber.get_stats()
        logger.info("Unloading current transcriber...")
        current_transcriber.unload_model()
    
    # Update config
    config.model.asr_engine = new_engine
    config.save_config()
    
    # Create new transcriber
    new_transcriber = create_transcriber(new_engine)
    
    if new_transcriber and saved_stats:
        # Log transition info
        logger.info(f"Engine switched from {saved_stats.get('model_name', 'unknown')} "
                   f"to {new_engine}")
    
    return new_transcriber
```

---

### Phase 5: Update Model Manager

#### Step 5.1: Extend Model Manager for WhisperX
**File:** `core/model_manager.py` (MODIFY)

Add WhisperX model definitions:

```python
# Add after existing MODELS dict
WHISPERX_MODELS: Dict[str, ModelInfo] = {
    "tiny": ModelInfo(
        name="tiny",
        file="",  # WhisperX uses HF cache, no local file
        size_mb=150,
        params="39M",
        url="openai/whisper-tiny",
        description="Smallest model, fastest but lowest quality"
    ),
    "base": ModelInfo(
        name="base",
        file="",
        size_mb=290,
        params="74M",
        url="openai/whisper-base",
        description="Base model, good for testing"
    ),
    "small": ModelInfo(
        name="small",
        file="",
        size_mb=900,
        params="244M",
        url="openai/whisper-small",
        description="Small model, good balance"
    ),
    "medium": ModelInfo(
        name="medium",
        file="",
        size_mb=3000,
        params="769M",
        url="openai/whisper-medium",
        description="Medium model, good quality"
    ),
    "large-v3": ModelInfo(
        name="large-v3",
        file="",
        size_mb=6000,
        params="1.55B",
        url="openai/whisper-large-v3",
        description="Large model v3, best quality"
    ),
}

# Update __init__ to track engine type
def __init__(self, models_dir: Optional[Path] = None):
    # ... existing init code ...
    from config.settings import config
    self.engine_type = config.model.asr_engine

# Add method to check WhisperX model availability
def is_whisperx_model_cached(self, model_name: str) -> bool:
    """Check if WhisperX model is cached locally."""
    try:
        from pathlib import Path
        import os
        
        # Check HuggingFace cache
        cache_dir = Path.home() / ".cache" / "whisper"
        if not cache_dir.exists():
            return False
        
        # Look for model in cache
        model_pattern = f"*{model_name}*"
        matches = list(cache_dir.glob(model_pattern))
        return len(matches) > 0
    except Exception:
        return False

# Update list_models() to return appropriate models
def list_models(self) -> List[str]:
    """List available models for current engine."""
    if self.engine_type == "whisperx":
        return list(WHISPERX_MODELS.keys())
    else:
        return list(self.MODELS.keys())

# Add method to get model info for current engine
def get_current_model_info(self, model_name: str) -> Optional[ModelInfo]:
    """Get model info based on current engine."""
    if self.engine_type == "whisperx":
        return WHISPERX_MODELS.get(model_name)
    else:
        return self.MODELS.get(model_name)
```

---

### Phase 6: Update CUDA/Memory Utilities

#### Step 6.1: Enhance CUDA Utilities
**File:** `utils/cuda_utils.py` (MODIFY)

Add WhisperX-specific memory checking:

```python
def get_gpu_memory_info() -> Optional[Dict[str, int]]:
    """
    Get GPU memory information.
    
    Returns:
        Dict with 'free_mb', 'total_mb', 'used_mb' or None if CUDA unavailable
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        torch.cuda.synchronize()
        free_memory = torch.cuda.mem_get_info()[0]
        total_memory = torch.cuda.mem_get_info()[1]
        
        return {
            'free_mb': free_memory // (1024 * 1024),
            'total_mb': total_memory // (1024 * 1024),
            'used_mb': (total_memory - free_memory) // (1024 * 1024)
        }
    except Exception as e:
        logger.debug(f"Could not get GPU memory info: {e}")
        return None


def check_whisperx_compatibility() -> tuple:
    """
    Check if system can run WhisperX comfortably.
    
    Returns:
        Tuple of (compatible: bool, warning: str, recommendation: str)
    """
    warnings = []
    
    # Check GPU memory
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        if gpu_info['free_mb'] < 4096:  # Less than 4GB free
            warnings.append(f"Low GPU memory: {gpu_info['free_mb']}MB free. "
                          f"WhisperX large model requires ~4GB VRAM.")
    else:
        warnings.append("No CUDA GPU detected. WhisperX will run on CPU (slower).")
    
    # Check system RAM (approximate)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb < 8:
            warnings.append(f"Low system RAM: {ram_gb:.1f}GB. Recommended: 16GB+ for WhisperX.")
    except ImportError:
        pass
    
    if warnings:
        return (
            False,
            "\n".join(warnings),
            "Consider using whisper.cpp for lower resource usage, or use a smaller model like 'base' or 'tiny'."
        )
    
    return (True, "", "")


def estimate_whisperx_memory(model_name: str, compute_type: str) -> int:
    """
    Estimate VRAM requirement for WhisperX model.
    
    Returns:
        Estimated VRAM in MB
    """
    # Rough estimates based on model size
    base_memory = {
        "tiny": 1000,
        "base": 1500,
        "small": 2500,
        "medium": 4000,
        "large-v3": 6000
    }
    
    memory = base_memory.get(model_name, 4000)
    
    # Adjust for compute type
    if compute_type == "int8":
        memory = int(memory * 0.5)
    elif compute_type == "float16":
        memory = int(memory * 0.75)
    # float32 uses full memory
    
    return memory
```

---

### Phase 7: GUI Updates

#### Step 7.1: Update Settings Tab
**File:** `gui/settings_tab.py` (MODIFY)

Add new UI elements:

```python
# In __init__ or setup_ui method, add:

# Engine Selection Section
engine_group = QGroupBox("ASR Engine")
engine_layout = QFormLayout()

self.engine_combo = QComboBox()
self.engine_combo.addItem("whisper.cpp (Default)", "whispercpp")
self.engine_combo.addItem("⚡ WhisperX (Fast)", "whisperx")
self.engine_combo.currentIndexChanged.connect(self.on_engine_changed)
engine_layout.addRow("Engine:", self.engine_combo)

# Memory Warning Label
self.memory_warning_label = QLabel()
self.memory_warning_label.setStyleSheet("color: orange;")
self.memory_warning_label.setWordWrap(True)
self.memory_warning_label.hide()
engine_layout.addRow(self.memory_warning_label)

engine_group.setLayout(engine_layout)
main_layout.addWidget(engine_group)

# WhisperX Options Section (initially hidden)
self.whisperx_group = QGroupBox("WhisperX Options")
whisperx_layout = QFormLayout()

# Model selector
self.whisperx_model_combo = QComboBox()
self.whisperx_model_combo.addItems(["tiny", "base", "small", "medium", "large-v3"])
whisperx_layout.addRow("Model:", self.whisperx_model_combo)

# Compute type
self.compute_type_combo = QComboBox()
self.compute_type_combo.addItem("int8 (Fastest, lower quality)", "int8")
self.compute_type_combo.addItem("float16 (Balanced)", "float16")
self.compute_type_combo.addItem("float32 (Best quality, slowest)", "float32")
whisperx_layout.addRow("Compute Type:", self.compute_type_combo)

# Diarization checkbox (unchecked by default)
self.diarization_checkbox = QCheckBox("Enable speaker diarization (requires HuggingFace token)")
self.diarization_checkbox.setChecked(False)
self.diarization_checkbox.stateChanged.connect(self.on_diarization_changed)
whisperx_layout.addRow(self.diarization_checkbox)

# HF Token (hidden unless diarization enabled)
self.hf_token_input = QLineEdit()
self.hf_token_input.setPlaceholderText("Enter HuggingFace token (optional)")
self.hf_token_input.setEchoMode(QLineEdit.Password)
self.hf_token_label = QLabel("HF Token:")
self.hf_token_label.hide()
self.hf_token_input.hide()
whisperx_layout.addRow(self.hf_token_label, self.hf_token_input)

# Help link for HF token
self.hf_help_label = QLabel('<a href="https://huggingface.co/settings/tokens">Get token here</a>')
self.hf_help_label.setOpenExternalLinks(True)
self.hf_help_label.hide()
whisperx_layout.addRow("", self.hf_help_label)

self.whisperx_group.setLayout(whisperx_layout)
main_layout.addWidget(self.whisperx_group)
self.whisperx_group.hide()  # Hidden by default

# Methods to handle visibility

def on_engine_changed(self, index):
    """Show/hide WhisperX options based on engine selection."""
    engine = self.engine_combo.currentData()
    
    if engine == "whisperx":
        self.whisperx_group.show()
        self.check_memory_compatibility()
    else:
        self.whisperx_group.hide()
        self.memory_warning_label.hide()

def on_diarization_changed(self, state):
    """Show/hide HF token input based on diarization checkbox."""
    enabled = state == Qt.Checked
    self.hf_token_label.setVisible(enabled)
    self.hf_token_input.setVisible(enabled)
    self.hf_help_label.setVisible(enabled)

def check_memory_compatibility(self):
    """Check system compatibility and show warning if needed."""
    from utils.cuda_utils import check_whisperx_compatibility
    
    compatible, warning, recommendation = check_whisperx_compatibility()
    
    if not compatible:
        self.memory_warning_label.setText(
            f"⚠️ {warning}\n\nRecommendation: {recommendation}"
        )
        self.memory_warning_label.show()
    else:
        self.memory_warning_label.hide()

def save_settings(self):
    """Save settings including new WhisperX options."""
    # Existing save logic...
    
    # Save new settings
    config.model.asr_engine = self.engine_combo.currentData()
    config.model.whisperx_model = self.whisperx_model_combo.currentText()
    config.model.whisperx_compute_type = self.compute_type_combo.currentData()
    config.model.whisperx_diarize = self.diarization_checkbox.isChecked()
    
    # Only save token if diarization enabled
    if self.diarization_checkbox.isChecked():
        config.model.whisperx_hf_token = self.hf_token_input.text()
    else:
        config.model.whisperx_hf_token = None
    
    config.save_config()
```

#### Step 7.2: Update Main Window
**File:** `gui/main_window.py` (MODIFY)

Add engine indicator and switch button:

```python
# In setup_ui or status bar setup:

# Engine indicator in status bar
self.engine_label = QLabel("Engine: whisper.cpp")
self.statusBar().addPermanentWidget(self.engine_label)

# Memory indicator (for WhisperX)
self.memory_label = QLabel("")
self.statusBar().addPermanentWidget(self.memory_label)

# Add "Switch Engine" button to toolbar
self.switch_engine_action = QAction("⚡ Switch Engine", self)
self.switch_engine_action.triggered.connect(self.on_switch_engine)
self.toolbar.addAction(self.switch_engine_action)

def on_switch_engine(self):
    """Handle engine switch request."""
    from PyQt6.QtWidgets import QMessageBox
    
    current_engine = config.model.asr_engine
    new_engine = "whisperx" if current_engine == "whispercpp" else "whispercpp"
    
    # Confirmation dialog
    reply = QMessageBox.question(
        self,
        "Switch ASR Engine",
        f"Switch from {current_engine} to {new_engine}?\n\n"
        "This will reload the transcription model.",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    
    if reply == QMessageBox.StandardButton.Yes:
        self.perform_engine_switch(new_engine)

def perform_engine_switch(self, new_engine: str):
    """Perform the actual engine switch."""
    from core.transcriber_factory import reload_transcriber
    
    # Show loading indicator
    self.statusBar().showMessage(f"Switching to {new_engine}...")
    self.setEnabled(False)
    
    # Perform switch
    new_transcriber = reload_transcriber(self.transcriber, new_engine)
    
    if new_transcriber:
        self.transcriber = new_transcriber
        
        # Load new model
        if self.transcriber.load_model():
            self.update_engine_display()
            self.statusBar().showMessage(f"✓ Switched to {new_engine}", 3000)
        else:
            QMessageBox.critical(self, "Error", f"Failed to load {new_engine} model")
    else:
        QMessageBox.critical(self, "Error", f"Failed to create {new_engine} transcriber")
    
    self.setEnabled(True)

def update_engine_display(self):
    """Update UI to reflect current engine."""
    engine = config.model.asr_engine
    
    if engine == "whisperx":
        self.engine_label.setText(f"Engine: WhisperX ({config.model.whisperx_model})")
        # Show memory usage for WhisperX
        self.update_memory_display()
    else:
        self.engine_label.setText(f"Engine: whisper.cpp ({config.model.default_model})")
        self.memory_label.hide()

def update_memory_display(self):
    """Update memory usage display (WhisperX only)."""
    from utils.cuda_utils import get_gpu_memory_info
    
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        self.memory_label.setText(
            f"VRAM: {gpu_info['used_mb']}MB / {gpu_info['total_mb']}MB"
        )
        self.memory_label.show()
    else:
        self.memory_label.hide()
```

---

### Phase 8: Update Main Entry Point

#### Step 8.1: Update Main Application
**File:** `main.py` (MODIFY)

Update imports and initialization:

```python
# Replace existing transcriber import
# OLD: from core.transcriber import create_transcriber
# NEW:
from core.transcriber_factory import create_transcriber, check_engine_availability

def check_dependencies():
    """Check which ASR engines are available."""
    engines = check_engine_availability()
    
    if not engines["whispercpp"] and not engines["whisperx"]:
        QMessageBox.critical(
            None,
            "Missing Dependencies",
            "No ASR engine available.\n\n"
            "Please install at least one:\n"
            "• pip install pywhispercpp (for whisper.cpp)\n"
            "• pip install whisperx torch torchaudio (for WhisperX)"
        )
        return False
    
    # Check if preferred engine is available
    preferred = config.model.asr_engine
    if preferred == "whisperx" and not engines["whisperx"]:
        logger.warning("WhisperX not available, falling back to whisper.cpp")
        config.model.asr_engine = "whispercpp"
        config.save_config()
    
    return True

def main():
    # Existing setup code...
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create transcriber using factory
    transcriber = create_transcriber()
    if not transcriber:
        QMessageBox.critical(
            None,
            "Initialization Error",
            "Failed to create transcriber. Check logs for details."
        )
        sys.exit(1)
    
    # Load model
    if not transcriber.load_model():
        QMessageBox.warning(
            None,
            "Model Loading Failed",
            "Failed to load ASR model. The app may not function correctly."
        )
    
    # Create and show main window with transcriber
    window = MainWindow(transcriber)
    window.show()
    
    # Existing event loop code...
```

---

### Phase 9: Update Terminal Application

#### Step 9.1: Update TUI Entry Point
**File:** `terminal_app.py` (MODIFY)

Similar updates to use transcriber factory:

```python
# Replace import
from core.transcriber_factory import create_transcriber, check_engine_availability

def main():
    # Check engine availability
    engines = check_engine_availability()
    
    # Create transcriber
    transcriber = create_transcriber()
    if not transcriber:
        print("ERROR: Failed to create transcriber", file=sys.stderr)
        sys.exit(1)
    
    # Load model
    if not transcriber.load_model():
        print("WARNING: Failed to load model", file=sys.stderr)
    
    # Continue with existing TUI setup...
```

---

### Phase 10: Testing & Validation

#### Test Cases

1. **Basic Functionality**
   - [ ] whisper.cpp transcription works
   - [ ] WhisperX transcription works
   - [ ] Transcription results have consistent format
   - [ ] Word-level timestamps work (WhisperX)
   - [ ] Speaker diarization works when enabled (WhisperX)

2. **Engine Switching**
   - [ ] Can switch from whisper.cpp → WhisperX
   - [ ] Can switch from WhisperX → whisper.cpp
   - [ ] Model loads correctly after switch
   - [ ] No memory leaks during switching
   - [ ] Settings persist after restart

3. **Memory Management**
   - [ ] Memory warning shows when < 4GB VRAM
   - [ ] Memory warning shows when < 8GB RAM
   - [ ] GPU memory display works (WhisperX)
   - [ ] Model unloads properly to free memory

4. **Configuration**
   - [ ] Settings save correctly
   - [ ] Settings load correctly on restart
   - [ ] Default engine is whisper.cpp
   - [ ] Diarization defaults to disabled
   - [ ] HF token only saved when diarization enabled

5. **Edge Cases**
   - [ ] App works with only whisper.cpp installed
   - [ ] App works with only WhisperX installed
   - [ ] Graceful error if selected engine unavailable
   - [ ] Works without CUDA (CPU mode)
   - [ ] Disk spool compatibility with both engines

---

### Phase 11: Documentation

#### Step 11.1: Update README
Add section on ASR engines:

```markdown
## ASR Engines

CripIt supports two speech recognition engines:

### whisper.cpp (Default)
- Fully offline
- Lower memory usage
- GGML model format
- Good for resource-constrained devices

### WhisperX (Optional)
- Faster transcription
- Word-level timestamps
- Optional speaker diarization
- Requires more memory (4GB+ VRAM recommended)
- Optional: HuggingFace token for diarization

### Switching Engines
1. Go to Settings → ASR Engine
2. Select your preferred engine
3. Configure model and options
4. Click "Apply"

**Note:** First-time WhisperX users will need to download models from HuggingFace.
```

#### Step 11.2: Create Migration Guide
**File:** `docs/MIGRATION.md`

Document any breaking changes and migration steps for existing users.

---

## Implementation Checklist

### Week 1: Foundation
- [ ] Create `core/base_transcriber.py`
- [ ] Update `config/settings.py`
- [ ] Update `requirements.txt`

### Week 2: Transcribers
- [ ] Rename/refactor `core/transcriber.py` → `core/whispercpp_transcriber.py`
- [ ] Create `core/whisperx_transcriber.py`
- [ ] Test both transcribers independently

### Week 3: Integration
- [ ] Create `core/transcriber_factory.py`
- [ ] Update `core/model_manager.py`
- [ ] Update `utils/cuda_utils.py`
- [ ] Update `main.py`
- [ ] Update `terminal_app.py`

### Week 4: GUI
- [ ] Update `gui/settings_tab.py`
- [ ] Update `gui/main_window.py`
- [ ] Add engine indicator
- [ ] Add memory warning
- [ ] Test UI interactions

### Week 5: Testing & Polish
- [ ] Run all test cases
- [ ] Update documentation
- [ ] Performance benchmarking
- [ ] Bug fixes

---

## Notes

### Memory Requirements
- **whisper.cpp**: 2-6GB RAM (depending on model)
- **WhisperX**: 4-8GB VRAM + 8GB+ RAM recommended
- **Diarization**: Additional 2GB VRAM when enabled

### Model Downloads
- **whisper.cpp**: Manual download or auto-download on first use
- **WhisperX**: Automatic download from HuggingFace on first use

### Known Limitations
1. WhisperX diarization requires internet connection (first time) and HF token
2. Engine switching requires model reload (brief downtime)
3. WhisperX models are larger than whisper.cpp GGML models

---

**Ready to begin implementation!** Start with Phase 1, Step 1.1.
