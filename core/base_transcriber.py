"""
Base Transcriber Module
Abstract base class for ASR transcriber implementations.
"""

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
    def transcribe(self, audio_data: np.ndarray, is_partial: bool = False) -> Optional[TranscriptionResult]:
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


class EngineNotAvailableError(Exception):
    """Raised when an ASR engine is not available."""
    pass


class ModelLoadError(Exception):
    """Raised when model fails to load."""
    pass


class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass
