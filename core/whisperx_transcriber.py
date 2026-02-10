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
    
    def transcribe(self, audio_data: np.ndarray, is_partial: bool = False) -> Optional[TranscriptionResult]:
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
            if self._align_model and result.get("segments"):
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
                try:
                    diarize_segments = self._diarize_model(audio_data)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    
                    # Extract speaker info
                    for seg in result.get("segments", []):
                        if "speaker" in seg:
                            speakers.append({
                                "speaker": seg["speaker"],
                                "start": seg["start"],
                                "end": seg["end"],
                                "text": seg["text"]
                            })
                except Exception as e:
                    logger.error(f"Diarization failed: {e}")
            
            # Build text
            text = " ".join([seg.get("text", "") for seg in result.get("segments", [])]).strip()
            
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
                segments=result.get("segments"),
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


def create_whisperx_transcriber(config=None, **kwargs) -> WhisperXTranscriber:
    """Create WhisperXTranscriber instance from config or kwargs."""
    if config:
        return WhisperXTranscriber(
            model_name=config.model.whisperx_model,
            device=config.model.whisperx_device,
            compute_type=config.model.whisperx_compute_type,
            language=config.model.language,
            enable_diarization=config.model.whisperx_diarize,
            hf_token=config.model.whisperx_hf_token,
        )
    else:
        return WhisperXTranscriber(**kwargs)
