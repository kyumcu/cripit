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
whisperx = None  # type: ignore[assignment]
torch = None  # type: ignore[assignment]
try:
    import whisperx
    import torch
    WHISPERX_AVAILABLE = True
    logger.info("WhisperX available")
except ImportError as e:
    logger.warning(f"WhisperX not available: {e}")


def _maybe_allowlist_torch_safe_globals_for_whisperx() -> bool:
    """Best-effort allowlist for PyTorch 2.6+ weights_only loads.

    Some WhisperX dependency checkpoints include OmegaConf objects.
    In PyTorch 2.6, torch.load defaults weights_only=True and may reject
    these types unless allowlisted.
    """
    try:
        if not WHISPERX_AVAILABLE:
            return False

        ser = getattr(torch, "serialization", None)
        add_safe = getattr(ser, "add_safe_globals", None)
        if add_safe is None:
            return False

        # Only import if installed; these are safe types to allowlist.
        from omegaconf.listconfig import ListConfig  # type: ignore
        from omegaconf.dictconfig import DictConfig  # type: ignore
        from omegaconf.base import ContainerMetadata  # type: ignore

        add_safe([ListConfig, DictConfig, ContainerMetadata])
        return True
    except Exception:
        return False


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
                 vad_method: str = "silero",
                 enable_diarization: bool = False,
                 hf_token: Optional[str] = None,
                 cpu_threads: int = 0,
                 num_workers: int = 1):
        """
        Initialize WhisperX transcriber.
        
        Args:
            model_name: WhisperX model (tiny, base, small, medium, large-v3)
            device: Device to use ("cuda" or "cpu")
            compute_type: Compute precision ("int8", "float16", "float32")
            language: Language code (e.g., 'en') or None for auto-detect
            vad_method: VAD method for WhisperX ("silero" or "pyannote")
            enable_diarization: Whether to enable speaker diarization
            hf_token: HuggingFace token (required for diarization)
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.vad_method = vad_method
        self.enable_diarization = enable_diarization
        self.hf_token = hf_token
        self.cpu_threads = int(cpu_threads) if cpu_threads is not None else 0
        self.num_workers = int(num_workers) if num_workers is not None else 1
        
        # Model instances
        self._model = None
        self._align_model = None
        self._diarize_model = None
        self._model_loaded = False

        # Last error (best-effort, for UI / CLI troubleshooting)
        self.last_error: Optional[str] = None
        
        # Threading lock
        self._transcription_lock = threading.Lock()
        
        # Stats
        self._total_transcriptions = 0
        self._total_audio_seconds = 0.0
        self._total_processing_seconds = 0.0
        
        logger.info(
            "WhisperXTranscriber initialized: %s on %s (vad=%s cpu_threads=%s workers=%s)",
            model_name,
            device,
            vad_method,
            self.cpu_threads,
            self.num_workers,
        )

    def load_model(self) -> bool:
        """Load WhisperX model."""
        if not WHISPERX_AVAILABLE:
            logger.error("Cannot load model - WhisperX not available")
            return False
        
        if self._model_loaded:
            logger.info("Model already loaded")
            return True
        
        try:
            self.last_error = None

            # Helpful debug context for failures.
            try:
                torch_version = getattr(torch, "__version__", "unknown") if torch else "missing"
                logger.info(
                    "WhisperX load_model: model=%s device=%s compute=%s language=%s vad=%s torch=%s",
                    self.model_name,
                    self.device,
                    self.compute_type,
                    (self.language or "auto"),
                    self.vad_method,
                    torch_version,
                )
            except Exception:
                pass

            # Preemptively apply a safe-globals allowlist when available.
            if _maybe_allowlist_torch_safe_globals_for_whisperx():
                logger.info("Applied PyTorch safe-globals allowlist for OmegaConf")

            def _load_with_vad(vad_method: str):
                # NOTE: whisperx.asr.load_model accepts `threads` (not cpu_threads/num_workers)
                # in the WhisperX version we vendor/use.
                threads = int(self.cpu_threads) if int(self.cpu_threads) > 0 else None
                kwargs = {
                    "compute_type": self.compute_type,
                    "language": self.language,
                    "vad_method": vad_method,
                }
                if threads is not None:
                    kwargs["threads"] = threads

                return whisperx.load_model(
                    self.model_name,
                    self.device,
                    **kwargs,
                )

            # Load main model (with VAD)
            try:
                self._model = _load_with_vad(self.vad_method)
            except Exception as e:
                msg = str(e)
                self.last_error = msg

                # Common failure mode on PyTorch 2.6+: weights_only safety blocks
                # deserializing some objects in dependency checkpoints (often VAD).
                if "Weights only load failed" in msg:
                    # Retry once after applying safe-globals allowlist (OmegaConf, etc.).
                    if _maybe_allowlist_torch_safe_globals_for_whisperx():
                        logger.warning(
                            "WhisperX model load failed due to PyTorch weights_only safety; retrying after allowlisting"
                        )
                        try:
                            self._model = _load_with_vad(self.vad_method)
                            msg = ""
                            self.last_error = None
                        except Exception as e2:
                            msg = str(e2)
                            self.last_error = msg

                    # If still failing and we're using pyannote, fall back to silero VAD.
                    if self.vad_method == "pyannote" and "Weights only load failed" in msg:
                        logger.warning(
                            "WhisperX load_model still failing with pyannote VAD; retrying with silero VAD"
                        )
                        self._model = _load_with_vad("silero")
                        self.vad_method = "silero"
                        self.last_error = None

                    if self._model is None:
                        raise
                else:
                    raise
            
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
            self.last_error = None
            logger.info("✓ WhisperX model loaded successfully")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.exception("Failed to load WhisperX model")
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
            # WhisperX typically returns a top-level "text" plus per-segment "text".
            # In practice we've seen cases where segment text is empty/None while
            # the top-level text is populated (or vice-versa), so prefer the
            # top-level field and fall back to segments.
            text = ""
            try:
                raw_text = result.get("text") if isinstance(result, dict) else None
                if isinstance(raw_text, str):
                    text = raw_text.strip()
            except Exception:
                text = ""

            if not text:
                seg_texts = []
                for seg in (result.get("segments", []) if isinstance(result, dict) else []):
                    try:
                        t = seg.get("text", "")
                        if t is None:
                            continue
                        t = str(t).strip()
                        if t:
                            seg_texts.append(t)
                    except Exception:
                        continue
                text = " ".join(seg_texts).strip()
            
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
            cpu_threads=getattr(config.model, "whisperx_cpu_threads", 0),
            num_workers=getattr(config.model, "whisperx_num_workers", 1),
        )
    else:
        return WhisperXTranscriber(**kwargs)
