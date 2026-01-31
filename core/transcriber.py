"""
Transcriber Module
Handles speech-to-text conversion using whisper.cpp via pywhispercpp
"""

import logging
import threading
import queue
import time
import numpy as np
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import pywhispercpp
try:
    from pywhispercpp.model import Model
    from pywhispercpp.constants import WHISPER_SAMPLE_RATE as SAMPLE_RATE
    PYWHISPERCPP_AVAILABLE = True
    logger.info("pywhispercpp available")
except ImportError as e:
    PYWHISPERCPP_AVAILABLE = False
    logger.warning(f"pywhispercpp not available: {e}")
    Model = None
    SAMPLE_RATE = 16000


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


class Transcriber:
    """
    Speech-to-text transcriber using whisper.cpp
    
    Supports multiple models and real-time transcription.
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 model_name: str = "large-v3-turbo",
                 language: Optional[str] = None,
                 n_threads: int = 4,
                 translate: bool = False,
                 use_cuda: bool = True,
                 cuda_device: int = 0,
                 gpu_layers: int = -1):
        """
        Initialize transcriber.

        Args:
            model_path: Direct path to GGML model file
            model_name: Model name (if model_path not provided)
            language: Language code (e.g., 'en', 'fr') or None for auto-detect
            n_threads: Number of threads for inference
            translate: Whether to translate to English
            use_cuda: Whether to try using GPU/CUDA if available
            cuda_device: GPU device ID (for multi-GPU systems)
            gpu_layers: Number of layers to offload to GPU (-1 = auto/all)
        """
        self.model_path = model_path
        self.model_name = model_name
        self.language = language
        self.n_threads = n_threads
        self.translate = translate
        self.use_cuda = use_cuda
        self.cuda_device = cuda_device
        self.gpu_layers = gpu_layers

        logger.info("=" * 50)
        logger.info("Initializing Transcriber")
        logger.info("=" * 50)
        logger.info(f"Model: {model_name}")
        logger.info(f"Language: {language or 'auto-detect'}")
        logger.info(f"Threads: {n_threads}")
        logger.info(f"Translate: {translate}")
        logger.info(f"Use CUDA: {use_cuda}")
        logger.info(f"CUDA Device: {cuda_device}")
        logger.info(f"GPU Layers: {gpu_layers}")
        
        # Model instance
        self._model: Optional[Any] = None
        self._model_loaded = False
        
        # Threading lock to prevent concurrent transcription (whisper.cpp is not thread-safe)
        self._transcription_lock = threading.Lock()
        self._is_transcribing = False
        
        # Threading
        self._transcription_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Callbacks
        self.on_transcription: Optional[Callable[[TranscriptionResult], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_model_loaded: Optional[Callable[[], None]] = None
        
        # Stats
        self._total_transcriptions = 0
        self._total_audio_seconds = 0.0
        self._total_processing_seconds = 0.0
        
        # GPU/Device info
        self._using_gpu = False
        self._device_name = "CPU"
        
        logger.info("Transcriber initialized (model not loaded yet)")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the whisper.cpp model.
        
        Args:
            model_path: Path to GGML model file
            
        Returns:
            True if loaded successfully
        """
        if not PYWHISPERCPP_AVAILABLE:
            logger.error("Cannot load model - pywhispercpp not available")
            return False

        if Model is None:
            logger.error("Cannot load model - pywhispercpp Model class not available")
            return False
        
        if self._model_loaded:
            logger.info("Model already loaded")
            return True

        acquired = self._transcription_lock.acquire(timeout=10.0)
        if not acquired:
            logger.error("Timed out waiting to load model (transcription busy)")
            return False
        
        model_file = model_path or self.model_path
        
        if not model_file:
            logger.error("No model path provided")
            return False
        
        model_file = Path(model_file)
        
        if not model_file.exists():
            logger.error(f"Model file not found: {model_file}")
            return False
        
        try:
            # Decide once whether fallback is allowed.
            fallback_allowed = False
            try:
                from config.settings import config
                fallback_allowed = bool(getattr(config.model, "cuda_fallback_to_cpu", False))
            except Exception:
                fallback_allowed = False

            while True:
                try:
                    logger.info(f"Loading model from: {model_file}")
                    logger.info(f"Model file size: {model_file.stat().st_size / (1024**3):.2f} GB")

                    # Check CUDA availability if requested
                    using_gpu = False
                    device_name = "CPU"

                    if self.use_cuda:
                        try:
                            from utils.cuda_utils import CUDAManager
                            cuda_manager = CUDAManager()
                            cuda_status = cuda_manager.detect_cuda()

                            if cuda_status.available and getattr(cuda_status, "pywhispercpp_cuda", False):
                                # pywhispercpp does not provide a reliable runtime switch to force CPU when built with CUDA.
                                # If the model does not fit, attempting to load can abort the process.
                                fits, msg = cuda_manager.check_model_fits_gpu(self.model_name, self.cuda_device)

                                if not fits:
                                    logger.error(f"{msg}")
                                    if fallback_allowed:
                                        logger.warning("Falling back to CPU mode due to GPU memory constraints")
                                        self.use_cuda = False
                                        fallback_allowed = False
                                        continue
                                    logger.error(
                                        "Model will not be loaded to avoid a CUDA/OOM abort. Choose a smaller model."
                                    )
                                    return False

                                using_gpu = True
                                device_name = f"GPU (device {self.cuda_device})"
                                logger.info(f"✓ CUDA detected - {msg}")
                            elif cuda_status.available and not getattr(cuda_status, "pywhispercpp_cuda", False):
                                logger.info("CUDA detected but pywhispercpp is not CUDA-enabled - using CPU mode")
                            else:
                                logger.info("CUDA not available - using CPU mode")
                        except ImportError:
                            logger.debug("CUDA utils not available - using CPU mode")
                        except Exception as e:
                            logger.warning(f"CUDA check failed: {e} - using CPU mode")

                    load_start = time.time()

                    # Load model with parameters
                    params = {
                        'language': self.language,
                        'n_threads': self.n_threads,
                        'translate': self.translate,
                        'print_progress': False,
                        'print_realtime': False,
                    }

                    # NOTE: whisper.cpp GPU acceleration (CUDA) is enabled at build-time.
                    # pywhispercpp does not expose llama.cpp-style GPU offload params like
                    # `n_gpu_layers`; passing unknown params can raise exceptions (or worse).
                    if using_gpu:
                        logger.info("GPU mode enabled (CUDA build); no extra load params required")

                    # Remove None values
                    params = {k: v for k, v in params.items() if v is not None}

                    logger.info(f"Loading model on {device_name}...")
                    self._model = Model(str(model_file), **params)

                    load_time = time.time() - load_start

                    self._model_loaded = True
                    self._using_gpu = using_gpu
                    self._device_name = device_name
                    logger.info(f"✓ Model loaded successfully on {device_name} in {load_time:.2f}s")

                    if self.on_model_loaded:
                        try:
                            self.on_model_loaded()
                        except Exception as e:
                            logger.error(f"Model loaded callback error: {e}")

                    return True

                except Exception as e:
                    logger.error(f"Failed to load model: {e}")

                    if self.use_cuda and fallback_allowed:
                        logger.info("Attempting fallback to CPU mode...")
                        self.use_cuda = False
                        fallback_allowed = False
                        continue

                    if self.on_error:
                        try:
                            self.on_error(e)
                        except Exception:
                            pass
                    return False

        finally:
            try:
                self._transcription_lock.release()
            except Exception:
                pass
    
    def unload_model(self):
        """Unload model to free memory."""
        if not self._model:
            return

        # Prevent unload while a transcription is running.
        acquired = self._transcription_lock.acquire(timeout=10.0)
        if not acquired:
            logger.warning("Timed out waiting to unload model (transcription busy)")
            return

        try:
            logger.info("Unloading model...")
            # pywhispercpp doesn't have explicit unload, just dereference
            self._model = None
            self._model_loaded = False
            logger.info("Model unloaded")
        finally:
            self._transcription_lock.release()
    
    def transcribe(self, audio_data: np.ndarray, 
                   is_partial: bool = False) -> Optional[TranscriptionResult]:
        """
        Transcribe audio data.
        
        Args:
            audio_data: Numpy array of audio samples (16kHz, int16)
            is_partial: Whether this is a partial result
            
        Returns:
            TranscriptionResult or None on error
        """
        if not self._model_loaded or not self._model:
            logger.error("Cannot transcribe - model not loaded")
            return None
        
        if not PYWHISPERCPP_AVAILABLE:
            logger.error("Cannot transcribe - pywhispercpp not available")
            return None
        
        # Acquire lock to prevent concurrent transcription.
        # Do not drop work here; pipeline guarantees sequential calls.
        if not self._transcription_lock.acquire(timeout=30.0):
            logger.error("Timed out waiting for transcription lock")
            return None
        
        try:
            self._is_transcribing = True
            start_time = time.time()
            
            # Calculate audio duration
            duration = len(audio_data) / SAMPLE_RATE
            
            logger.info(f"Transcribing {duration:.2f}s of audio...")
            
            # Ensure correct dtype
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Transcribe
            segments = self._model.transcribe(audio_data)
            
            # Process results
            text_parts = []
            for segment in segments:
                if hasattr(segment, 'text'):
                    text_parts.append(segment.text)
                elif isinstance(segment, dict):
                    text_parts.append(segment.get('text', ''))
            
            text = ' '.join(text_parts).strip()
            
            # Detect language from first segment if auto-detect
            detected_language = self.language
            if not detected_language and segments:
                first_seg = segments[0]
                if hasattr(first_seg, 'language'):
                    detected_language = first_seg.language
                elif isinstance(first_seg, dict):
                    detected_language = first_seg.get('language')
            
            processing_time = time.time() - start_time
            
            # Calculate RTF (Real-Time Factor)
            rtf = processing_time / duration if duration > 0 else 0
            
            logger.info(f"Transcription complete: {len(text)} chars, RTF={rtf:.2f}x")
            
            # Update stats
            self._total_transcriptions += 1
            self._total_audio_seconds += duration
            self._total_processing_seconds += processing_time
            
            result = TranscriptionResult(
                text=text,
                language=detected_language,
                confidence=0.0,  # Not provided by whisper.cpp
                duration=duration,
                processing_time=processing_time,
                segments=[{'text': text, 'language': detected_language}] if text else [],
                is_partial=is_partial
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            
            if self.on_error:
                try:
                    self.on_error(e)
                except:
                    pass
            
            return None
        
        finally:
            # Always release the lock and reset flag
            self._is_transcribing = False
            self._transcription_lock.release()
    
    def transcribe_async(self, audio_data: np.ndarray):
        """
        Queue audio for asynchronous transcription.
        
        Args:
            audio_data: Numpy array of audio samples
        """
        try:
            self._transcription_queue.put_nowait(audio_data)
            logger.debug(f"Audio queued for transcription")
        except queue.Full:
            logger.warning("Transcription queue full, dropping audio")
    
    def start_worker(self) -> bool:
        """Start background transcription worker."""
        if self._worker_thread and self._worker_thread.is_alive():
            logger.info("Worker already running")
            return True
        
        if not self._model_loaded:
            logger.error("Cannot start worker - model not loaded")
            return False
        
        logger.info("Starting transcription worker...")
        self._stop_event.clear()
        
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="TranscriptionWorker",
            daemon=True
        )
        self._worker_thread.start()
        
        logger.info("Transcription worker started")
        return True
    
    def stop_worker(self):
        """Stop background transcription worker."""
        logger.info("Stopping transcription worker...")
        self._stop_event.set()
        
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
            if self._worker_thread.is_alive():
                logger.warning("Worker thread did not stop in time")
        
        logger.info("Transcription worker stopped")
    
    def _worker_loop(self):
        """Background worker loop for transcription."""
        logger.info("Worker loop started")
        
        while not self._stop_event.is_set():
            try:
                # Wait for audio with timeout
                audio_data = self._transcription_queue.get(timeout=0.5)
                
                # Transcribe
                result = self.transcribe(audio_data)
                
                if result and self.on_transcription:
                    try:
                        self.on_transcription(result)
                    except Exception as e:
                        logger.error(f"Transcription callback error: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
        
        logger.info("Worker loop ended")
    
    def is_ready(self) -> bool:
        """Check if transcriber is ready (model loaded)."""
        return self._model_loaded and self._model is not None

    def is_using_gpu(self) -> bool:
        """Return True if the currently loaded model is using GPU."""
        return bool(getattr(self, "_using_gpu", False))

    def get_device_name(self) -> str:
        """Return a human-readable device name (CPU/GPU)."""
        return str(getattr(self, "_device_name", "CPU"))
    
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
            'using_gpu': self.is_using_gpu(),
            'device': self.get_device_name(),
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_worker()
        self.unload_model()


class MultiModelTranscriber:
    """
    Manages multiple transcriber models for hot-swapping.
    """
    
    def __init__(self, config=None):
        """
        Initialize multi-model transcriber.
        
        Args:
            config: AppConfig instance
        """
        self.config = config
        self.transcribers: Dict[str, Transcriber] = {}
        self.current_transcriber: Optional[Transcriber] = None
        
        logger.info("MultiModelTranscriber initialized")
    
    def add_model(self, name: str, model_path: str, 
                  language: Optional[str] = None) -> bool:
        """
        Add a model to the pool.
        
        Args:
            name: Model identifier
            model_path: Path to GGML model file
            language: Language code or None
            
        Returns:
            True if added successfully
        """
        try:
            transcoder = Transcriber(
                model_path=model_path,
                model_name=name,
                language=language,
                n_threads=self.config.model.n_threads if self.config else 4,
                use_cuda=self.config.model.use_cuda if self.config else True,
                cuda_device=self.config.model.cuda_device if self.config else 0,
            )
            
            self.transcribers[name] = transcoder
            logger.info(f"Added model '{name}' to pool")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add model '{name}': {e}")
            return False
    
    def load_model(self, name: str) -> bool:
        """
        Load a specific model.
        
        Args:
            name: Model identifier
            
        Returns:
            True if loaded successfully
        """
        if name not in self.transcribers:
            logger.error(f"Model '{name}' not in pool")
            return False
        
        # Unload current
        if self.current_transcriber:
            self.current_transcriber.unload_model()
        
        # Load new
        transcoder = self.transcribers[name]
        if transcoder.load_model():
            self.current_transcriber = transcoder
            logger.info(f"Switched to model '{name}'")
            return True
        else:
            logger.error(f"Failed to load model '{name}'")
            return False
    
    def transcribe(self, audio_data: np.ndarray) -> Optional[TranscriptionResult]:
        """Transcribe using current model."""
        if not self.current_transcriber:
            logger.error("No model loaded")
            return None
        
        return self.current_transcriber.transcribe(audio_data)
    
    def get_current_model(self) -> Optional[str]:
        """Get name of currently loaded model."""
        if self.current_transcriber:
            return self.current_transcriber.model_name
        return None
    
    def list_models(self) -> List[str]:
        """List available models."""
        return list(self.transcribers.keys())


# Factory function
def create_transcriber(config=None, model_path: Optional[str] = None) -> Transcriber:
    """Create Transcriber instance from config or defaults."""
    if config:
        return Transcriber(
            model_path=model_path,
            model_name=config.model.default_model,
            language=config.model.language,
            n_threads=config.model.n_threads,
            translate=config.model.translate,
            use_cuda=getattr(config.model, "use_cuda", True),
            cuda_device=getattr(config.model, "cuda_device", 0),
        )
    else:
        return Transcriber(model_path=model_path)
