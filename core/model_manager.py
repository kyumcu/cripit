"""
Model Manager Module
Handles downloading and managing GGML models from Hugging Face
"""

import logging
import os
import requests
from pathlib import Path
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
import threading
import hashlib
import time
import traceback

logger = logging.getLogger(__name__)

# Import crash handler if available
try:
    from utils.crash_handler import get_crash_handler
    crash_handler = get_crash_handler()
except ImportError:
    crash_handler = None


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    file: str
    size_mb: int
    params: str
    url: str
    description: str
    sha256: Optional[str] = None


class ModelManager:
    """
    Manages downloading and caching of GGML models.
    
    Models are stored in the models/ directory and downloaded from
    the official whisper.cpp Hugging Face repository.
    """
    
    # Hugging Face repository for whisper.cpp models
    HF_REPO = "https://huggingface.co/ggerganov/whisper.cpp"
    HF_API = "https://huggingface.co/api/models/ggerganov/whisper.cpp"
    
    # Model definitions
    MODELS: Dict[str, ModelInfo] = {
        "tiny": ModelInfo(
            name="tiny",
            file="ggml-tiny.bin",
            size_mb=75,
            params="39M",
            url=f"{HF_REPO}/resolve/main/ggml-tiny.bin",
            description="Smallest model, fastest but lowest quality"
        ),
        "tiny.en": ModelInfo(
            name="tiny.en",
            file="ggml-tiny.en.bin",
            size_mb=75,
            params="39M",
            url=f"{HF_REPO}/resolve/main/ggml-tiny.en.bin",
            description="Tiny model optimized for English"
        ),
        "base": ModelInfo(
            name="base",
            file="ggml-base.bin",
            size_mb=142,
            params="74M",
            url=f"{HF_REPO}/resolve/main/ggml-base.bin",
            description="Base model, good for testing"
        ),
        "base.en": ModelInfo(
            name="base.en",
            file="ggml-base.en.bin",
            size_mb=142,
            params="74M",
            url=f"{HF_REPO}/resolve/main/ggml-base.en.bin",
            description="Base model optimized for English"
        ),
        "small": ModelInfo(
            name="small",
            file="ggml-small.bin",
            size_mb=466,
            params="244M",
            url=f"{HF_REPO}/resolve/main/ggml-small.bin",
            description="Small model, good balance for resource-limited devices"
        ),
        "small.en": ModelInfo(
            name="small.en",
            file="ggml-small.en.bin",
            size_mb=466,
            params="244M",
            url=f"{HF_REPO}/resolve/main/ggml-small.en.bin",
            description="Small model optimized for English"
        ),
        "medium": ModelInfo(
            name="medium",
            file="ggml-medium.bin",
            size_mb=1500,
            params="769M",
            url=f"{HF_REPO}/resolve/main/ggml-medium.bin",
            description="Medium model, good quality"
        ),
        "medium.en": ModelInfo(
            name="medium.en",
            file="ggml-medium.en.bin",
            size_mb=1500,
            params="769M",
            url=f"{HF_REPO}/resolve/main/ggml-medium.en.bin",
            description="Medium model optimized for English"
        ),
        "large-v1": ModelInfo(
            name="large-v1",
            file="ggml-large-v1.bin",
            size_mb=2900,
            params="1.55B",
            url=f"{HF_REPO}/resolve/main/ggml-large-v1.bin",
            description="Original large model"
        ),
        "large-v2": ModelInfo(
            name="large-v2",
            file="ggml-large-v2.bin",
            size_mb=2900,
            params="1.55B",
            url=f"{HF_REPO}/resolve/main/ggml-large-v2.bin",
            description="Large model v2, improved quality"
        ),
        "large-v3": ModelInfo(
            name="large-v3",
            file="ggml-large-v3.bin",
            size_mb=2900,
            params="1.55B",
            url=f"{HF_REPO}/resolve/main/ggml-large-v3.bin",
            description="Large model v3, best quality"
        ),
        "large-v3-turbo": ModelInfo(
            name="large-v3-turbo",
            file="ggml-large-v3-turbo.bin",
            size_mb=1500,
            params="809M",
            url=f"{HF_REPO}/resolve/main/ggml-large-v3-turbo.bin",
            description="Large v3 Turbo - 6x faster, RECOMMENDED"
        ),
        "distil-large-v3": ModelInfo(
            name="distil-large-v3",
            file="ggml-distil-large-v3.bin",
            size_mb=1300,
            params="756M",
            url="https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin",
            description="Distilled large v3, English only, very fast"
        ),
    }
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize model manager.
        
        Args:
            models_dir: Directory to store models (default: models/)
        """
        if models_dir is None:
            from config.settings import MODELS_DIR
            self.models_dir = MODELS_DIR
        else:
            self.models_dir = Path(models_dir)
        
        self.models_dir.mkdir(exist_ok=True)
        
        # Callbacks for download progress
        self.on_progress: Optional[Callable[[str, int, int], None]] = None
        self.on_complete: Optional[Callable[[str, bool], None]] = None
        
        # Track current downloads
        self._current_downloads: Dict[str, threading.Thread] = {}
        
        logger.info("=" * 50)
        logger.info("Initializing ModelManager")
        logger.info("=" * 50)
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Available models: {len(self.MODELS)}")
    
    def list_models(self) -> List[str]:
        """List all available model names."""
        return list(self.MODELS.keys())
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a model."""
        return self.MODELS.get(model_name)
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if model is downloaded locally."""
        info = self.get_model_info(model_name)
        if not info:
            return False
        
        path = self.models_dir / info.file
        exists = path.exists()
        
        if exists:
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.debug(f"Model {model_name}: {size_mb:.1f} MB")
        
        return exists
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get path to model file if available."""
        if not self.is_model_available(model_name):
            return None
        
        info = self.get_model_info(model_name)
        return self.models_dir / info.file
    
    def get_available_models(self) -> List[str]:
        """List models that are downloaded locally."""
        return [name for name in self.MODELS.keys() if self.is_model_available(name)]
    
    def get_missing_models(self) -> List[str]:
        """List models that are not downloaded."""
        return [name for name in self.MODELS.keys() if not self.is_model_available(name)]
    
    def download_model(self, model_name: str, 
                       blocking: bool = False) -> bool:
        """
        Download a model.
        
        Args:
            model_name: Name of model to download
            blocking: If True, block until download complete
            
        Returns:
            True if download started or completed successfully
        """
        info = self.get_model_info(model_name)
        if not info:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        # Check if already downloading
        if model_name in self._current_downloads:
            logger.warning(f"Model {model_name} already downloading")
            return True
        
        # Check if already available
        if self.is_model_available(model_name):
            logger.info(f"Model {model_name} already available")
            if self.on_complete:
                self.on_complete(model_name, True)
            return True
        
        logger.info(f"Starting download: {model_name} ({info.size_mb} MB)")
        
        if blocking:
            return self._download_blocking(info)
        else:
            # Start download in background thread
            thread = threading.Thread(
                target=self._download_blocking,
                args=(info,),
                name=f"Download-{model_name}"
            )
            self._current_downloads[model_name] = thread
            thread.start()
            return True
    
    def _download_blocking(self, info: ModelInfo) -> bool:
        """
        Download model (blocking).
        
        Args:
            info: ModelInfo to download
            
        Returns:
            True if download successful
        """
        output_path = self.models_dir / info.file
        download_start_time = time.time()
        
        try:
            logger.info(f"="*60)
            logger.info(f"DOWNLOAD START: {info.name}")
            logger.info(f"URL: {info.url}")
            logger.info(f"Expected size: {info.size_mb} MB")
            logger.info(f"Output path: {output_path}")
            logger.info(f"="*60)
            
            # Log network activity to crash handler
            if crash_handler:
                crash_handler.log_network_activity(info.url, "GET", "STARTING")
            
            # Stream download with progress
            logger.info("Sending HTTP request...")
            response = requests.get(info.url, stream=True, timeout=30)
            
            logger.info(f"Response received: HTTP {response.status_code}")
            logger.info(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
            logger.info(f"Content-Length: {response.headers.get('content-length', 'unknown')} bytes")
            
            response.raise_for_status()
            
            if crash_handler:
                crash_handler.log_network_activity(info.url, "GET", f"HTTP_{response.status_code}")
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192  # 8KB chunks
            last_progress_time = time.time()
            
            logger.info("Starting download stream...")
            
            with open(output_path, 'wb') as f:
                chunk_count = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        chunk_count += 1
                        
                        # Log progress every 5 seconds
                        current_time = time.time()
                        if current_time - last_progress_time > 5.0:
                            mb_downloaded = downloaded / (1024 * 1024)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                logger.info(f"Progress: {mb_downloaded:.1f} MB / {total_size/(1024*1024):.1f} MB ({percent:.1f}%)")
                            else:
                                logger.info(f"Progress: {mb_downloaded:.1f} MB downloaded")
                            last_progress_time = current_time
                        
                        # Report progress to callback
                        if total_size > 0 and self.on_progress:
                            percent = int(100 * downloaded / total_size)
                            self.on_progress(info.name, percent, total_size)
            
            download_time = time.time() - download_start_time
            logger.info(f"Download stream complete: {chunk_count} chunks in {download_time:.1f}s")
            
            # Verify file
            file_size = output_path.stat().st_size
            expected_size = info.size_mb * 1024 * 1024
            
            logger.info(f"Verifying download...")
            logger.info(f"Downloaded size: {file_size} bytes ({file_size/(1024*1024):.1f} MB)")
            logger.info(f"Expected size: {expected_size} bytes ({info.size_mb} MB)")
            
            if file_size < expected_size * 0.9:  # Allow 10% tolerance
                logger.error(f"FAILED: Downloaded file too small!")
                logger.error(f"Expected at least {expected_size * 0.9} bytes, got {file_size} bytes")
                output_path.unlink(missing_ok=True)
                if self.on_complete:
                    self.on_complete(info.name, False)
                return False
            
            logger.info(f"="*60)
            logger.info(f"DOWNLOAD SUCCESS: {info.name}")
            logger.info(f"Final size: {file_size/(1024*1024):.1f} MB")
            logger.info(f"Time: {download_time:.1f}s")
            logger.info(f"Speed: {(file_size/1024/1024)/download_time:.1f} MB/s")
            logger.info(f"="*60)
            
            if self.on_complete:
                self.on_complete(info.name, True)
            
            return True
            
        except requests.exceptions.Timeout as e:
            logger.error(f"DOWNLOAD TIMEOUT: {info.name}")
            logger.error(f"URL: {info.url}")
            logger.error(f"Error: {e}")
            if crash_handler:
                crash_handler.log_network_activity(info.url, "GET", "TIMEOUT")
            output_path.unlink(missing_ok=True)
            if self.on_complete:
                self.on_complete(info.name, False)
            return False
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"DOWNLOAD CONNECTION ERROR: {info.name}")
            logger.error(f"URL: {info.url}")
            logger.error(f"Error: {e}")
            if crash_handler:
                crash_handler.log_network_activity(info.url, "GET", "CONNECTION_ERROR")
            output_path.unlink(missing_ok=True)
            if self.on_complete:
                self.on_complete(info.name, False)
            return False
            
        except requests.exceptions.RequestException as e:
            logger.error(f"DOWNLOAD REQUEST FAILED: {info.name}")
            logger.error(f"URL: {info.url}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error: {e}")
            if crash_handler:
                crash_handler.log_network_activity(info.url, "GET", f"ERROR_{type(e).__name__}")
            output_path.unlink(missing_ok=True)
            if self.on_complete:
                self.on_complete(info.name, False)
            return False
            
        except Exception as e:
            logger.error(f"DOWNLOAD UNEXPECTED ERROR: {info.name}")
            logger.error(f"URL: {info.url}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error: {e}")
            logger.error("Traceback:")
            logger.error(traceback.format_exc())
            if crash_handler:
                crash_handler.log_network_activity(info.url, "GET", f"EXCEPTION_{type(e).__name__}")
            output_path.unlink(missing_ok=True)
            if self.on_complete:
                self.on_complete(info.name, False)
            return False
            
        finally:
            # Remove from current downloads
            if info.name in self._current_downloads:
                del self._current_downloads[info.name]
                logger.debug(f"Removed {info.name} from current downloads")
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a downloaded model.
        
        Args:
            model_name: Name of model to delete
            
        Returns:
            True if deleted successfully
        """
        info = self.get_model_info(model_name)
        if not info:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        path = self.models_dir / info.file
        
        if not path.exists():
            logger.warning(f"Model {model_name} not found locally")
            return True
        
        try:
            path.unlink()
            logger.info(f"Deleted model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {model_name}: {e}")
            return False
    
    def get_model_size(self, model_name: str) -> Optional[int]:
        """
        Get size of downloaded model in bytes.
        
        Args:
            model_name: Name of model
            
        Returns:
            Size in bytes, or None if not available
        """
        if not self.is_model_available(model_name):
            return None
        
        info = self.get_model_info(model_name)
        path = self.models_dir / info.file
        return path.stat().st_size
    
    def get_total_size(self) -> int:
        """Get total size of all downloaded models in bytes."""
        total = 0
        for name in self.get_available_models():
            size = self.get_model_size(name)
            if size:
                total += size
        return total
    
    def is_downloading(self, model_name: str) -> bool:
        """Check if a model is currently being downloaded."""
        return model_name in self._current_downloads
    
    def cancel_download(self, model_name: str) -> bool:
        """
        Cancel an in-progress download.
        
        Note: This won't immediately stop the download thread,
        but will prevent completion callback from firing.
        """
        if model_name in self._current_downloads:
            del self._current_downloads[model_name]
            logger.info(f"Cancelled download: {model_name}")
            return True
        return False


# Singleton instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get singleton ModelManager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
