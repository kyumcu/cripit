"""
CripIt Configuration Module
Handles all application settings, paths, and constants.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json

# Configure logging
logger = logging.getLogger(__name__)

# Base paths
BASE_DIR = Path(__file__).parent.parent.absolute()
MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class AudioSettings:
    """Audio capture settings."""
    sample_rate: int = 16000  # Whisper requires 16kHz
    channels: int = 1  # Mono
    chunk_size: int = 1024  # Frames per buffer
    format: str = "int16"  # 16-bit PCM
    max_recording_duration: float = 30.0  # Max seconds per chunk
    
    # VAD settings
    vad_enabled: bool = True
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive
    vad_frame_duration: int = 30  # ms (10, 20, or 30)
    silence_timeout: float = 0.5  # Seconds of silence to end recording
    
    # Device and gain settings
    device_index: Optional[int] = None  # None = default device
    gain_db: float = 0.0  # Gain in decibels (0 = no change, positive = louder)
    
    def __post_init__(self):
        logger.info(f"AudioSettings initialized: {self}")
    
    def __repr__(self):
        return f"AudioSettings(sample_rate={self.sample_rate}, channels={self.channels}, device={self.device_index}, gain={self.gain_db}dB)"


@dataclass
class ModelSettings:
    """Model configuration settings."""
    default_model: str = "large-v3-turbo"
    available_models: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "tiny": {"params": "39M", "file": "ggml-tiny.bin", "size_mb": 75},
        "base": {"params": "74M", "file": "ggml-base.bin", "size_mb": 142},
        "small": {"params": "244M", "file": "ggml-small.bin", "size_mb": 466},
        "medium": {"params": "769M", "file": "ggml-medium.bin", "size_mb": 1500},
        "large-v3": {"params": "1.5B", "file": "ggml-large-v3.bin", "size_mb": 2900},
        "large-v3-turbo": {"params": "809M", "file": "ggml-large-v3-turbo.bin", "size_mb": 1500},
        "distil-large-v3": {"params": "756M", "file": "ggml-distil-large-v3.bin", "size_mb": 1300},
    })
    
    # whisper.cpp parameters
    language: Optional[str] = None  # None = auto-detect
    translate: bool = False
    n_threads: int = 4
    max_context: int = -1
    
    # CUDA/GPU settings
    use_cuda: bool = True  # Try to use CUDA if available
    cuda_device: int = 0  # GPU device ID (for multi-GPU systems)
    cuda_fallback_to_cpu: bool = True  # Automatically fall back to CPU if CUDA fails
    cuda_warn_on_fallback: bool = True  # Show warning when falling back to CPU
    
    # WhisperX settings
    asr_engine: str = "whispercpp"  # Options: "whispercpp", "whisperx"
    whisperx_model: str = "large-v3"  # Options: tiny, base, small, medium, large-v3
    whisperx_compute_type: str = "int8"  # Options: int8, float16, float32
    whisperx_diarize: bool = False  # Disabled by default (opt-in)
    whisperx_hf_token: Optional[str] = None  # Required only if diarization enabled
    whisperx_device: str = "cuda"  # Options: "cuda", "cpu"
    whisperx_cpu_threads: int = 0  # 0 = let backend decide
    whisperx_num_workers: int = 1
    
    def __post_init__(self):
        logger.info(f"ModelSettings initialized with default model: {self.default_model}")
        logger.info(f"CUDA settings: use_cuda={self.use_cuda}, device={self.cuda_device}, fallback={self.cuda_fallback_to_cpu}")
    
    def __repr__(self):
        return f"ModelSettings(default_model={self.default_model}, language={self.language}, use_cuda={self.use_cuda})"


@dataclass
class UISettings:
    """UI configuration settings."""
    window_title: str = "CripIt - Real-time Speech-to-Text"
    window_width: int = 800
    window_height: int = 600
    font_family: str = "Segoe UI" if os.name == "nt" else "Helvetica"
    font_size: int = 14
    
    # Colors
    bg_color: str = "#f5f5f5"
    text_color: str = "#333333"
    accent_color: str = "#2196F3"
    
    # Behavior
    auto_copy: bool = False
    minimize_to_tray: bool = True
    global_hotkey: str = "ctrl+shift+r"
    
    # Output directory for transcriptions
    output_dir: str = str(BASE_DIR / "output")
    
    def __post_init__(self):
        logger.info(f"UISettings initialized: {self.window_width}x{self.window_height}")
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def __repr__(self):
        return f"UISettings(title={self.window_title}, output={self.output_dir})"


@dataclass
class SpoolSettings:
    """Disk-backed recording spool settings."""

    # Root directory for the recording spool (queued/processing/failed)
    dir: str = str(BASE_DIR / "output" / "spool")

    # If free space falls below this after spooling, we stop recording soon.
    soft_min_free_mb: int = 1024

    # If free space is below (job_size + reserve), we refuse to spool and stop.
    hard_reserve_mb: int = 256

    def __post_init__(self):
        Path(self.dir).mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return (
            f"SpoolSettings(dir={self.dir}, soft_min_free_mb={self.soft_min_free_mb}, "
            f"hard_reserve_mb={self.hard_reserve_mb})"
        )


class AppConfig:
    """Main application configuration singleton."""
    
    _instance: Optional['AppConfig'] = None
    _config_file: Path = BASE_DIR / "app_config.json"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        logger.info("=" * 50)
        logger.info("Initializing AppConfig")
        logger.info("=" * 50)
        
        self.audio = AudioSettings()
        self.model = ModelSettings()
        self.ui = UISettings()
        self.spool = SpoolSettings()
        
        # Load saved config if exists
        self._load_config()
        
        self._initialized = True
        logger.info("AppConfig initialization complete")
    
    def _load_config(self):
        """Load configuration from file."""
        if self._config_file.exists():
            try:
                logger.info(f"Loading config from {self._config_file}")
                with open(self._config_file, 'r') as f:
                    data = json.load(f)
                
                # Update settings from file
                if 'audio' in data:
                    for key, value in data['audio'].items():
                        if hasattr(self.audio, key):
                            setattr(self.audio, key, value)
                            logger.debug(f"Loaded audio.{key} = {value}")
                
                if 'model' in data:
                    for key, value in data['model'].items():
                        if hasattr(self.model, key) and key != 'available_models':
                            setattr(self.model, key, value)
                            logger.debug(f"Loaded model.{key} = {value}")
                    # Log CUDA settings after loading
                    logger.info(f"Loaded CUDA config: use_cuda={self.model.use_cuda}, device={self.model.cuda_device}")
                
                if 'ui' in data:
                    for key, value in data['ui'].items():
                        if hasattr(self.ui, key):
                            setattr(self.ui, key, value)
                            logger.debug(f"Loaded ui.{key} = {value}")

                if 'spool' in data:
                    for key, value in data['spool'].items():
                        if hasattr(self.spool, key):
                            setattr(self.spool, key, value)
                            logger.debug(f"Loaded spool.{key} = {value}")
                
                logger.info("Configuration loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        else:
            logger.info("No existing config file found, using defaults")
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            logger.info(f"Saving config to {self._config_file}")
            data = {
                'audio': {
                    'sample_rate': self.audio.sample_rate,
                    'channels': self.audio.channels,
                    'vad_enabled': self.audio.vad_enabled,
                    'vad_aggressiveness': self.audio.vad_aggressiveness,
                    'silence_timeout': self.audio.silence_timeout,
                    'device_index': self.audio.device_index,
                    'gain_db': self.audio.gain_db,
                },
                'model': {
                    'default_model': self.model.default_model,
                    'language': self.model.language,
                    'translate': self.model.translate,
                    'n_threads': self.model.n_threads,
                    'use_cuda': self.model.use_cuda,
                    'cuda_device': self.model.cuda_device,
                    'cuda_fallback_to_cpu': self.model.cuda_fallback_to_cpu,
                    'cuda_warn_on_fallback': self.model.cuda_warn_on_fallback,
                    'asr_engine': self.model.asr_engine,
                    'whisperx_model': self.model.whisperx_model,
                    'whisperx_compute_type': self.model.whisperx_compute_type,
                    'whisperx_diarize': self.model.whisperx_diarize,
                    'whisperx_hf_token': self.model.whisperx_hf_token,
                    'whisperx_device': self.model.whisperx_device,
                    'whisperx_cpu_threads': self.model.whisperx_cpu_threads,
                    'whisperx_num_workers': self.model.whisperx_num_workers,
                },
                'ui': {
                    'auto_copy': self.ui.auto_copy,
                    'minimize_to_tray': self.ui.minimize_to_tray,
                    'global_hotkey': self.ui.global_hotkey,
                    'output_dir': self.ui.output_dir,
                },
                'spool': {
                    'dir': self.spool.dir,
                    'soft_min_free_mb': self.spool.soft_min_free_mb,
                    'hard_reserve_mb': self.spool.hard_reserve_mb,
                }
            }
            
            with open(self._config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def get_model_path(self, model_name: Optional[str] = None) -> Path:
        """Get path to model file."""
        model = model_name or self.model.default_model
        
        if model in self.model.available_models:
            filename = self.model.available_models[model]['file']
            path = MODELS_DIR / filename
            logger.debug(f"Model path for {model}: {path}")
            return path
        else:
            logger.warning(f"Unknown model: {model}, using default")
            return self.get_model_path(self.model.default_model)
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if model file exists locally."""
        path = self.get_model_path(model_name)
        exists = path.exists()
        logger.debug(f"Model {model_name} available: {exists}")
        return exists


# Global config instance
config = AppConfig()
