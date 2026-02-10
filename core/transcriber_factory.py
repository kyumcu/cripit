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
                hf_token=config.model.whisperx_hf_token,
                cpu_threads=getattr(config.model, "whisperx_cpu_threads", 0),
                num_workers=getattr(config.model, "whisperx_num_workers", 1),
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
