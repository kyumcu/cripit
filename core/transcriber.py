"""
Transcriber Module (Backwards Compatibility)
This module re-exports from whispercpp_transcriber for backwards compatibility.
New code should import from core.whispercpp_transcriber or use the factory.
"""

# Re-export everything from whispercpp_transcriber for backwards compatibility
from core.whispercpp_transcriber import (
    WhisperCppTranscriber,
    MultiModelTranscriber,
    create_transcriber,
    create_whispercpp_transcriber,
    PYWHISPERCPP_AVAILABLE,
)

# Import TranscriptionResult from base_transcriber
from core.base_transcriber import TranscriptionResult

# SAMPLE_RATE is conditionally defined in whispercpp_transcriber
# Define it here for compatibility
SAMPLE_RATE = 16000

# Keep Transcriber as an alias for backwards compatibility
Transcriber = WhisperCppTranscriber

__all__ = [
    'WhisperCppTranscriber',
    'Transcriber',  # Alias for backwards compatibility
    'MultiModelTranscriber',
    'TranscriptionResult',
    'create_transcriber',
    'create_whispercpp_transcriber',
    'PYWHISPERCPP_AVAILABLE',
    'SAMPLE_RATE',
]
