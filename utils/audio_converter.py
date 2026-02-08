"""Audio conversion utilities.

Provides conversion from various audio formats (m4a, mp3, ogg, flac, etc.)
to WAV format suitable for Whisper transcription.

Requires FFmpeg to be installed on the system.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Supported input formats
SUPPORTED_FORMATS = [".m4a", ".mp3", ".ogg", ".flac", ".aac", ".wma", ".wav"]
TARGET_SAMPLE_RATE = 16000  # Whisper requirement


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def get_supported_formats() -> List[str]:
    """Return list of supported audio file extensions."""
    return SUPPORTED_FORMATS.copy()


def convert_to_wav(
    input_path: Path,
    output_dir: Optional[Path] = None,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> Tuple[bool, Optional[Path], str]:
    """Convert an audio file to WAV format.

    Args:
        input_path: Path to input audio file
        output_dir: Directory for output (default: same as input)
        target_sample_rate: Target sample rate (default: 16000 for Whisper)

    Returns:
        Tuple of (success: bool, output_path: Optional[Path], error_message: str)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        return False, None, f"Input file not found: {input_path}"

    # Check if already WAV
    if input_path.suffix.lower() == ".wav":
        # Still need to ensure correct format (16kHz, mono, 16-bit)
        return _ensure_wav_format(input_path, output_dir, target_sample_rate)

    # Check FFmpeg availability
    if not check_ffmpeg_available():
        return (
            False,
            None,
            "FFmpeg not found. Please install FFmpeg:\n"
            "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "  macOS: brew install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/download.html",
        )

    try:
        from pydub import AudioSegment
    except ImportError:
        return (
            False,
            None,
            "pydub not installed. Run: pip install pydub>=0.25.1",
        )

    try:
        # Determine output path
        if output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Create output filename with timestamp to avoid collisions
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{input_path.stem}_{timestamp}.wav"
        output_path = output_dir / output_name

        # Load and convert audio
        logger.info(f"Converting {input_path} to WAV...")
        audio = AudioSegment.from_file(str(input_path))

        # Convert to target format
        audio = (
            audio.set_channels(1)  # Mono
            .set_frame_rate(target_sample_rate)  # 16kHz
            .set_sample_width(2)  # 16-bit
        )

        # Export as WAV
        audio.export(str(output_path), format="wav")

        logger.info(f"Conversion complete: {output_path}")
        return True, output_path, ""

    except Exception as e:
        error_msg = f"Conversion failed: {type(e).__name__}: {e}"
        logger.error(error_msg)
        return False, None, error_msg


def _ensure_wav_format(
    input_path: Path,
    output_dir: Optional[Path] = None,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> Tuple[bool, Optional[Path], str]:
    """Ensure a WAV file is in the correct format for Whisper.

    If the file is already in the correct format, returns the original path.
    Otherwise, converts it.
    """
    try:
        import wave

        with wave.open(str(input_path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()

        # Check if already in correct format
        if (
            channels == 1
            and sample_rate == target_sample_rate
            and sample_width == 2
        ):
            logger.info(f"WAV file already in correct format: {input_path}")
            return True, input_path, ""

        # Needs conversion
        logger.info(
            f"Converting WAV from {channels}ch/{sample_rate}Hz/{sample_width * 8}bit "
            f"to 1ch/{target_sample_rate}Hz/16bit"
        )

    except Exception as e:
        return False, None, f"Failed to read WAV file: {e}"

    # Use pydub to convert
    try:
        from pydub import AudioSegment

        if output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{input_path.stem}_{timestamp}.wav"
        output_path = output_dir / output_name

        audio = AudioSegment.from_wav(str(input_path))
        audio = (
            audio.set_channels(1)
            .set_frame_rate(target_sample_rate)
            .set_sample_width(2)
        )
        audio.export(str(output_path), format="wav")

        logger.info(f"WAV conversion complete: {output_path}")
        return True, output_path, ""

    except Exception as e:
        error_msg = f"WAV conversion failed: {type(e).__name__}: {e}"
        logger.error(error_msg)
        return False, None, error_msg


def get_converted_files(converted_dir: Path) -> List[Tuple[Path, float, int]]:
    """Get list of converted WAV files with metadata.

    Returns:
        List of tuples: (file_path, modification_time, size_bytes)
    """
    converted_dir = Path(converted_dir)
    if not converted_dir.exists():
        return []

    files = []
    for wav_file in converted_dir.glob("*.wav"):
        stat = wav_file.stat()
        files.append((wav_file, stat.st_mtime, stat.st_size))

    # Sort by modification time (newest first)
    files.sort(key=lambda x: x[1], reverse=True)
    return files


def delete_converted_file(file_path: Path) -> bool:
    """Delete a converted audio file."""
    try:
        file_path = Path(file_path)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted converted file: {file_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to delete {file_path}: {e}")
        return False


def clear_all_converted_files(converted_dir: Path) -> Tuple[int, int]:
    """Delete all converted WAV files in directory.

    Returns:
        Tuple of (deleted_count, failed_count)
    """
    converted_dir = Path(converted_dir)
    if not converted_dir.exists():
        return 0, 0

    deleted = 0
    failed = 0

    for wav_file in converted_dir.glob("*.wav"):
        try:
            wav_file.unlink()
            deleted += 1
            logger.info(f"Deleted: {wav_file}")
        except Exception as e:
            logger.error(f"Failed to delete {wav_file}: {e}")
            failed += 1

    logger.info(f"Cleared converted files: {deleted} deleted, {failed} failed")
    return deleted, failed


def get_total_converted_size(converted_dir: Path) -> int:
    """Get total size of all converted files in bytes."""
    converted_dir = Path(converted_dir)
    if not converted_dir.exists():
        return 0

    total_size = 0
    for wav_file in converted_dir.glob("*.wav"):
        try:
            total_size += wav_file.stat().st_size
        except Exception:
            pass

    return total_size


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_duration(duration_seconds: float) -> str:
    """Format duration in human-readable format."""
    if duration_seconds < 60:
        return f"{int(duration_seconds)}s"
    elif duration_seconds < 3600:
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def read_wav_duration(wav_path: Path) -> float:
    """Read duration of a WAV file in seconds."""
    try:
        import wave

        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate
    except Exception as e:
        logger.warning(f"Failed to read duration of {wav_path}: {e}")
        return 0.0
