#!/usr/bin/env python3
"""
CripIt - Real-time Speech-to-Text Application
Main entry point

Usage:
    python main.py
    python main.py --model tiny --language en
"""

import sys
import os
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging before other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize crash handler early
try:
    from utils.crash_handler import get_crash_handler
    crash_handler = get_crash_handler()
    logger.info("Crash handler initialized - unexpected exits will be logged")
except Exception as e:
    logger.warning(f"Could not initialize crash handler: {e}")
    crash_handler = None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CripIt - Real-time Speech-to-Text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Start with default model (large-v3-turbo)
  %(prog)s --model tiny              # Use tiny model (fastest)
  %(prog)s --language en             # Force English language
  %(prog)s --list-models             # List available models
  %(prog)s --download tiny           # Download tiny model
        """
    )

    parser.add_argument(
        '--terminal', '--tui',
        action='store_true',
        help='Run the headless terminal UI (curses) instead of the PyQt GUI'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Model to use (default: from config)'
    )
    
    parser.add_argument(
        '--language', '-l',
        type=str,
        default=None,
        help='Language code (e.g., en, es, fr) or auto-detect if not specified'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models and exit'
    )
    
    parser.add_argument(
        '--download',
        type=str,
        metavar='MODEL',
        help='Download a model and exit'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    return parser.parse_args()


def _maybe_run_terminal(argv: list[str]) -> Optional[int]:
    """If --terminal/--tui is present, run terminal_app.py and return its exit code."""
    if '--terminal' not in argv and '--tui' not in argv:
        return None

    # Remove our helper flag(s) and exec the TUI entrypoint.
    pass_args = [a for a in argv if a not in ('--terminal', '--tui')]
    terminal_app = Path(__file__).parent / 'terminal_app.py'

    if not terminal_app.exists():
        print("Error: terminal_app.py not found")
        return 1

    cmd = [sys.executable, str(terminal_app)] + pass_args
    return subprocess.call(cmd)


def list_models():
    """List all available models."""
    from config.settings import config
    from core.model_manager import get_model_manager
    
    manager = get_model_manager()
    
    print("\n" + "="*60)
    print("AVAILABLE MODELS")
    print("="*60)
    
    available = manager.get_available_models()
    
    for name in manager.list_models():
        info = manager.get_model_info(name)
        is_avail = name in available
        status = "✓ Downloaded" if is_avail else "  Not downloaded"
        
        print(f"\n{name:20s} {status}")
        print(f"  Parameters: {info.params:>8s}")
        print(f"  Size:       {info.size_mb:>6d} MB")
        print(f"  File:       {info.file}")
        print(f"  {info.description}")
    
    print("\n" + "="*60)
    print(f"Total: {len(manager.list_models())} models, {len(available)} downloaded")
    print("="*60 + "\n")


def download_model(model_name: str):
    """Download a model."""
    from core.model_manager import get_model_manager
    
    manager = get_model_manager()
    
    if model_name not in manager.list_models():
        print(f"Error: Unknown model '{model_name}'")
        print(f"Run with --list-models to see available models")
        return False
    
    info = manager.get_model_info(model_name)
    
    print(f"\nDownloading model: {model_name}")
    print(f"Size: {info.size_mb} MB")
    print(f"URL: {info.url}")
    print("")
    
    # Setup progress callback
    def on_progress(name, percent, total):
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r[{bar}] {percent}%", end='', flush=True)
    
    def on_complete(name, success):
        print()  # New line after progress bar
        if success:
            print(f"✓ Successfully downloaded {name}")
        else:
            print(f"✗ Failed to download {name}")
    
    manager.on_progress = on_progress
    manager.on_complete = on_complete
    
    success = manager.download_model(model_name, blocking=True)
    return success


def check_dependencies():
    """Check if required dependencies are available."""
    logger.info("Checking dependencies...")
    
    missing = []
    
    # Check PyQt6
    try:
        import PyQt6
        logger.info("✓ PyQt6 available")
    except ImportError:
        missing.append("PyQt6")
        logger.error("✗ PyQt6 not available")
    
    # Check audio
    from core.audio_capture import SOUNDDEVICE_AVAILABLE, WEBRTC_AVAILABLE
    if not SOUNDDEVICE_AVAILABLE:
        missing.append("sounddevice (pip install sounddevice)")
        logger.error("✗ sounddevice not available")
    else:
        logger.info("✓ sounddevice available")
    
    if not WEBRTC_AVAILABLE:
        missing.append("webrtcvad (pip install webrtcvad-wheels)")
        logger.error("✗ WebRTC VAD not available")
    else:
        logger.info("✓ WebRTC VAD available")
    
    # Check transcriber engines
    from core.transcriber_factory import check_engine_availability
    engines = check_engine_availability()
    
    if not engines["whispercpp"] and not engines["whisperx"]:
        missing.append("pywhispercpp or whisperx (pip install pywhispercpp OR pip install whisperx torch torchaudio)")
        logger.error("✗ No ASR engine available")
    else:
        if engines["whispercpp"]:
            logger.info("✓ whisper.cpp available")
        if engines["whisperx"]:
            logger.info("✓ WhisperX available")
    
    if missing:
        print("\n" + "="*60)
        print("MISSING DEPENDENCIES")
        print("="*60)
        for dep in missing:
            print(f"  ✗ {dep}")
        print("="*60)
        print("\nInstall with: pip install " + " ".join(missing))
        return False
    
    logger.info("All dependencies available")
    return True


def main():
    """Main application entry point."""
    terminal_exit = _maybe_run_terminal(sys.argv[1:])
    if terminal_exit is not None:
        return int(terminal_exit)

    args = parse_arguments()
    
    # Set verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    # Handle command-line only operations
    if args.list_models:
        list_models()
        return 0
    
    if args.download:
        success = download_model(args.download)
        return 0 if success else 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Import GUI dependencies
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        logger.error("PyQt6 is required to run the GUI")
        return 1
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("CripIt")
    app.setApplicationVersion("1.0.0")
    
    logger.info("=" * 60)
    logger.info("Starting CripIt - Real-time Speech-to-Text")
    logger.info("=" * 60)
    
    # Initialize components
    try:
        from config.settings import config
        from core.audio_capture import AudioCapture
        from core.transcriber_factory import create_transcriber, check_engine_availability
        from core.model_manager import get_model_manager
        from gui.main_window import MainWindow
        
        # Update config from arguments
        if args.model:
            config.model.default_model = args.model
            logger.info(f"Using model from args: {args.model}")
        
        if args.language:
            config.model.language = args.language
            logger.info(f"Using language from args: {args.language}")
        
        # Check if preferred engine is available
        engines = check_engine_availability()
        preferred = config.model.asr_engine
        if preferred == "whisperx" and not engines["whisperx"]:
            logger.warning("WhisperX not available, falling back to whisper.cpp")
            config.model.asr_engine = "whispercpp"
            config.save_config()
        
        # Initialize components
        logger.info("Initializing components...")
        
        # Audio capture
        audio = AudioCapture(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            vad_type="webrtc",
            vad_aggressiveness=config.audio.vad_aggressiveness,
            silence_timeout=config.audio.silence_timeout,
            max_recording_duration=getattr(config.audio, "max_recording_duration", 30.0),
        )
        logger.info("✓ Audio capture initialized")
        
        # Model manager
        model_manager = get_model_manager()
        logger.info("✓ Model manager initialized")
        
        # Transcriber (using factory)
        transcoder = create_transcriber()
        if not transcoder:
            QMessageBox.critical(
                None,
                "Initialization Error",
                "Failed to create transcriber. Check that the selected ASR engine is installed."
            )
            return 1
        logger.info(f"✓ Transcriber initialized ({config.model.asr_engine})")
        
        # Create main window
        window = MainWindow(config, audio, transcoder, model_manager)
        window.show()
        
        logger.info("Application ready")
        
        # Run application
        return app.exec()
        
    except Exception as e:
        logger.exception("="*60)
        logger.exception("CRITICAL: Failed to start application")
        logger.exception("="*60)
        logger.exception(f"Exception type: {type(e).__name__}")
        logger.exception(f"Exception message: {e}")
        
        # Log to crash handler if available
        if crash_handler:
            import traceback
            crash_handler._crash_occurred = True
            crash_handler._write_crash_report(type(e), e, traceback.format_exc())
        
        print(f"\n" + "="*60)
        print(f"CRITICAL ERROR: {e}")
        print("="*60)
        print("\nCheck logs/ directory for crash reports")
        return 1


if __name__ == "__main__":
    sys.exit(main())
