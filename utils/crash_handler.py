"""
Crash Handler and Debug Logging for CripIt
Provides comprehensive error tracking and network monitoring
"""

import sys
import os
import logging
import traceback
import atexit
import signal
from datetime import datetime
from pathlib import Path

# Create crash log directory (in project root)
CRASH_LOG_DIR = Path(__file__).parent.parent / "logs"
CRASH_LOG_DIR.mkdir(exist_ok=True)

# Setup crash logger
crash_logger = logging.getLogger('cripit.crash')
crash_logger.setLevel(logging.DEBUG)

# File handler for crashes
crash_file = CRASH_LOG_DIR / f"crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
crash_handler = logging.FileHandler(crash_file)
crash_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
crash_handler.setFormatter(formatter)
crash_logger.addHandler(crash_handler)

# Also log to console
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
crash_logger.addHandler(console_handler)


class CrashHandler:
    """Handles unexpected crashes and exits."""
    
    def __init__(self):
        self._original_excepthook = sys.excepthook
        self._crash_occurred = False
        self._exit_code = 0
        
        # Install handlers
        sys.excepthook = self._handle_exception
        atexit.register(self._handle_exit)
        
        # Signal handlers for abrupt termination
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        
        crash_logger.info("="*60)
        crash_logger.info("CrashHandler initialized")
        crash_logger.info(f"Crash log: {crash_file}")
        crash_logger.info("="*60)
    
    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        self._crash_occurred = True
        
        # Log the exception
        crash_logger.error("="*60)
        crash_logger.error("UNCAUGHT EXCEPTION")
        crash_logger.error("="*60)
        crash_logger.error(f"Type: {exc_type.__name__}")
        crash_logger.error(f"Value: {exc_value}")
        crash_logger.error("Traceback:")
        
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        for line in tb_lines:
            crash_logger.error(line.rstrip())
        
        crash_logger.error("="*60)
        
        # Write crash report
        self._write_crash_report(exc_type, exc_value, tb_lines)
        
        # Call original handler
        self._original_excepthook(exc_type, exc_value, exc_traceback)
    
    def _handle_signal(self, signum, frame):
        """Handle termination signals."""
        signal_name = signal.Signals(signum).name
        crash_logger.warning(f"Received signal {signal_name} ({signum})")
        
        # Log stack trace
        crash_logger.warning("Stack trace at signal:")
        for line in traceback.format_stack(frame):
            crash_logger.warning(line.rstrip())
        
        sys.exit(128 + signum)
    
    def _handle_exit(self):
        """Handle normal exit."""
        if self._crash_occurred:
            crash_logger.error("Application exited after crash")
        else:
            crash_logger.info("Application exited normally")
        
        crash_logger.info("="*60)
    
    def _write_crash_report(self, exc_type, exc_value, tb_lines):
        """Write detailed crash report to file."""
        report_file = CRASH_LOG_DIR / f"crash_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            with open(report_file, 'w') as f:
                f.write("="*70 + "\n")
                f.write("CRIPIT CRASH REPORT\n")
                f.write("="*70 + "\n\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Python: {sys.version}\n")
                f.write(f"Platform: {sys.platform}\n")
                f.write(f"Working Directory: {os.getcwd()}\n")
                f.write(f"\nException Type: {exc_type.__name__}\n")
                f.write(f"Exception Value: {exc_value}\n\n")
                f.write("Traceback:\n")
                f.write("-"*70 + "\n")
                f.writelines(tb_lines)
                f.write("\n" + "="*70 + "\n")
            
            crash_logger.info(f"Crash report written to: {report_file}")
        except Exception as e:
            crash_logger.error(f"Failed to write crash report: {e}")
    
    def log_network_activity(self, url: str, method: str = "GET", status: str = "PENDING"):
        """Log network activity for debugging."""
        crash_logger.debug(f"NETWORK: {method} {url} - {status}")
    
    def log_critical_section(self, section_name: str, action: str):
        """Log entry/exit of critical sections."""
        crash_logger.debug(f"CRITICAL SECTION: {section_name} - {action}")


# Global crash handler instance
_crash_handler = None

def get_crash_handler():
    """Get or create crash handler singleton."""
    global _crash_handler
    if _crash_handler is None:
        _crash_handler = CrashHandler()
    return _crash_handler


if __name__ == "__main__":
    # Test crash handler
    handler = get_crash_handler()
    
    # Simulate a crash
    try:
        x = 1 / 0
    except Exception:
        raise RuntimeError("Test crash")
