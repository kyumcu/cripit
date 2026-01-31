"""
GUI Components for CripIt
PyQt6-based interface for real-time speech-to-text
"""

import logging
from typing import Optional, Callable
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QComboBox, QLabel, QStatusBar,
    QGroupBox, QProgressBar, QSystemTrayIcon, QMenu, QApplication,
    QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSize, QObject
from PyQt6.QtGui import QIcon, QKeySequence, QFont, QAction, QShortcut, QCloseEvent

logger = logging.getLogger(__name__)


class AudioSignals(QObject):
    """Signals for thread-safe audio callbacks."""
    recording_ready = pyqtSignal(object)  # FinalizedRecording
    vad_detected = pyqtSignal(bool)   # is_speech
    state_changed = pyqtSignal(object)  # RecordingState


class PipelineSignals(QObject):
    """Signals for thread-safe pipeline callbacks."""
    backlog_changed = pyqtSignal(int, int)  # queued, processing
    job_started = pyqtSignal(int)  # seq
    job_done = pyqtSignal(int, str, float)  # seq, text, duration
    job_failed = pyqtSignal(int, str, str)  # seq, error, failed_path


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, config, audio_capture, transcriber, model_manager):
        super().__init__()
        
        self.config = config
        self.audio_capture = audio_capture
        self.transcriber = transcriber
        self.model_manager = model_manager

        self.spool = None
        self.pipeline = None
        self.is_recording = False

        # Initialize UI fields to satisfy type checkers; real widgets are built in _setup_ui.
        self.text_display = QTextEdit()
        self.status_label = QLabel("")
        self.vad_label = QLabel("")
        self.model_label = QLabel("")
        self.device_label = QLabel("")
        self.queue_label = QLabel("")
        self.progress_bar = QProgressBar()
        
        # Create signals object for thread-safe callbacks
        self.audio_signals = AudioSignals()
        self.audio_signals.recording_ready.connect(self._handle_recording_ready)
        self.audio_signals.vad_detected.connect(self._handle_vad_detected)
        self.audio_signals.state_changed.connect(self._handle_state_changed)

        self.pipeline_signals = PipelineSignals()
        self.pipeline_signals.backlog_changed.connect(self._handle_backlog_changed)
        self.pipeline_signals.job_started.connect(self._handle_job_started)
        self.pipeline_signals.job_done.connect(self._handle_job_done)
        self.pipeline_signals.job_failed.connect(self._handle_job_failed)
        
        # Timer for VAD updates
        self.vad_timer = QTimer(self)
        self.vad_timer.timeout.connect(self._update_vad_indicator)
        self._current_vad_state = False
        
        logger.info("Initializing MainWindow")
        self._setup_pipeline()
        self._setup_ui()
        self._setup_connections()
        self._setup_shortcuts()
        
        # Apply config
        self.resize(config.ui.window_width, config.ui.window_height)
        self.setWindowTitle(config.ui.window_title)
        
        logger.info("MainWindow initialized")
    
    def _setup_ui(self):
        """Setup user interface."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # === Text Display ===
        text_group = QGroupBox("Transcription")
        text_layout = QVBoxLayout(text_group)
        
        self.text_display = QTextEdit()
        self.text_display.setPlaceholderText("Transcribed text will appear here...")
        font = QFont(self.config.ui.font_family, self.config.ui.font_size)
        self.text_display.setFont(font)
        text_layout.addWidget(self.text_display)
        
        # Text controls
        text_controls = QHBoxLayout()
        
        self.clear_btn = QPushButton("Clear")
        self.copy_btn = QPushButton("Copy to Clipboard")
        self.save_btn = QPushButton("Save to File...")
        
        text_controls.addWidget(self.clear_btn)
        text_controls.addWidget(self.copy_btn)
        text_controls.addStretch()
        text_controls.addWidget(self.save_btn)
        
        text_layout.addLayout(text_controls)
        layout.addWidget(text_group, stretch=1)
        
        # === Controls ===
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        # Record button
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setCheckable(True)
        self.record_btn.setMinimumHeight(40)
        controls_layout.addWidget(self.record_btn)
        
        # Model selector
        controls_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self._populate_models()
        controls_layout.addWidget(self.model_combo)
        
        # Language selector
        controls_layout.addWidget(QLabel("Language:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["Auto-detect", "English", "Spanish", "French", "German", "Italian", "Portuguese", "Chinese", "Japanese", "Korean", "Russian"])
        controls_layout.addWidget(self.lang_combo)
        
        # Settings button
        self.settings_btn = QPushButton("Settings...")
        controls_layout.addWidget(self.settings_btn)
        
        layout.addWidget(controls_group)
        
        # === Status Bar ===
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Status labels
        self.status_label = QLabel("Ready")
        self.vad_label = QLabel("VAD: Idle")
        self.model_label = QLabel(f"Model: {self.config.model.default_model}")
        self.device_label = QLabel("Device: CPU")
        self.queue_label = QLabel("Queue: 0")
        
        self.status_bar.addWidget(self.status_label, stretch=1)
        self.status_bar.addWidget(self.vad_label)
        self.status_bar.addWidget(self.model_label)
        self.status_bar.addWidget(self.device_label)
        self.status_bar.addWidget(self.queue_label)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addWidget(self.progress_bar)
    
    def _populate_models(self):
        """Populate model dropdown."""
        available = self.model_manager.get_available_models()
        all_models = self.model_manager.list_models()
        
        for model in all_models:
            suffix = " ✓" if model in available else " (download required)"
            self.model_combo.addItem(f"{model}{suffix}", model)
        
        # Set current model
        default = self.config.model.default_model
        index = self.model_combo.findData(default)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
    
    def _setup_connections(self):
        """Setup signal/slot connections."""
        # Audio capture callbacks (thread-safe via signals)
        self.audio_capture.on_recording_ready = self._on_recording_ready_threadsafe
        self.audio_capture.on_state_change = self._on_state_change_threadsafe
        self.audio_capture.on_speech_detected = self._on_speech_detected_threadsafe
        
        # Transcriber callbacks
        self.transcriber.on_transcription = self._on_transcription
        self.transcriber.on_error = self._on_transcription_error
        self.transcriber.on_model_loaded = self._on_model_loaded
        
        # Button connections
        self.record_btn.clicked.connect(self._toggle_recording)
        self.clear_btn.clicked.connect(self.text_display.clear)
        self.copy_btn.clicked.connect(self._copy_to_clipboard)
        self.save_btn.clicked.connect(self._save_to_file)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        self.settings_btn.clicked.connect(self._open_settings)
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Ctrl+R - Toggle recording
        shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        shortcut.activated.connect(self._toggle_recording)
        
        # Ctrl+S - Save
        shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        shortcut_save.activated.connect(self._save_to_file)
        
        # Ctrl+Shift+C - Copy
        shortcut_copy = QShortcut(QKeySequence("Ctrl+Shift+C"), self)
        shortcut_copy.activated.connect(self._copy_to_clipboard)
    
    def _on_recording_ready_threadsafe(self, recording):
        """Thread-safe wrapper for recording ready callback."""
        self.audio_signals.recording_ready.emit(recording)
    
    def _on_state_change_threadsafe(self, state):
        """Thread-safe wrapper for state change callback."""
        self.audio_signals.state_changed.emit(state)
    
    def _on_speech_detected_threadsafe(self, is_speech):
        """Thread-safe wrapper for speech detection callback."""
        self.audio_signals.vad_detected.emit(is_speech)
    
    def _handle_recording_ready(self, recording):
        """Handle finalized recording (main thread).

        This is the durable boundary: we spool to disk first (never drop due to
        transcription backlog), then enqueue a job to the sequential pipeline.
        """
        if not self.spool or not self.pipeline:
            logger.error("Pipeline not initialized; cannot accept recording")
            return

        try:
            job = self.spool.enqueue(recording)
        except Exception as e:
            # Disk-low or write failure: stop recording and show error.
            logger.error(f"Failed to spool recording: {e}")
            if self.is_recording:
                self._stop_recording()
            QMessageBox.critical(
                self,
                "Recording Stopped",
                "CripIt stopped recording because it could not spool the next recording segment.\n\n"
                f"Reason: {e}\n\n"
                "Spool directory: output/spool\n"
                "Failed jobs (if any) are kept in: output/spool/failed",
            )
            return

        self.pipeline.enqueue(job)

        if self.spool.should_stop_for_low_disk() and self.is_recording:
            logger.warning("Low disk detected after spooling; stopping recording")
            self._stop_recording()
            QMessageBox.warning(
                self,
                "Low Disk Space",
                "Disk space is running low. Recording has been stopped to avoid dropping segments.\n\n"
                "Free up space and start recording again.",
            )
    
    def _handle_vad_detected(self, is_speech):
        """Handle VAD detection (main thread)."""
        self._current_vad_state = is_speech
        if not self.vad_timer.isActive():
            self.vad_timer.start(100)  # Update every 100ms
    
    def _update_vad_indicator(self):
        """Update VAD indicator in status bar."""
        if self._current_vad_state:
            self.vad_label.setText("VAD: Speech detected")
            self.vad_label.setStyleSheet("color: green;")
        else:
            self.vad_label.setText("VAD: Silence")
            self.vad_label.setStyleSheet("color: gray;")
    
    def _handle_state_changed(self, state):
        """Handle state change (main thread)."""
        logger.info(f"Audio state: {state}")
    
    def _toggle_recording(self):
        """Toggle recording on/off."""
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()
    
    def _start_recording(self):
        """Start audio recording."""
        logger.info("Starting recording...")

        # Apply latest config to runtime components before starting capture.
        self._apply_runtime_settings()
        
        # Check if model is loaded
        if not self.transcriber.is_ready():
            # Try to load model
            current_model = self.model_combo.currentData()
            model_path = self.model_manager.get_model_path(current_model)
            
            if not model_path:
                QMessageBox.warning(self, "Model Not Available", 
                    f"Model '{current_model}' is not downloaded.\n\n"
                    "Please download the model from Settings.")
                return
            
            logger.info(f"Loading model: {current_model}")
            self.status_label.setText(f"Loading model: {current_model}...")
            
            if not self.transcriber.load_model(str(model_path)):
                QMessageBox.critical(self, "Error", "Failed to load model!")
                return
        
        # Start audio capture
        if self.audio_capture.start():
            self.is_recording = True
            self.record_btn.setText("Stop Recording")
            self.record_btn.setChecked(True)
            self.status_label.setText("Recording... (speak now)")
            logger.info("Recording started")
        else:
            QMessageBox.critical(self, "Error", "Failed to start audio capture!")

    def _apply_runtime_settings(self):
        """Apply config changes to running objects (best-effort).

        Some settings only take effect on the next start of audio capture.
        """
        # ---- AudioCapture ----
        try:
            # These are safe to update at runtime (used by the capture loop).
            self.audio_capture.silence_timeout = float(getattr(self.config.audio, "silence_timeout", 1.5))
            self.audio_capture.max_recording_duration = float(getattr(self.config.audio, "max_recording_duration", 30.0))

            # Gain can be changed live.
            if hasattr(self.audio_capture, "set_gain"):
                self.audio_capture.set_gain(float(getattr(self.config.audio, "gain_db", 0.0)))

            # Device switch only applies when not recording.
            if hasattr(self.audio_capture, "set_device"):
                self.audio_capture.set_device(getattr(self.config.audio, "device_index", None))
        except Exception as e:
            logger.warning(f"Failed to apply audio settings at runtime: {e}")

        # ---- Transcriber ----
        try:
            self.transcriber.use_cuda = bool(getattr(self.config.model, "use_cuda", True))
            self.transcriber.cuda_device = int(getattr(self.config.model, "cuda_device", 0))
            # These are used during transcription/model load.
            self.transcriber.n_threads = int(getattr(self.config.model, "n_threads", self.transcriber.n_threads))
            self.transcriber.translate = bool(getattr(self.config.model, "translate", self.transcriber.translate))
            self.transcriber.language = getattr(self.config.model, "language", self.transcriber.language)
        except Exception as e:
            logger.warning(f"Failed to apply transcriber settings at runtime: {e}")

        # ---- Spool thresholds ----
        try:
            if self.spool and hasattr(self.config, "spool"):
                soft_mb = int(getattr(self.config.spool, "soft_min_free_mb", 1024))
                hard_mb = int(getattr(self.config.spool, "hard_reserve_mb", 256))
                self.spool.soft_min_free_bytes = soft_mb * 1024 * 1024
                self.spool.hard_reserve_bytes = hard_mb * 1024 * 1024

                # Changing spool dir at runtime is only safe when idle.
                new_dir = Path(getattr(self.config.spool, "dir", str(self.spool.root_dir)))
                if new_dir.resolve() != Path(self.spool.root_dir).resolve():
                    backlog = self.pipeline.backlog() if self.pipeline else None
                    if self.is_recording or (backlog and (backlog.processing or backlog.queued)):
                        logger.warning("Spool dir changed but pipeline is active; change will take effect after restart")
                    else:
                        logger.info(f"Spool dir changed to {new_dir}; reinitializing pipeline")
                        # Reinitialize pipeline/spool cleanly.
                        try:
                            if self.pipeline:
                                self.pipeline.stop(timeout=2.0)
                        finally:
                            self.spool = None
                            self.pipeline = None
                        self._setup_pipeline()
        except Exception as e:
            logger.warning(f"Failed to apply spool settings at runtime: {e}")
    
    def _stop_recording(self):
        """Stop audio recording."""
        logger.info("Stopping recording...")
        
        self.audio_capture.stop()
        self.is_recording = False
        self.record_btn.setText("Start Recording")
        self.record_btn.setChecked(False)
        self.status_label.setText("Ready")
        self.vad_label.setText("VAD: Idle")
        self.vad_label.setStyleSheet("")
        
        if self.vad_timer.isActive():
            self.vad_timer.stop()
        
        logger.info("Recording stopped")
    
    def _handle_backlog_changed(self, queued: int, processing: int):
        if not self.queue_label:
            return
        self.queue_label.setText(f"Queue: {queued}{' (processing)' if processing else ''}")

    def _handle_job_started(self, seq: int):
        self.status_label.setText(f"Transcribing job #{seq}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

    def _handle_job_done(self, seq: int, text: str, duration: float):
        logger.info(f"Job #{seq} complete: {len(text)} chars")
        if text.strip():
            current = self.text_display.toPlainText()
            if current:
                self.text_display.append("")
            self.text_display.append(text)
            if self.config.ui.auto_copy:
                self._copy_to_clipboard()

        stats = self.transcriber.get_stats()
        self.status_label.setText(
            f"Job #{seq} done | {len(text)} chars | RTF: {stats.get('avg_realtime_factor', 0):.2f}x"
        )

        # Hide progress bar if nothing is queued
        if self.pipeline and self.pipeline.backlog().queued == 0:
            self.progress_bar.setVisible(False)

    def _handle_job_failed(self, seq: int, error: str, failed_path: str):
        logger.error(f"Job #{seq} failed: {error}")
        self.status_label.setText(f"Job #{seq} failed")
        QMessageBox.warning(
            self,
            "Transcription Failed",
            f"Job #{seq} failed and was kept for debugging.\n\nError: {error}\n\nFile: {failed_path}",
        )
    
    def _on_transcription(self, result):
        """Handle transcription result (from async worker)."""
        try:
            txt = getattr(result, "text", "")
            logger.info(f"Transcription callback: {len(txt)} chars")
        except Exception:
            logger.info("Transcription callback")
    
    def _on_transcription_error(self, error):
        """Handle transcription error."""
        logger.error(f"Transcription error: {error}")
        self.status_label.setText(f"Error: {error}")
    
    def _on_model_loaded(self):
        """Handle model loaded."""
        logger.info("Model loaded")
        self.model_label.setText(f"Model: {self.transcriber.model_name}")
        try:
            self.device_label.setText(f"Device: {self.transcriber.get_device_name()}")
        except Exception:
            self.device_label.setText("Device: Unknown")
    
    def _on_model_changed(self, index):
        """Handle model selection change."""
        model = self.model_combo.currentData()
        logger.info(f"Model changed to: {model}")
        
        # Unload current model
        self.transcriber.unload_model()
        self.model_label.setText(f"Model: {model} (not loaded)")
        self.device_label.setText("Device: -")
        
        # Check if available
        if not self.model_manager.is_model_available(model):
            info = self.model_manager.get_model_info(model)
            size_str = f"{info.size_mb} MB" if info else "unknown size"
            
            reply = QMessageBox.question(
                self, "Download Model",
                f"Model '{model}' is not available locally.\n\n"
                f"Download now? ({size_str})",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self._download_model(model)
    
    def _on_language_changed(self, index):
        """Handle language selection change."""
        lang_map = {
            0: None,  # Auto-detect
            1: "en",
            2: "es",
            3: "fr",
            4: "de",
            5: "it",
            6: "pt",
            7: "zh",
            8: "ja",
            9: "ko",
            10: "ru"
        }
        
        lang = lang_map.get(index)
        logger.info(f"Language changed to: {lang or 'auto'}")
        self.transcriber.language = lang
        self.config.model.language = lang
    
    def _download_model(self, model_name: str):
        """Download a model."""
        logger.info(f"Downloading model: {model_name}")
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.status_label.setText(f"Downloading {model_name}...")
        
        def on_progress(name, percent, total):
            self.progress_bar.setValue(percent)
            self.status_label.setText(f"Downloading {name}: {percent}%")
        
        def on_complete(name, success):
            self.progress_bar.setVisible(False)
            if success:
                self.status_label.setText(f"Downloaded {name}")
                # Refresh model list
                self.model_combo.clear()
                self._populate_models()
            else:
                self.status_label.setText(f"Failed to download {name}")
                QMessageBox.critical(self, "Error", f"Failed to download model {name}")
        
        self.model_manager.on_progress = on_progress
        self.model_manager.on_complete = on_complete
        self.model_manager.download_model(model_name, blocking=False)
    
    def _copy_to_clipboard(self):
        """Copy text to clipboard."""
        text = self.text_display.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            if clipboard is None:
                return
            clipboard.setText(text)
            self.status_label.setText("Copied to clipboard")
            logger.info("Text copied to clipboard")
    
    def _save_to_file(self):
        """Save transcription to file."""
        text = self.text_display.toPlainText()
        if not text:
            QMessageBox.information(self, "Info", "No text to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Transcription",
            "", "Text Files (*.txt);;All Files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.status_label.setText(f"Saved to {file_path}")
                logger.info(f"Transcription saved to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")
    
    def _open_settings(self):
        """Open the settings dialog."""
        logger.info("Opening settings dialog...")
        
        try:
            from gui.settings_dialog import SettingsDialog
            
            dialog = SettingsDialog(self.config, self.model_manager, self)
            dialog.models_changed.connect(self._on_models_changed)
            dialog.config_changed.connect(self._apply_runtime_settings)
            dialog.exec()

            # Apply updated settings immediately (best-effort)
            self._apply_runtime_settings()

            # Update device label if available
            try:
                if hasattr(self, "device_label"):
                    self.device_label.setText(f"Device: {self.transcriber.get_device_name()}")
            except Exception:
                pass
            
            logger.info("Settings dialog closed")
            
        except Exception as e:
            logger.exception(f"Failed to open settings dialog: {e}")
            QMessageBox.critical(self, "Error", f"Could not open settings: {e}")
    
    def _on_models_changed(self):
        """Handle model changes from settings dialog."""
        logger.info("Models changed, refreshing model list...")
        
        # Remember current selection
        current_model = self.model_combo.currentData()
        
        # Refresh the model dropdown
        self.model_combo.clear()
        self._populate_models()
        
        # Try to restore selection
        if current_model:
            index = self.model_combo.findData(current_model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
    
    def closeEvent(self, a0: Optional[QCloseEvent]):
        """Handle window close."""
        logger.info("Closing application...")
        
        # Stop recording
        if self.is_recording:
            self._stop_recording()
        
        # Save config
        self.config.save_config()
        
        # Stop worker
        if self.pipeline:
            try:
                self.pipeline.stop(timeout=2.0)
            except Exception:
                pass
        
        logger.info("Application closed")
        if a0 is not None:
            a0.accept()

    def _setup_pipeline(self):
        """Initialize spool + sequential transcription pipeline."""
        try:
            from core.recording_spool import RecordingSpool
            from core.transcription_pipeline import TranscriptionPipeline

            spool_dir = Path("output") / "spool"
            # Config extension may add config.spool later; keep a safe default.
            cfg_spool = getattr(self.config, "spool", None)
            if cfg_spool is not None:
                spool_dir = Path(getattr(cfg_spool, "dir", str(spool_dir)))
                soft_mb = int(getattr(cfg_spool, "soft_min_free_mb", 1024))
                hard_mb = int(getattr(cfg_spool, "hard_reserve_mb", 256))
            else:
                soft_mb = 1024
                hard_mb = 256

            self.spool = RecordingSpool(spool_dir, soft_min_free_mb=soft_mb, hard_reserve_mb=hard_mb)
            self.pipeline = TranscriptionPipeline(spool_root=spool_dir, transcriber=self.transcriber)

            self.pipeline.on_backlog_changed = lambda b: self.pipeline_signals.backlog_changed.emit(b.queued, b.processing)
            self.pipeline.on_job_started = lambda job: self.pipeline_signals.job_started.emit(int(job.seq))
            self.pipeline.on_job_done = lambda job, result: self.pipeline_signals.job_done.emit(
                int(job.seq),
                str(getattr(result, "text", "")),
                float(getattr(result, "duration", 0.0)),
            )
            self.pipeline.on_job_failed = lambda job, err, path: self.pipeline_signals.job_failed.emit(int(job.seq), str(err), str(path))

            self.pipeline.start()
            recovered = self.pipeline.recover()
            if recovered:
                logger.info(f"Recovered {recovered} queued job(s) from spool")
        except Exception as e:
            logger.exception(f"Failed to initialize spool pipeline: {e}")
            self.spool = None
            self.pipeline = None
