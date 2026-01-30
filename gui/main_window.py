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
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QObject
from PyQt6.QtGui import QIcon, QKeySequence, QFont, QAction, QShortcut

logger = logging.getLogger(__name__)


class TranscriptionWorker(QThread):
    """Background worker for transcription to avoid blocking GUI."""
    
    transcription_ready = pyqtSignal(str, float)  # text, duration
    error_occurred = pyqtSignal(str)
    
    def __init__(self, transcriber):
        super().__init__()
        self.transcriber = transcriber
        self.audio_data = None
        self._is_running = True
    
    def set_audio(self, audio_data):
        """Set audio data to transcribe."""
        self.audio_data = audio_data
    
    def run(self):
        """Run transcription in background thread."""
        if self.audio_data is None:
            return
        
        try:
            result = self.transcriber.transcribe(self.audio_data)
            
            if result:
                self.transcription_ready.emit(result.text, result.duration)
            else:
                self.error_occurred.emit("Transcription failed")
                
        except Exception as e:
            logger.error(f"Transcription worker error: {e}")
            self.error_occurred.emit(str(e))
    
    def stop(self):
        """Stop the worker."""
        self._is_running = False
        self.wait(1000)


class AudioSignals(QObject):
    """Signals for thread-safe audio callbacks."""
    audio_ready = pyqtSignal(object)  # audio_data
    vad_detected = pyqtSignal(bool)   # is_speech
    state_changed = pyqtSignal(object)  # RecordingState


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, config, audio_capture, transcriber, model_manager):
        super().__init__()
        
        self.config = config
        self.audio_capture = audio_capture
        self.transcriber = transcriber
        self.model_manager = model_manager
        
        self.worker = None
        self.is_recording = False
        
        # Create signals object for thread-safe callbacks
        self.audio_signals = AudioSignals()
        self.audio_signals.audio_ready.connect(self._handle_audio_ready)
        self.audio_signals.vad_detected.connect(self._handle_vad_detected)
        self.audio_signals.state_changed.connect(self._handle_state_changed)
        
        # Timer for VAD updates
        self.vad_timer = QTimer(self)
        self.vad_timer.timeout.connect(self._update_vad_indicator)
        self._current_vad_state = False
        
        logger.info("Initializing MainWindow")
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
        
        self.status_bar.addWidget(self.status_label, stretch=1)
        self.status_bar.addWidget(self.vad_label)
        self.status_bar.addWidget(self.model_label)
        
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
        self.audio_capture.on_audio_ready = self._on_audio_ready_threadsafe
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
    
    def _on_audio_ready_threadsafe(self, audio_data):
        """Thread-safe wrapper for audio ready callback."""
        self.audio_signals.audio_ready.emit(audio_data)
    
    def _on_state_change_threadsafe(self, state):
        """Thread-safe wrapper for state change callback."""
        self.audio_signals.state_changed.emit(state)
    
    def _on_speech_detected_threadsafe(self, is_speech):
        """Thread-safe wrapper for speech detection callback."""
        self.audio_signals.vad_detected.emit(is_speech)
    
    def _handle_audio_ready(self, audio_data):
        """Handle audio ready (main thread)."""
        logger.info(f"Audio ready: {len(audio_data)} samples")
        
        # Check if a transcription is already in progress
        if self.worker and self.worker.isRunning():
            logger.warning("Transcription already in progress, waiting for completion...")
            # Wait for current transcription to finish (with timeout)
            if not self.worker.wait(500):  # Wait up to 500ms
                logger.error("Previous transcription still running, skipping this chunk")
                return
        
        # Run transcription in background thread
        self.worker = TranscriptionWorker(self.transcriber)
        self.worker.transcription_ready.connect(self._on_transcription_complete)
        self.worker.error_occurred.connect(self._on_worker_error)
        self.worker.set_audio(audio_data)
        
        self.status_label.setText("Transcribing...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        self.worker.start()
    
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
    
    def _on_transcription_complete(self, text: str, duration: float):
        """Handle completed transcription."""
        logger.info(f"Transcription complete: {len(text)} chars")
        
        # Append to text display
        if text.strip():
            current = self.text_display.toPlainText()
            if current:
                self.text_display.append("")
            self.text_display.append(text)
            
            # Auto-copy if enabled
            if self.config.ui.auto_copy:
                self._copy_to_clipboard()
        
        self.status_label.setText("Ready")
        self.progress_bar.setVisible(False)
        
        # Show stats in status bar
        stats = self.transcriber.get_stats()
        self.status_label.setText(
            f"Transcribed {len(text)} chars | "
            f"RTF: {stats.get('avg_realtime_factor', 0):.2f}x"
        )
    
    def _on_worker_error(self, error_msg: str):
        """Handle worker error."""
        logger.error(f"Transcription error: {error_msg}")
        self.status_label.setText(f"Error: {error_msg}")
        self.progress_bar.setVisible(False)
        QMessageBox.warning(self, "Transcription Error", error_msg)
    
    def _on_transcription(self, result):
        """Handle transcription result (from async worker)."""
        logger.info(f"Transcription callback: {len(result.text)} chars")
    
    def _on_transcription_error(self, error):
        """Handle transcription error."""
        logger.error(f"Transcription error: {error}")
        self.status_label.setText(f"Error: {error}")
    
    def _on_model_loaded(self):
        """Handle model loaded."""
        logger.info("Model loaded")
        self.model_label.setText(f"Model: {self.transcriber.model_name}")
    
    def _on_model_changed(self, index):
        """Handle model selection change."""
        model = self.model_combo.currentData()
        logger.info(f"Model changed to: {model}")
        
        # Unload current model
        self.transcriber.unload_model()
        self.model_label.setText(f"Model: {model} (not loaded)")
        
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
            dialog.exec()
            
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
    
    def closeEvent(self, event):
        """Handle window close."""
        logger.info("Closing application...")
        
        # Stop recording
        if self.is_recording:
            self._stop_recording()
        
        # Save config
        self.config.save_config()
        
        # Stop worker
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        
        logger.info("Application closed")
        event.accept()
