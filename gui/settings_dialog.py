"""
Settings Dialog for CripIt
Model management and configuration settings
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path
from typing import Optional, List

from core.audio_capture import AudioCapture
from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QListWidget, QListWidgetItem, QMessageBox,
    QGroupBox, QProgressBar, QSplitter, QTextEdit,
    QComboBox, QSlider, QSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont

logger = logging.getLogger(__name__)


class ModelDownloadThread(QThread):
    """Background thread for model downloads in settings."""
    
    progress = pyqtSignal(str, int)  # model_name, percent
    completed = pyqtSignal(str, bool, str)  # model_name, success, message
    
    def __init__(self, model_manager, model_name):
        super().__init__()
        self.model_manager = model_manager
        self.model_name = model_name
    
    def run(self):
        """Download model in background."""
        try:
            info = self.model_manager.get_model_info(self.model_name)
            if not info:
                self.completed.emit(str(self.model_name), False, "Model info not found")
                return
            
            # Progress callback
            def on_progress(name, percent, total):
                try:
                    # Ensure we're emitting proper types
                    model_str = str(name) if name else "unknown"
                    percent_int = int(percent) if percent else 0
                    self.progress.emit(model_str, percent_int)
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")
            
            # Completion callback  
            def on_complete(name, success):
                try:
                    model_str = str(name) if name else "unknown"
                    success_bool = bool(success)
                    if success_bool:
                        self.completed.emit(model_str, True, f"Successfully downloaded {model_str}")
                    else:
                        self.completed.emit(model_str, False, f"Failed to download {model_str}")
                except Exception as e:
                    logger.error(f"Error in complete callback: {e}")
            
            self.model_manager.on_progress = on_progress
            self.model_manager.on_complete = on_complete
            
            # Start download (blocking in this thread)
            result = self.model_manager.download_model(self.model_name, blocking=True)
            
            if not result:
                self.completed.emit(str(self.model_name), False, "Download failed to start")
                
        except Exception as e:
            logger.exception(f"Error downloading {self.model_name}")
            self.completed.emit(str(self.model_name), False, f"Error: {str(e)}")


class SettingsDialog(QDialog):
    """Settings dialog with model management."""
    
    models_changed = pyqtSignal()  # Emitted when models are added/removed
    
    def __init__(self, config, model_manager, parent=None):
        super().__init__(parent)
        
        self.config = config
        self.model_manager = model_manager
        self.download_thread: Optional[ModelDownloadThread] = None
        
        self.setWindowTitle("CripIt Settings - Model Management")
        self.setMinimumSize(700, 500)
        
        self._setup_ui()
        self._refresh_model_list()
        
        logger.info("Settings dialog initialized")
    
    def _setup_ui(self):
        """Setup the settings UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # === Audio Settings Group ===
        audio_group = QGroupBox("Audio Settings")
        audio_layout = QVBoxLayout(audio_group)
        
        # Microphone selection
        mic_layout = QHBoxLayout()
        mic_layout.addWidget(QLabel("Microphone:"))
        self.mic_combo = QComboBox()
        self.mic_combo.setMinimumWidth(300)
        self._populate_microphone_list()
        mic_layout.addWidget(self.mic_combo, stretch=1)
        audio_layout.addLayout(mic_layout)
        
        # Gain control
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain (dB):"))
        
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(-20, 20)
        self.gain_slider.setValue(int(self.config.audio.gain_db))
        self.gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.gain_slider.setTickInterval(5)
        gain_layout.addWidget(self.gain_slider, stretch=1)
        
        self.gain_spinbox = QSpinBox()
        self.gain_spinbox.setRange(-20, 20)
        self.gain_spinbox.setValue(int(self.config.audio.gain_db))
        self.gain_spinbox.setSuffix(" dB")
        gain_layout.addWidget(self.gain_spinbox)
        
        audio_layout.addLayout(gain_layout)
        
        # Connect slider and spinbox
        self.gain_slider.valueChanged.connect(self.gain_spinbox.setValue)
        self.gain_spinbox.valueChanged.connect(self.gain_slider.setValue)
        
        # Apply button
        apply_layout = QHBoxLayout()
        apply_layout.addStretch()
        self.apply_audio_btn = QPushButton("Apply Audio Settings")
        self.apply_audio_btn.clicked.connect(self._apply_audio_settings)
        apply_layout.addWidget(self.apply_audio_btn)
        audio_layout.addLayout(apply_layout)
        
        layout.addWidget(audio_group)

        # === GPU / CUDA Settings Group ===
        cuda_group = QGroupBox("GPU Acceleration (CUDA)")
        cuda_layout = QVBoxLayout(cuda_group)

        self.cuda_status_label = QLabel("CUDA status: checking...")
        self.cuda_status_label.setWordWrap(True)
        cuda_layout.addWidget(self.cuda_status_label)

        self.use_cuda_checkbox = QCheckBox("Enable GPU acceleration when available")
        self.use_cuda_checkbox.setChecked(bool(getattr(self.config.model, "use_cuda", True)))
        cuda_layout.addWidget(self.use_cuda_checkbox)

        cuda_device_layout = QHBoxLayout()
        cuda_device_layout.addWidget(QLabel("CUDA device:"))
        self.cuda_device_spinbox = QSpinBox()
        self.cuda_device_spinbox.setRange(0, 16)
        self.cuda_device_spinbox.setValue(int(getattr(self.config.model, "cuda_device", 0)))
        cuda_device_layout.addWidget(self.cuda_device_spinbox)
        cuda_device_layout.addStretch()
        cuda_layout.addLayout(cuda_device_layout)

        self.cuda_fallback_checkbox = QCheckBox("Auto-fallback to CPU if GPU fails")
        self.cuda_fallback_checkbox.setChecked(bool(getattr(self.config.model, "cuda_fallback_to_cpu", True)))
        cuda_layout.addWidget(self.cuda_fallback_checkbox)

        self.cuda_warn_checkbox = QCheckBox("Warn when falling back to CPU")
        self.cuda_warn_checkbox.setChecked(bool(getattr(self.config.model, "cuda_warn_on_fallback", True)))
        cuda_layout.addWidget(self.cuda_warn_checkbox)

        cuda_buttons = QHBoxLayout()
        self.cuda_build_btn = QPushButton("Build CUDA Backend...")
        self.cuda_build_btn.clicked.connect(self._show_cuda_build_instructions)
        cuda_buttons.addWidget(self.cuda_build_btn)

        self.cuda_refresh_btn = QPushButton("Refresh CUDA Status")
        self.cuda_refresh_btn.clicked.connect(self._refresh_cuda_status)
        cuda_buttons.addWidget(self.cuda_refresh_btn)

        cuda_buttons.addStretch()
        self.apply_cuda_btn = QPushButton("Apply GPU Settings")
        self.apply_cuda_btn.clicked.connect(self._apply_cuda_settings)
        cuda_buttons.addWidget(self.apply_cuda_btn)
        cuda_layout.addLayout(cuda_buttons)

        layout.addWidget(cuda_group)
        
        # === Model Management Group ===
        models_group = QGroupBox("Model Management")
        models_layout = QVBoxLayout(models_group)
        
        # Info label
        info_label = QLabel("Manage downloaded models. You can delete models to free space or reinstall them.")
        info_label.setWordWrap(True)
        models_layout.addWidget(info_label)
        
        # Splitter for list and details
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Model list
        self.model_list = QListWidget()
        self.model_list.setMinimumWidth(300)
        self.model_list.currentItemChanged.connect(self._on_model_selected)
        splitter.addWidget(self.model_list)
        
        # Details panel
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(5, 5, 5, 5)
        
        # Model info
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setMaximumHeight(150)
        details_layout.addWidget(QLabel("Model Information:"))
        details_layout.addWidget(self.model_info_text)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        self.delete_btn = QPushButton("🗑️ Delete Model")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._delete_selected_model)
        buttons_layout.addWidget(self.delete_btn)
        
        self.reinstall_btn = QPushButton("🔄 Reinstall Model")
        self.reinstall_btn.setEnabled(False)
        self.reinstall_btn.clicked.connect(self._reinstall_selected_model)
        buttons_layout.addWidget(self.reinstall_btn)
        
        buttons_layout.addStretch()
        details_layout.addLayout(buttons_layout)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        details_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Select a model to view details")
        details_layout.addWidget(self.status_label)
        
        details_layout.addStretch()
        splitter.addWidget(details_widget)
        splitter.setSizes([300, 400])
        
        models_layout.addWidget(splitter)
        layout.addWidget(models_group, stretch=1)
        
        # === Disk Usage Group ===
        disk_group = QGroupBox("Disk Usage")
        disk_layout = QHBoxLayout(disk_group)
        
        self.disk_usage_label = QLabel("Calculating...")
        disk_layout.addWidget(self.disk_usage_label)
        disk_layout.addStretch()
        
        self.refresh_disk_btn = QPushButton("🔄 Refresh")
        self.refresh_disk_btn.clicked.connect(self._update_disk_usage)
        disk_layout.addWidget(self.refresh_disk_btn)
        
        layout.addWidget(disk_group)
        
        # === Dialog Buttons ===
        dialog_buttons = QHBoxLayout()
        dialog_buttons.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        dialog_buttons.addWidget(close_btn)
        
        layout.addLayout(dialog_buttons)
        
        # Update disk usage
        self._update_disk_usage()

        # Update CUDA status
        self._refresh_cuda_status()
    
    def _refresh_model_list(self):
        """Refresh the model list."""
        self.model_list.clear()
        
        available = self.model_manager.get_available_models()
        all_models = self.model_manager.list_models()
        
        for model_name in all_models:
            info = self.model_manager.get_model_info(model_name)
            if not info:
                continue
            
            is_available = model_name in available
            
            # Create list item
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, model_name)
            
            if is_available:
                # Get actual file size
                size_bytes = self.model_manager.get_model_size(model_name) or 0
                size_mb = size_bytes / (1024 * 1024)
                item.setText(f"✓ {model_name} ({size_mb:.1f} MB)")
                item.setToolTip(f"Installed - {info.description}")
            else:
                item.setText(f"  {model_name} (Not installed)")
                item.setToolTip(f"Not installed - {info.description}")
                # Gray out not installed models
                font = item.font()
                font.setItalic(True)
                item.setFont(font)
            
            self.model_list.addItem(item)
        
        logger.info(f"Model list refreshed: {len(available)} installed, {len(all_models) - len(available)} missing")
    
    def _on_model_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Handle model selection."""
        if not current:
            self.delete_btn.setEnabled(False)
            self.reinstall_btn.setEnabled(False)
            self.model_info_text.clear()
            return
        
        model_name = current.data(Qt.ItemDataRole.UserRole)
        info = self.model_manager.get_model_info(model_name)
        
        if not info:
            return
        
        is_available = self.model_manager.is_model_available(model_name)
        
        # Update info text
        info_text = f"""
<b>Name:</b> {info.name}
<b>Parameters:</b> {info.params}
<b>Expected Size:</b> {info.size_mb} MB
<b>File:</b> {info.file}
<b>Status:</b> {'Installed ✓' if is_available else 'Not installed'}
<b>Description:</b> {info.description}
"""
        self.model_info_text.setHtml(info_text)
        
        # Enable/disable buttons
        self.delete_btn.setEnabled(is_available)
        self.reinstall_btn.setEnabled(True)  # Can always reinstall
        
        if is_available:
            self.status_label.setText(f"Model '{model_name}' is installed. You can delete or reinstall it.")
        else:
            self.status_label.setText(f"Model '{model_name}' is not installed. Click 'Reinstall' to download.")
    
    def _delete_selected_model(self):
        """Delete the selected model."""
        current = self.model_list.currentItem()
        if not current:
            return
        
        model_name = current.data(Qt.ItemDataRole.UserRole)
        info = self.model_manager.get_model_info(model_name)
        
        if not info:
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the model '{model_name}'?\n\n"
            f"This will free up {info.size_mb} MB of disk space.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Delete the model
        logger.info(f"User requested deletion of model: {model_name}")
        success = self.model_manager.delete_model(model_name)
        
        if success:
            QMessageBox.information(self, "Success", f"Model '{model_name}' has been deleted.")
            self._refresh_model_list()
            self._update_disk_usage()
            self.models_changed.emit()
            logger.info(f"Model deleted successfully: {model_name}")
        else:
            QMessageBox.critical(self, "Error", f"Failed to delete model '{model_name}'.")
            logger.error(f"Failed to delete model: {model_name}")
    
    def _reinstall_selected_model(self):
        """Reinstall (download) the selected model."""
        current = self.model_list.currentItem()
        if not current:
            return
        
        model_name = current.data(Qt.ItemDataRole.UserRole)
        info = self.model_manager.get_model_info(model_name)
        
        if not info:
            return
        
        # Check if already downloading
        if self.model_manager.is_downloading(model_name):
            QMessageBox.information(self, "In Progress", f"Model '{model_name}' is already being downloaded.")
            return
        
        # Confirm if already installed (reinstall)
        is_available = self.model_manager.is_model_available(model_name)
        if is_available:
            reply = QMessageBox.question(
                self,
                "Confirm Reinstall",
                f"Model '{model_name}' is already installed.\n\n"
                f"Do you want to delete it and download again?\n"
                f"This will use {info.size_mb} MB of bandwidth.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
            
            # Delete first
            logger.info(f"Deleting model for reinstall: {model_name}")
            if not self.model_manager.delete_model(model_name):
                QMessageBox.critical(self, "Error", f"Failed to delete existing model '{model_name}'.")
                return
        
        # Start download
        logger.info(f"Starting reinstall of model: {model_name}")
        self._start_download(model_name)
    
    def _start_download(self, model_name: str):
        """Start downloading a model."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 100)
        self.status_label.setText(f"Downloading {model_name}...")
        self.delete_btn.setEnabled(False)
        self.reinstall_btn.setEnabled(False)
        
        # Create download thread
        self.download_thread = ModelDownloadThread(self.model_manager, model_name)
        self.download_thread.progress.connect(self._on_download_progress)
        self.download_thread.completed.connect(self._on_download_completed)
        self.download_thread.start()
    
    def _on_download_progress(self, model_name: str, percent: int):
        """Handle download progress."""
        try:
            self.progress_bar.setValue(int(percent))
            self.status_label.setText(f"Downloading {model_name}: {percent}%")
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
    
    def _on_download_completed(self, model_name: str, success: bool, message: str):
        """Handle download completion."""
        try:
            self.progress_bar.setVisible(False)
            self.status_label.setText(str(message))
            
            if success:
                QMessageBox.information(self, "Success", str(message))
                self._refresh_model_list()
                self._update_disk_usage()
                self.models_changed.emit()
            else:
                QMessageBox.critical(self, "Error", str(message))
            
            # Re-enable buttons
            self._on_model_selected(self.model_list.currentItem(), None)
        except Exception as e:
            logger.error(f"Error handling download completion: {e}")
    
    def _update_disk_usage(self):
        """Update disk usage display."""
        try:
            total_size = self.model_manager.get_total_size()
            available = len(self.model_manager.get_available_models())
            total = len(self.model_manager.list_models())
            
            total_mb = total_size / (1024 * 1024)
            total_gb = total_size / (1024 * 1024 * 1024)
            
            if total_gb >= 1:
                size_str = f"{total_gb:.2f} GB"
            else:
                size_str = f"{total_mb:.1f} MB"
            
            self.disk_usage_label.setText(
                f"{available} of {total} models installed | Total size: {size_str}"
            )
            
            logger.debug(f"Disk usage updated: {size_str} for {available} models")
            
        except Exception as e:
            logger.error(f"Failed to update disk usage: {e}")
            self.disk_usage_label.setText("Unable to calculate disk usage")
    
    def _populate_microphone_list(self):
        """Populate the microphone dropdown with available devices."""
        self.mic_combo.clear()
        
        # Add default option
        self.mic_combo.addItem("Default", None)
        
        try:
            devices = AudioCapture.list_devices()
            current_device = self.config.audio.device_index
            
            for i, device in enumerate(devices):
                device_name = device['name']
                device_index = device['index']
                is_default = device.get('default', False)
                
                # Mark default device with special text
                if is_default and current_device is None:
                    display_name = f"{device_name} (System Default)"
                else:
                    display_name = device_name
                
                self.mic_combo.addItem(display_name, device_index)
                
                # Select current device if configured
                if current_device is not None and device_index == current_device:
                    self.mic_combo.setCurrentIndex(i + 1)  # +1 because of "Default" at index 0
                    
        except Exception as e:
            logger.error(f"Failed to list audio devices: {e}")
            self.mic_combo.addItem("Error loading devices", None)
    
    def _apply_audio_settings(self):
        """Apply audio settings to config."""
        try:
            # Get selected device
            device_index = self.mic_combo.currentData()
            self.config.audio.device_index = device_index
            
            # Get gain value
            gain_db = self.gain_spinbox.value()
            self.config.audio.gain_db = gain_db
            
            # Save config
            if self.config.save_config():
                QMessageBox.information(self, "Success", "Audio settings saved successfully.")
                logger.info(f"Audio settings saved: device={device_index}, gain={gain_db}dB")
            else:
                QMessageBox.warning(self, "Warning", "Failed to save audio settings.")
                
        except Exception as e:
            logger.error(f"Error applying audio settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply audio settings: {str(e)}")

    def _refresh_cuda_status(self):
        """Refresh CUDA status label (best-effort)."""
        try:
            from utils.cuda_utils import CUDAManager
            cm = CUDAManager()
            status = cm.detect_cuda(force_check=True)

            if status.available:
                base = f"CUDA detected (nvcc: {status.nvcc_path or 'unknown'}, version: {status.cuda_version or 'unknown'})"
            else:
                base = f"CUDA not detected ({status.error_message or 'unknown reason'})"

            if status.pywhispercpp_cuda:
                extra = "pywhispercpp: CUDA-linked"
            else:
                extra = "pywhispercpp: CPU-only (or not verified)"

            self.cuda_status_label.setText(f"{base} | {extra}")
            logger.info(f"CUDA status refreshed: available={status.available}, pywhispercpp_cuda={status.pywhispercpp_cuda}")
        except Exception as e:
            self.cuda_status_label.setText(f"CUDA status: error checking ({e})")
            logger.warning(f"Failed to refresh CUDA status: {e}")

    def _apply_cuda_settings(self):
        """Apply CUDA settings to config."""
        try:
            self.config.model.use_cuda = bool(self.use_cuda_checkbox.isChecked())
            self.config.model.cuda_device = int(self.cuda_device_spinbox.value())
            self.config.model.cuda_fallback_to_cpu = bool(self.cuda_fallback_checkbox.isChecked())
            self.config.model.cuda_warn_on_fallback = bool(self.cuda_warn_checkbox.isChecked())

            if self.config.save_config():
                QMessageBox.information(self, "Success", "GPU settings saved successfully.")
                logger.info(
                    "GPU settings saved: "
                    f"use_cuda={self.config.model.use_cuda}, "
                    f"cuda_device={self.config.model.cuda_device}, "
                    f"fallback={self.config.model.cuda_fallback_to_cpu}, "
                    f"warn={self.config.model.cuda_warn_on_fallback}"
                )
            else:
                QMessageBox.warning(self, "Warning", "Failed to save GPU settings.")
        except Exception as e:
            logger.error(f"Error applying GPU settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply GPU settings: {str(e)}")

    def _show_cuda_build_instructions(self):
        """Show instructions for building pywhispercpp with CUDA."""
        QMessageBox.information(
            self,
            "Build CUDA Backend",
            "To enable GPU acceleration, build pywhispercpp with CUDA support:\n\n"
            "1) Activate your environment:\n  conda activate py_cripit\n\n"
            "2) Run the build script from the project directory:\n  bash build_cuda.sh\n\n"
            "After it finishes, click 'Refresh CUDA Status'."
        )
    
    def closeEvent(self, event):
        """Handle dialog close."""
        # Stop any ongoing download
        if self.download_thread and self.download_thread.isRunning():
            logger.warning("Download in progress, waiting for completion...")
            self.download_thread.wait(2000)  # Wait up to 2 seconds
        
        event.accept()
