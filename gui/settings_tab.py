"""Settings Tab for CripIt.

Settings and model management as a tab instead of a dialog.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path
from typing import Optional, List

from core.audio_capture import AudioCapture
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QListWidget, QListWidgetItem, QMessageBox,
    QGroupBox, QProgressBar, QSplitter, QTextEdit,
    QComboBox, QSlider, QSpinBox, QCheckBox, QTabWidget
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


class SettingsTab(QWidget):
    """Settings tab with model management and configuration."""
    
    models_changed = pyqtSignal()  # Emitted when models are added/removed
    config_changed = pyqtSignal()  # Emitted when settings are applied
    
    def __init__(self, config, model_manager, parent=None):
        super().__init__(parent)
        
        self.config = config
        self.model_manager = model_manager
        self.download_thread: Optional[ModelDownloadThread] = None
        
        self._setup_ui()
        self._refresh_model_list()
        self._update_disk_usage()
        self._refresh_cuda_status()
        
        logger.info("Settings tab initialized")
    
    def _setup_ui(self):
        """Setup the settings UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Create inner tab widget for settings categories
        self.inner_tabs = QTabWidget()
        
        # === Audio Settings Tab ===
        audio_tab = QWidget()
        audio_layout = QVBoxLayout(audio_tab)
        audio_layout.setSpacing(10)
        
        # Audio Settings Group
        audio_group = QGroupBox("Audio Settings")
        audio_inner_layout = QVBoxLayout(audio_group)
        
        # Microphone selection
        mic_layout = QHBoxLayout()
        mic_layout.addWidget(QLabel("Microphone:"))
        self.mic_combo = QComboBox()
        self.mic_combo.setMinimumWidth(300)
        self._populate_microphone_list()
        mic_layout.addWidget(self.mic_combo, stretch=1)
        audio_inner_layout.addLayout(mic_layout)
        
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
        
        audio_inner_layout.addLayout(gain_layout)
        
        # Connect slider and spinbox
        self.gain_slider.valueChanged.connect(self.gain_spinbox.setValue)
        self.gain_spinbox.valueChanged.connect(self.gain_slider.setValue)
        
        # Apply button
        apply_layout = QHBoxLayout()
        apply_layout.addStretch()
        self.apply_audio_btn = QPushButton("Apply Audio Settings")
        self.apply_audio_btn.clicked.connect(self._apply_audio_settings)
        apply_layout.addWidget(self.apply_audio_btn)
        audio_inner_layout.addLayout(apply_layout)
        
        audio_layout.addWidget(audio_group)
        audio_layout.addStretch()
        self.inner_tabs.addTab(audio_tab, "Audio")
        
        # === GPU / CUDA Settings Tab ===
        cuda_tab = QWidget()
        cuda_layout = QVBoxLayout(cuda_tab)
        cuda_layout.setSpacing(10)
        
        cuda_group = QGroupBox("GPU Acceleration (CUDA)")
        cuda_inner_layout = QVBoxLayout(cuda_group)
        
        self.cuda_status_label = QLabel("CUDA status: checking...")
        self.cuda_status_label.setWordWrap(True)
        cuda_inner_layout.addWidget(self.cuda_status_label)
        
        self.use_cuda_checkbox = QCheckBox("Enable GPU acceleration when available")
        self.use_cuda_checkbox.setChecked(bool(getattr(self.config.model, "use_cuda", True)))
        cuda_inner_layout.addWidget(self.use_cuda_checkbox)
        
        cuda_device_layout = QHBoxLayout()
        cuda_device_layout.addWidget(QLabel("CUDA device:"))
        self.cuda_device_spinbox = QSpinBox()
        self.cuda_device_spinbox.setRange(0, 16)
        self.cuda_device_spinbox.setValue(int(getattr(self.config.model, "cuda_device", 0)))
        cuda_device_layout.addWidget(self.cuda_device_spinbox)
        cuda_device_layout.addStretch()
        cuda_inner_layout.addLayout(cuda_device_layout)
        
        self.cuda_fallback_checkbox = QCheckBox("Auto-fallback to CPU if GPU fails")
        self.cuda_fallback_checkbox.setChecked(bool(getattr(self.config.model, "cuda_fallback_to_cpu", True)))
        cuda_inner_layout.addWidget(self.cuda_fallback_checkbox)
        
        self.cuda_warn_checkbox = QCheckBox("Warn when falling back to CPU")
        self.cuda_warn_checkbox.setChecked(bool(getattr(self.config.model, "cuda_warn_on_fallback", True)))
        cuda_inner_layout.addWidget(self.cuda_warn_checkbox)
        
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
        cuda_inner_layout.addLayout(cuda_buttons)
        
        cuda_layout.addWidget(cuda_group)
        cuda_layout.addStretch()
        self.inner_tabs.addTab(cuda_tab, "GPU")
        
        # === Model Management Tab ===
        models_tab = QWidget()
        models_layout = QVBoxLayout(models_tab)
        models_layout.setSpacing(10)
        
        models_group = QGroupBox("Model Management")
        models_inner_layout = QVBoxLayout(models_group)
        
        # Info label
        info_label = QLabel("Manage downloaded models. You can delete models to free space or reinstall them.")
        info_label.setWordWrap(True)
        models_inner_layout.addWidget(info_label)
        
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
        
        models_inner_layout.addWidget(splitter)
        models_layout.addWidget(models_group)
        self.inner_tabs.addTab(models_tab, "Models")
        
        # === Disk Usage Tab ===
        disk_tab = QWidget()
        disk_layout = QVBoxLayout(disk_tab)
        disk_layout.setSpacing(10)
        
        disk_group = QGroupBox("Disk Usage")
        disk_inner_layout = QHBoxLayout(disk_group)
        
        self.disk_usage_label = QLabel("Calculating...")
        disk_inner_layout.addWidget(self.disk_usage_label)
        disk_inner_layout.addStretch()
        
        self.refresh_disk_btn = QPushButton("🔄 Refresh")
        self.refresh_disk_btn.clicked.connect(self._update_disk_usage)
        disk_inner_layout.addWidget(self.refresh_disk_btn)
        
        disk_layout.addWidget(disk_group)
        disk_layout.addStretch()
        self.inner_tabs.addTab(disk_tab, "Disk")
        
        # Add inner tabs to main layout
        layout.addWidget(self.inner_tabs)
    
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
<b>Status:</b> {'✓ Installed' if is_available else '❌ Not installed'}

<b>Description:</b><br>{info.description}
"""
        self.model_info_text.setHtml(info_text.strip())
        
        # Update buttons
        self.delete_btn.setEnabled(is_available)
        self.reinstall_btn.setEnabled(True)
        
        if is_available:
            self.status_label.setText(f"Model '{model_name}' is installed and ready to use")
        else:
            self.status_label.setText(f"Model '{model_name}' is not installed. Click 'Reinstall Model' to download.")
    
    def _delete_selected_model(self):
        """Delete the currently selected model."""
        current_item = self.model_list.currentItem()
        if not current_item:
            return
        
        model_name = current_item.data(Qt.ItemDataRole.UserRole)
        
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete the model '{model_name}'?\n\n"
            "This will free up disk space but you'll need to download it again to use it.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                success = self.model_manager.delete_model(model_name)
                if success:
                    self.status_label.setText(f"✓ Deleted {model_name}")
                    self._refresh_model_list()
                    self.models_changed.emit()
                    self._update_disk_usage()
                else:
                    QMessageBox.warning(self, "Error", f"Failed to delete {model_name}")
            except Exception as e:
                logger.exception(f"Error deleting model {model_name}")
                QMessageBox.critical(self, "Error", f"Error deleting model: {e}")
    
    def _reinstall_selected_model(self):
        """Reinstall/download the currently selected model."""
        current_item = self.model_list.currentItem()
        if not current_item:
            return
        
        model_name = current_item.data(Qt.ItemDataRole.UserRole)
        info = self.model_manager.get_model_info(model_name)
        
        if not info:
            QMessageBox.warning(self, "Error", "Model info not found")
            return
        
        # Confirm download
        reply = QMessageBox.question(
            self,
            "Download Model",
            f"Download model '{model_name}' ({info.size_mb} MB)?\n\n"
            f"{info.description}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Disable buttons during download
        self.delete_btn.setEnabled(False)
        self.reinstall_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.status_label.setText(f"Downloading {model_name}...")
        
        # Start download in background thread
        self.download_thread = ModelDownloadThread(self.model_manager, model_name)
        self.download_thread.progress.connect(self._on_download_progress)
        self.download_thread.completed.connect(self._on_download_complete)
        self.download_thread.start()
    
    def _on_download_progress(self, model_name: str, percent: int):
        """Handle download progress update."""
        self.progress_bar.setValue(percent)
        self.status_label.setText(f"Downloading {model_name}: {percent}%")
    
    def _on_download_complete(self, model_name: str, success: bool, message: str):
        """Handle download completion."""
        self.progress_bar.setVisible(False)
        self.status_label.setText(message)
        
        if success:
            self._refresh_model_list()
            self.models_changed.emit()
            self._update_disk_usage()
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)
            # Re-enable buttons if failed
            self._on_model_selected(self.model_list.currentItem(), None)
    
    def _update_disk_usage(self):
        """Update disk usage display."""
        try:
            total_bytes = 0
            available_models = self.model_manager.get_available_models()
            
            for model_name in available_models:
                size = self.model_manager.get_model_size(model_name)
                if size:
                    total_bytes += size
            
            total_mb = total_bytes / (1024 * 1024)
            self.disk_usage_label.setText(f"Models using {total_mb:.1f} MB ({len(available_models)} models)")
            
        except Exception as e:
            logger.error(f"Error calculating disk usage: {e}")
            self.disk_usage_label.setText("Error calculating disk usage")
    
    def _populate_microphone_list(self):
        """Populate the microphone dropdown."""
        try:
            self.mic_combo.clear()
            devices = AudioCapture.list_devices()
            
            # Add default option
            self.mic_combo.addItem("System Default", None)
            
            for device in devices:
                name = device['name']
                index = device['index']
                is_default = device.get('is_default', False)
                
                display_text = f"{name}"
                if is_default:
                    display_text += " (Default)"
                
                self.mic_combo.addItem(display_text, index)
            
            # Set current selection based on config
            current_device = getattr(self.config.audio, "device_index", None)
            for i in range(self.mic_combo.count()):
                if self.mic_combo.itemData(i) == current_device:
                    self.mic_combo.setCurrentIndex(i)
                    break
                    
        except Exception as e:
            logger.error(f"Failed to list audio devices: {e}")
            self.mic_combo.addItem("Error listing devices", None)
    
    def _apply_audio_settings(self):
        """Apply audio settings to config."""
        try:
            device_index = self.mic_combo.currentData()
            gain_db = self.gain_spinbox.value()
            
            self.config.audio.device_index = device_index
            self.config.audio.gain_db = gain_db
            
            if self.config.save_config():
                self.config_changed.emit()
                QMessageBox.information(self, "Success", "Audio settings applied successfully")
            else:
                QMessageBox.warning(self, "Warning", "Failed to save audio settings.")
                
        except Exception as e:
            logger.error(f"Error applying audio settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply audio settings: {str(e)}")
    
    def _refresh_cuda_status(self):
        """Refresh CUDA status display."""
        try:
            from utils.cuda_utils import get_cuda_info
            
            cuda_info = get_cuda_info()
            
            if cuda_info['available']:
                status_text = f"""<b>CUDA Status:</b> ✓ Available
<b>Version:</b> {cuda_info.get('version', 'Unknown')}
<b>GPUs:</b> {cuda_info.get('gpu_count', 0)}
"""
                if cuda_info.get('gpus'):
                    for i, gpu in enumerate(cuda_info['gpus']):
                        status_text += f"<br><b>GPU {i}:</b> {gpu['name']} ({gpu['memory_mb']}MB)"
                
                self.cuda_status_label.setStyleSheet("color: green;")
            else:
                status_text = f"""<b>CUDA Status:</b> ❌ Not Available
<b>Reason:</b> {cuda_info.get('error', 'CUDA not found')}
"""
                self.cuda_status_label.setStyleSheet("color: orange;")
            
            self.cuda_status_label.setText(status_text)
            
        except Exception as e:
            logger.error(f"Error refreshing CUDA status: {e}")
            self.cuda_status_label.setText(f"CUDA Status: Error checking - {e}")
            self.cuda_status_label.setStyleSheet("color: red;")
    
    def _apply_cuda_settings(self):
        """Apply CUDA settings to config."""
        try:
            use_cuda = self.use_cuda_checkbox.isChecked()
            cuda_device = self.cuda_device_spinbox.value()
            cuda_fallback = self.cuda_fallback_checkbox.isChecked()
            cuda_warn = self.cuda_warn_checkbox.isChecked()
            
            self.config.model.use_cuda = use_cuda
            self.config.model.cuda_device = cuda_device
            self.config.model.cuda_fallback_to_cpu = cuda_fallback
            self.config.model.cuda_warn_on_fallback = cuda_warn
            
            if self.config.save_config():
                self.config_changed.emit()
                QMessageBox.information(self, "Success", "GPU settings applied successfully")
            else:
                QMessageBox.warning(self, "Warning", "Failed to save GPU settings.")
                
        except Exception as e:
            logger.error(f"Error applying CUDA settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply GPU settings: {str(e)}")
    
    def _show_cuda_build_instructions(self):
        """Show CUDA build instructions."""
        QMessageBox.information(
            self,
            "Build CUDA Backend",
            "To build the CUDA backend:\n\n"
            "1. Ensure CUDA toolkit is installed (nvcc --version)\n"
            "2. Run: bash build_cuda.sh\n"
            "3. Restart the application\n\n"
            "For more details, see CUDA_SETUP.md"
        )
    
    def refresh(self):
        """Refresh all settings."""
        self._refresh_model_list()
        self._update_disk_usage()
        self._refresh_cuda_status()
        self._populate_microphone_list()
