"""Audio Converter Tab for CripIt.

Provides UI for converting audio files (m4a, mp3, ogg, etc.) to WAV format
and managing converted files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QGroupBox, QAbstractItemView, QMenu,
    QProgressBar
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction

from utils.audio_converter import (
    convert_to_wav,
    get_supported_formats,
    get_converted_files,
    delete_converted_file,
    clear_all_converted_files,
    get_total_converted_size,
    format_file_size,
    format_duration,
    read_wav_duration,
    check_ffmpeg_available,
)

logger = logging.getLogger(__name__)


class ConverterTab(QWidget):
    """Tab for audio file conversion and management."""

    def __init__(
        self,
        converted_dir: Path,
        pipeline,
        on_transcription_ready: Optional[Callable[[str], None]] = None,
        parent=None,
    ):
        super().__init__(parent)

        self.converted_dir = Path(converted_dir)
        self.pipeline = pipeline
        self.on_transcription_ready = on_transcription_ready

        # Ensure converted directory exists
        self.converted_dir.mkdir(parents=True, exist_ok=True)

        # Track if FFmpeg is available
        self.ffmpeg_available = check_ffmpeg_available()

        self._setup_ui()
        self._refresh_file_list()

        # Timer to refresh file list periodically
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_file_list)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds

        logger.info(f"ConverterTab initialized (FFmpeg available: {self.ffmpeg_available})")

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # === FFmpeg Warning ===
        if not self.ffmpeg_available:
            self.ffmpeg_warning = QLabel(
                "⚠️ FFmpeg not found. Audio conversion requires FFmpeg.\n"
                "Install: Ubuntu/Debian: sudo apt-get install ffmpeg | macOS: brew install ffmpeg"
            )
            self.ffmpeg_warning.setStyleSheet("color: orange; font-weight: bold;")
            layout.addWidget(self.ffmpeg_warning)

        # === File Conversion Section ===
        convert_group = QGroupBox("Convert Audio File")
        convert_layout = QVBoxLayout(convert_group)

        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: gray;")
        file_layout.addWidget(self.file_label, stretch=1)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(self.browse_btn)

        convert_layout.addLayout(file_layout)

        # Convert button
        self.convert_btn = QPushButton("Convert to WAV")
        self.convert_btn.setEnabled(False)
        self.convert_btn.clicked.connect(self._convert_file)
        convert_layout.addWidget(self.convert_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        convert_layout.addWidget(self.progress_bar)

        layout.addWidget(convert_group)

        # === Converted Files Section ===
        files_group = QGroupBox("Converted Files")
        files_layout = QVBoxLayout(files_group)

        # File table
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(5)
        self.file_table.setHorizontalHeaderLabels(
            ["Filename", "Date", "Size", "Duration", "Actions"]
        )
        self.file_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.file_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.file_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.file_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.file_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self.file_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.file_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_table.customContextMenuRequested.connect(self._show_context_menu)
        files_layout.addWidget(self.file_table)

        # Storage info and clear button
        info_layout = QHBoxLayout()
        self.storage_label = QLabel("Total size: 0 B")
        info_layout.addWidget(self.storage_label)
        info_layout.addStretch()

        self.clear_all_btn = QPushButton("Clear All Files")
        self.clear_all_btn.clicked.connect(self._clear_all_files)
        info_layout.addWidget(self.clear_all_btn)

        files_layout.addLayout(info_layout)

        layout.addWidget(files_group, stretch=1)

        # Store selected file path
        self.selected_file: Optional[Path] = None

    def _browse_file(self):
        """Open file dialog to select audio file."""
        filters = " ".join(f"*{ext}" for ext in get_supported_formats())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            f"Audio Files ({filters});;All Files (*.*)",
        )

        if file_path:
            self.selected_file = Path(file_path)
            self.file_label.setText(self.selected_file.name)
            self.file_label.setStyleSheet("color: black;")
            self.convert_btn.setEnabled(True)
            logger.info(f"Selected file: {file_path}")

    def _convert_file(self):
        """Convert the selected file to WAV."""
        if not self.selected_file:
            return

        if not self.ffmpeg_available:
            QMessageBox.warning(
                self,
                "FFmpeg Required",
                "FFmpeg is required for audio conversion.\n\n"
                "Install FFmpeg:\n"
                "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                "  macOS: brew install ffmpeg\n"
                "  Windows: Download from https://ffmpeg.org/download.html",
            )
            return

        self.convert_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        try:
            success, output_path, error_msg = convert_to_wav(
                self.selected_file, self.converted_dir
            )

            if success:
                QMessageBox.information(
                    self,
                    "Conversion Complete",
                    f"File converted successfully:\n{output_path.name}",
                )
                self._refresh_file_list()
                # Reset selection
                self.selected_file = None
                self.file_label.setText("No file selected")
                self.file_label.setStyleSheet("color: gray;")
                self.convert_btn.setEnabled(False)
            else:
                QMessageBox.critical(
                    self,
                    "Conversion Failed",
                    f"Failed to convert file:\n{error_msg}",
                )
                self.convert_btn.setEnabled(True)

        except Exception as e:
            logger.exception("Conversion failed")
            QMessageBox.critical(
                self,
                "Error",
                f"Unexpected error during conversion:\n{type(e).__name__}: {e}",
            )
            self.convert_btn.setEnabled(True)

        finally:
            self.browse_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    def _refresh_file_list(self):
        """Refresh the list of converted files."""
        files = get_converted_files(self.converted_dir)

        self.file_table.setRowCount(len(files))

        for row, (file_path, mtime, size) in enumerate(files):
            # Filename
            name_item = QTableWidgetItem(file_path.name)
            name_item.setData(Qt.ItemDataRole.UserRole, str(file_path))
            self.file_table.setItem(row, 0, name_item)

            # Date
            date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            self.file_table.setItem(row, 1, QTableWidgetItem(date_str))

            # Size
            size_str = format_file_size(size)
            self.file_table.setItem(row, 2, QTableWidgetItem(size_str))

            # Duration
            duration = read_wav_duration(file_path)
            duration_str = format_duration(duration) if duration > 0 else "-"
            self.file_table.setItem(row, 3, QTableWidgetItem(duration_str))

            # Actions (transcribe + delete buttons)
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(4, 2, 4, 2)
            actions_layout.setSpacing(4)

            transcribe_btn = QPushButton("Transcribe")
            transcribe_btn.setProperty("file_path", str(file_path))
            transcribe_btn.clicked.connect(self._transcribe_file)
            actions_layout.addWidget(transcribe_btn)

            delete_btn = QPushButton("Delete")
            delete_btn.setProperty("file_path", str(file_path))
            delete_btn.clicked.connect(self._delete_file)
            actions_layout.addWidget(delete_btn)

            actions_layout.addStretch()
            self.file_table.setCellWidget(row, 4, actions_widget)

        # Update storage label
        total_size = get_total_converted_size(self.converted_dir)
        self.storage_label.setText(f"Total size: {format_file_size(total_size)}")

    def _safe_update_button_text(self, row: int, text: str):
        """Safely update button text by row (handles widget recreation)."""
        if row < 0 or row >= self.file_table.rowCount():
            return

        widget = self.file_table.cellWidget(row, 4)
        if not widget:
            return

        buttons = widget.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() in ["Processing..."] or "Transcribing" in btn.text():
                btn.setText(text[:20] + "..." if len(text) > 20 else text)
                return

    def _safe_update_button_state(self, row: int):
        """Safely restore button state by row."""
        if row < 0 or row >= self.file_table.rowCount():
            return

        widget = self.file_table.cellWidget(row, 4)
        if not widget:
            return

        buttons = widget.findChildren(QPushButton)
        for btn in buttons:
            if btn.text() != "Delete":  # The transcribe button
                btn.setEnabled(True)
                btn.setText("Transcribe")
                return

    def _transcribe_file(self):
        """Transcribe the selected converted file."""
        sender = self.sender()
        if not sender:
            return

        file_path = Path(sender.property("file_path"))

        if not file_path.exists():
            QMessageBox.warning(self, "Error", "File not found")
            return

        # Check if pipeline is available
        if not self.pipeline:
            QMessageBox.warning(
                self,
                "Not Available",
                "Transcription pipeline is not available",
            )
            return

        # Stop auto-refresh timer to prevent button destruction during transcription
        self.refresh_timer.stop()

        # Store button reference and row for safe updates
        button_ref = sender
        button_row = -1

        # Find which row this button is in
        for row in range(self.file_table.rowCount()):
            widget = self.file_table.cellWidget(row, 4)
            if widget:
                transcribe_btn = widget.findChild(QPushButton, "", Qt.FindChildOption.FindChildrenRecursively)
                if transcribe_btn and transcribe_btn == button_ref:
                    button_row = row
                    break

        # Disable button during transcription
        button_ref.setEnabled(False)
        button_ref.setText("Processing...")

        try:
            success, text, error_msg = self.pipeline.transcribe_single_file(
                file_path,
                progress_callback=lambda msg: self._safe_update_button_text(button_row, msg),
            )

            if success and text:
                if self.on_transcription_ready:
                    self.on_transcription_ready(text)
                QMessageBox.information(
                    self,
                    "Transcription Complete",
                    f"Transcription:\n\n{text[:500]}{'...' if len(text) > 500 else ''}",
                )
            elif success:
                QMessageBox.information(
                    self,
                    "Transcription Complete",
                    "Transcription completed but no text was produced",
                )
            else:
                QMessageBox.critical(
                    self,
                    "Transcription Failed",
                    f"Failed to transcribe:\n{error_msg}",
                )

        except Exception as e:
            logger.exception("Transcription failed")
            QMessageBox.critical(
                self,
                "Error",
                f"Unexpected error during transcription:\n{type(e).__name__}: {e}",
            )

        finally:
            # Restore button state safely
            self._safe_update_button_state(button_row)
            # Restart refresh timer
            self.refresh_timer.start(5000)

    def _delete_file(self):
        """Delete the selected converted file."""
        sender = self.sender()
        if not sender:
            return

        file_path = Path(sender.property("file_path"))

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete file '{file_path.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            if delete_converted_file(file_path):
                self._refresh_file_list()
            else:
                QMessageBox.warning(self, "Error", "Failed to delete file")

    def _clear_all_files(self):
        """Clear all converted files."""
        files = get_converted_files(self.converted_dir)

        if not files:
            QMessageBox.information(self, "Info", "No converted files to delete")
            return

        total_size = sum(f[2] for f in files)

        reply = QMessageBox.question(
            self,
            "Confirm Clear All",
            f"Delete all {len(files)} converted files?\n"
            f"Total size: {format_file_size(total_size)}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            deleted, failed = clear_all_converted_files(self.converted_dir)
            self._refresh_file_list()

            if failed > 0:
                QMessageBox.warning(
                    self,
                    "Partial Success",
                    f"Deleted {deleted} files, {failed} failed",
                )
            else:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Deleted {deleted} files",
                )

    def _show_context_menu(self, position):
        """Show context menu for file table."""
        row = self.file_table.rowAt(position.y())
        if row < 0:
            return

        item = self.file_table.item(row, 0)
        if not item:
            return

        file_path = Path(item.data(Qt.ItemDataRole.UserRole))

        menu = QMenu(self)

        transcribe_action = QAction("Transcribe", self)
        transcribe_action.triggered.connect(
            lambda: self._transcribe_file_from_path(file_path)
        )
        menu.addAction(transcribe_action)

        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(
            lambda: self._delete_file_from_path(file_path)
        )
        menu.addAction(delete_action)

        menu.exec(self.file_table.viewport().mapToGlobal(position))

    def _transcribe_file_from_path(self, file_path: Path):
        """Transcribe file from path (for context menu)."""
        # Find the button for this file and simulate click
        for row in range(self.file_table.rowCount()):
            item = self.file_table.item(row, 0)
            if item and Path(item.data(Qt.ItemDataRole.UserRole)) == file_path:
                actions_widget = self.file_table.cellWidget(row, 4)
                if actions_widget:
                    transcribe_btn = actions_widget.findChild(QPushButton, "", Qt.FindChildOption.FindChildrenRecursively)
                    if transcribe_btn:
                        transcribe_btn.click()
                break

    def _delete_file_from_path(self, file_path: Path):
        """Delete file from path (for context menu)."""
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete file '{file_path.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            if delete_converted_file(file_path):
                self._refresh_file_list()
            else:
                QMessageBox.warning(self, "Error", "Failed to delete file")

    def refresh(self):
        """Public method to refresh the file list."""
        self._refresh_file_list()
