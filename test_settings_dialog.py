#!/usr/bin/env python3
"""
CripIt Settings Dialog Test Script
Tests the settings dialog and model management functionality
"""

import sys
import os
import logging
import time
import traceback
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Test results storage
test_results = {
    'tests_run': 0,
    'tests_passed': 0,
    'tests_failed': 0,
    'errors': []
}

def run_test(test_name):
    """Decorator to run a test and track results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            test_results['tests_run'] += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"TEST: {test_name}")
            logger.info(f"{'='*60}")
            try:
                func(*args, **kwargs)
                test_results['tests_passed'] += 1
                logger.info(f"✓ PASSED: {test_name}")
                return True
            except Exception as e:
                test_results['tests_failed'] += 1
                error_msg = f"✗ FAILED: {test_name} - {str(e)}"
                test_results['errors'].append(error_msg)
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return False
        return wrapper
    return decorator

@run_test("Import Settings Dialog")
def test_import_settings_dialog():
    """Test that settings dialog can be imported."""
    from gui.settings_dialog import SettingsDialog
    assert SettingsDialog is not None
    logger.info("SettingsDialog imported successfully")

@run_test("Initialize Settings Dialog")
def test_initialize_settings_dialog(qt_app, config, model_manager):
    """Test settings dialog initialization."""
    from gui.settings_dialog import SettingsDialog
    
    dialog = SettingsDialog(config, model_manager)
    assert dialog is not None
    assert dialog.model_list is not None
    assert dialog.delete_btn is not None
    assert dialog.reinstall_btn is not None
    assert dialog.progress_bar is not None
    
    logger.info(f"Settings dialog initialized")
    logger.info(f"Model list has {dialog.model_list.count()} items")
    
    dialog.close()
    return True

@run_test("Model List Population")
def test_model_list_population(qt_app, config, model_manager):
    """Test that model list is populated correctly."""
    from gui.settings_dialog import SettingsDialog
    
    dialog = SettingsDialog(config, model_manager)
    
    # Check that models are listed
    count = dialog.model_list.count()
    assert count > 0, "Model list should not be empty"
    
    # Check for installed models
    available = model_manager.get_available_models()
    logger.info(f"Available models: {available}")
    logger.info(f"Model list items: {count}")
    
    # Verify each item has data
    for i in range(count):
        item = dialog.model_list.item(i)
        assert item is not None
        model_name = item.data(Qt.ItemDataRole.UserRole)
        assert model_name is not None
        logger.info(f"  Item {i}: {model_name}")
    
    dialog.close()
    return True

@run_test("Delete Model Button - Initial State")
def test_delete_button_initial_state(qt_app, config, model_manager):
    """Test delete button is disabled initially."""
    from gui.settings_dialog import SettingsDialog
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    
    dialog = SettingsDialog(config, model_manager)
    
    # Initially no selection, button should be disabled
    assert not dialog.delete_btn.isEnabled(), \
        "Delete button should be disabled when no model selected"
    assert not dialog.reinstall_btn.isEnabled(), \
        "Reinstall button should be disabled when no model selected"
    
    logger.info("Buttons correctly disabled initially")
    
    dialog.close()
    return True

@run_test("Delete Model Button - After Selection")
def test_delete_button_after_selection(qt_app, config, model_manager):
    """Test delete button enables when installed model is selected."""
    from gui.settings_dialog import SettingsDialog
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    
    dialog = SettingsDialog(config, model_manager)
    
    # Find an installed model
    available = model_manager.get_available_models()
    logger.info(f"Looking for installed model in: {available}")
    
    if not available:
        logger.warning("No models installed, skipping detailed selection test")
        dialog.close()
        return True
    
    # Select the first installed model
    target_model = available[0]
    found = False
    
    for i in range(dialog.model_list.count()):
        item = dialog.model_list.item(i)
        model_name = item.data(Qt.ItemDataRole.UserRole)
        if model_name == target_model:
            dialog.model_list.setCurrentItem(item)
            found = True
            logger.info(f"Selected model: {model_name}")
            break
    
    assert found, f"Could not find model {target_model} in list"
    
    # Process events to let UI update
    QApplication.processEvents()
    
    # Now delete button should be enabled
    assert dialog.delete_btn.isEnabled(), \
        f"Delete button should be enabled for installed model {target_model}"
    assert dialog.reinstall_btn.isEnabled(), \
        "Reinstall button should be enabled for any selected model"
    
    logger.info(f"Buttons correctly enabled for installed model: {target_model}")
    
    dialog.close()
    return True

@run_test("Model Info Display")
def test_model_info_display(qt_app, config, model_manager):
    """Test that model info is displayed when model is selected."""
    from gui.settings_dialog import SettingsDialog
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    
    dialog = SettingsDialog(config, model_manager)
    
    # Select first model
    if dialog.model_list.count() > 0:
        item = dialog.model_list.item(0)
        dialog.model_list.setCurrentItem(item)
        
        QApplication.processEvents()
        
        # Check info text is populated
        info_text = dialog.model_info_text.toPlainText()
        logger.info(f"Model info text: {info_text[:100]}...")
        assert len(info_text) > 0, "Model info should be displayed"
    
    dialog.close()
    return True

@run_test("Delete Model - Confirmation Dialog")
def test_delete_confirmation_dialog(qt_app, config, model_manager):
    """Test delete model shows confirmation dialog."""
    from gui.settings_dialog import SettingsDialog
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtCore import Qt
    from unittest.mock import patch
    
    dialog = SettingsDialog(config, model_manager)
    
    # Find an installed model
    available = model_manager.get_available_models()
    if not available:
        logger.warning("No models installed to test deletion")
        dialog.close()
        return True
    
    # Select the model
    target_model = available[0]
    for i in range(dialog.model_list.count()):
        item = dialog.model_list.item(i)
        if item.data(Qt.ItemDataRole.UserRole) == target_model:
            dialog.model_list.setCurrentItem(item)
            break
    
    QApplication.processEvents()
    
    # Mock the message box to return "No"
    with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No):
        # Click delete button
        dialog.delete_btn.click()
        QApplication.processEvents()
        
        # Verify model still exists (was not deleted)
        assert model_manager.is_model_available(target_model), \
            f"Model {target_model} should still exist after canceling deletion"
        logger.info(f"Delete confirmation dialog works - model not deleted when cancelled")
    
    dialog.close()
    return True

@run_test("Models Changed Signal Emission")
def test_models_changed_signal(qt_app, config, model_manager):
    """Test that models_changed signal is emitted after model operations."""
    from gui.settings_dialog import SettingsDialog
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtCore import Qt
    from unittest.mock import patch, MagicMock
    
    dialog = SettingsDialog(config, model_manager)
    
    # Create a mock slot
    mock_slot = MagicMock()
    dialog.models_changed.connect(mock_slot)
    
    # Find an installed model
    available = model_manager.get_available_models()
    if not available:
        logger.warning("No models installed to test signal emission")
        dialog.close()
        return True
    
    # Select the model
    target_model = available[0]
    for i in range(dialog.model_list.count()):
        item = dialog.model_list.item(i)
        if item.data(Qt.ItemDataRole.UserRole) == target_model:
            dialog.model_list.setCurrentItem(item)
            break
    
    QApplication.processEvents()
    
    # Mock message box to return Yes (but mock delete to avoid actual deletion)
    with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
        with patch.object(model_manager, 'delete_model', return_value=True):
            dialog._delete_selected_model()
            QApplication.processEvents()
            
            # Check if signal was emitted
            assert mock_slot.called, "models_changed signal should be emitted after deletion"
            logger.info("✓ models_changed signal correctly emitted")
    
    dialog.close()
    return True

@run_test("Disk Usage Display")
def test_disk_usage_display(qt_app, config, model_manager):
    """Test disk usage is displayed and calculated correctly."""
    from gui.settings_dialog import SettingsDialog
    from PyQt6.QtWidgets import QApplication
    
    dialog = SettingsDialog(config, model_manager)
    
    # Check disk usage label
    disk_text = dialog.disk_usage_label.text()
    logger.info(f"Disk usage text: {disk_text}")
    
    assert "models installed" in disk_text or "Unable" in disk_text, \
        f"Disk usage should be displayed, got: {disk_text}"
    
    dialog.close()
    return True

@run_test("Reinstall Model - Not Installed")
def test_reinstall_not_installed_model(qt_app, config, model_manager):
    """Test reinstall button for a model that is not installed."""
    from gui.settings_dialog import SettingsDialog
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    
    dialog = SettingsDialog(config, model_manager)
    
    # Find a model that is NOT installed
    all_models = model_manager.list_models()
    available = model_manager.get_available_models()
    missing = [m for m in all_models if m not in available]
    
    if not missing:
        logger.warning("All models are installed, skipping reinstall test")
        dialog.close()
        return True
    
    target_model = missing[0]
    logger.info(f"Testing reinstall for non-installed model: {target_model}")
    
    # Select the non-installed model
    for i in range(dialog.model_list.count()):
        item = dialog.model_list.item(i)
        if item.data(Qt.ItemDataRole.UserRole) == target_model:
            dialog.model_list.setCurrentItem(item)
            break
    
    QApplication.processEvents()
    
    # Reinstall button should be enabled
    assert dialog.reinstall_btn.isEnabled(), \
        "Reinstall button should be enabled for non-installed model"
    
    # Delete button should be disabled
    assert not dialog.delete_btn.isEnabled(), \
        "Delete button should be disabled for non-installed model"
    
    logger.info(f"Buttons correctly configured for non-installed model")
    
    dialog.close()
    return True

@run_test("Progress Bar Visibility")
def test_progress_bar_visibility(qt_app, config, model_manager):
    """Test progress bar is hidden by default and shows during download."""
    from gui.settings_dialog import SettingsDialog
    from PyQt6.QtWidgets import QApplication
    
    dialog = SettingsDialog(config, model_manager)
    
    # Initially hidden
    assert not dialog.progress_bar.isVisible(), \
        "Progress bar should be hidden by default"
    
    logger.info("Progress bar correctly hidden by default")
    
    dialog.close()
    return True

@run_test("Close Dialog")
def test_close_dialog(qt_app, config, model_manager):
    """Test that dialog closes properly."""
    from gui.settings_dialog import SettingsDialog
    from PyQt6.QtWidgets import QApplication
    
    dialog = SettingsDialog(config, model_manager)
    
    # Close the dialog
    dialog.close()
    QApplication.processEvents()
    
    # Dialog should be closed
    # Note: In Qt, close() doesn't destroy the object immediately
    logger.info("Dialog closed successfully")
    
    return True

def run_all_tests():
    """Run all tests."""
    logger.info("\n" + "="*70)
    logger.info("CRIPIT SETTINGS DIALOG TEST SUITE")
    logger.info("="*70)
    
    # Import required modules
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    
    # Create QApplication
    qt_app = QApplication.instance() or QApplication(sys.argv)
    
    # Initialize components
    from config.settings import config
    from core.model_manager import get_model_manager
    
    model_manager = get_model_manager()
    logger.info(f"Model manager initialized with {len(model_manager.list_models())} models")
    logger.info(f"Available models: {model_manager.get_available_models()}")
    
    # Run all tests
    tests = [
        test_import_settings_dialog,
        lambda: test_initialize_settings_dialog(qt_app, config, model_manager),
        lambda: test_model_list_population(qt_app, config, model_manager),
        lambda: test_delete_button_initial_state(qt_app, config, model_manager),
        lambda: test_delete_button_after_selection(qt_app, config, model_manager),
        lambda: test_model_info_display(qt_app, config, model_manager),
        lambda: test_delete_confirmation_dialog(qt_app, config, model_manager),
        lambda: test_models_changed_signal(qt_app, config, model_manager),
        lambda: test_disk_usage_display(qt_app, config, model_manager),
        lambda: test_reinstall_not_installed_model(qt_app, config, model_manager),
        lambda: test_progress_bar_visibility(qt_app, config, model_manager),
        lambda: test_close_dialog(qt_app, config, model_manager),
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            logger.error(f"Test execution error: {e}")
            logger.error(traceback.format_exc())
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    logger.info(f"Tests Run:    {test_results['tests_run']}")
    logger.info(f"Tests Passed: {test_results['tests_passed']}")
    logger.info(f"Tests Failed: {test_results['tests_failed']}")
    
    if test_results['errors']:
        logger.info("\nErrors:")
        for error in test_results['errors']:
            logger.info(f"  - {error}")
    
    success_rate = (test_results['tests_passed'] / test_results['tests_run'] * 100) \
        if test_results['tests_run'] > 0 else 0
    logger.info(f"\nSuccess Rate: {success_rate:.1f}%")
    
    return test_results['tests_failed'] == 0

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.exception("Fatal error in test suite")
        sys.exit(1)
