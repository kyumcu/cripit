"""
Test suite for configuration module
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path

# Setup logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_config_singleton():
    """Test that AppConfig is a singleton."""
    print("\n" + "="*50)
    print("TEST 1: Configuration Singleton")
    print("="*50)
    
    from config.settings import AppConfig, config
    
    # Get two instances
    config1 = AppConfig()
    config2 = AppConfig()
    
    # They should be the same object
    assert config1 is config2, "AppConfig should be a singleton!"
    assert config1 is config, "Global config should be same instance!"
    
    print("✓ Singleton pattern working correctly")
    print(f"✓ Config instance: {config1}")
    return True

def test_audio_settings():
    """Test audio settings initialization."""
    print("\n" + "="*50)
    print("TEST 2: Audio Settings")
    print("="*50)
    
    from config.settings import AudioSettings
    
    audio = AudioSettings()
    
    # Check default values
    assert audio.sample_rate == 16000, f"Expected 16000, got {audio.sample_rate}"
    assert audio.channels == 1, f"Expected 1, got {audio.channels}"
    assert audio.vad_enabled == True, f"Expected True, got {audio.vad_enabled}"
    
    print(f"✓ Sample rate: {audio.sample_rate} Hz")
    print(f"✓ Channels: {audio.channels}")
    print(f"✓ VAD enabled: {audio.vad_enabled}")
    print(f"✓ VAD aggressiveness: {audio.vad_aggressiveness}")
    return True

def test_model_settings():
    """Test model settings initialization."""
    print("\n" + "="*50)
    print("TEST 3: Model Settings")
    print("="*50)
    
    from config.settings import ModelSettings
    
    model = ModelSettings()
    
    # Check default model
    assert model.default_model == "large-v3-turbo", f"Expected large-v3-turbo, got {model.default_model}"
    assert "large-v3-turbo" in model.available_models, "large-v3-turbo should be in available models"
    
    # Check model info
    turbo_info = model.available_models["large-v3-turbo"]
    print(f"✓ Default model: {model.default_model}")
    print(f"✓ Parameters: {turbo_info['params']}")
    print(f"✓ File: {turbo_info['file']}")
    print(f"✓ Size: {turbo_info['size_mb']} MB")
    print(f"✓ Available models: {list(model.available_models.keys())}")

    # Check CUDA defaults
    assert hasattr(model, "use_cuda"), "ModelSettings should include use_cuda"
    assert hasattr(model, "cuda_device"), "ModelSettings should include cuda_device"
    assert hasattr(model, "cuda_fallback_to_cpu"), "ModelSettings should include cuda_fallback_to_cpu"
    assert hasattr(model, "cuda_warn_on_fallback"), "ModelSettings should include cuda_warn_on_fallback"

    print(f"✓ CUDA enabled by default: {model.use_cuda}")
    print(f"✓ CUDA device: {model.cuda_device}")
    print(f"✓ CUDA fallback to CPU: {model.cuda_fallback_to_cpu}")
    print(f"✓ CUDA warn on fallback: {model.cuda_warn_on_fallback}")
    return True

def test_ui_settings():
    """Test UI settings initialization."""
    print("\n" + "="*50)
    print("TEST 4: UI Settings")
    print("="*50)
    
    from config.settings import UISettings
    
    ui = UISettings()
    
    assert ui.window_width == 800, f"Expected 800, got {ui.window_width}"
    assert ui.window_height == 600, f"Expected 600, got {ui.window_height}"
    
    print(f"✓ Window size: {ui.window_width}x{ui.window_height}")
    print(f"✓ Title: {ui.window_title}")
    print(f"✓ Auto copy: {ui.auto_copy}")
    print(f"✓ Minimize to tray: {ui.minimize_to_tray}")
    return True

def test_model_paths():
    """Test model path resolution."""
    print("\n" + "="*50)
    print("TEST 5: Model Path Resolution")
    print("="*50)
    
    from config.settings import config, MODELS_DIR
    
    # Test default model path
    default_path = config.get_model_path()
    print(f"✓ Default model path: {default_path}")
    assert default_path.parent == MODELS_DIR, f"Path should be in models dir"
    assert "large-v3-turbo" in str(default_path), "Should reference large-v3-turbo"
    
    # Test specific model paths
    for model_name in ["tiny", "base", "small"]:
        path = config.get_model_path(model_name)
        print(f"✓ {model_name} path: {path}")
        assert path.parent == MODELS_DIR
    
    # Test unknown model falls back to default
    unknown_path = config.get_model_path("nonexistent")
    assert "large-v3-turbo" in str(unknown_path), "Unknown model should fallback to default"
    print("✓ Unknown model correctly falls back to default")
    
    return True

def test_config_save_load():
    """Test configuration save and load."""
    print("\n" + "="*50)
    print("TEST 6: Config Save/Load")
    print("="*50)
    
    from config.settings import AppConfig
    
    # Create a fresh config instance
    config = AppConfig()
    
    # Modify some settings
    original_lang = config.model.language
    original_use_cuda = getattr(config.model, "use_cuda", True)
    config.model.language = "en"
    config.audio.vad_enabled = False
    config.model.use_cuda = True
    config.model.cuda_device = 0
    config.model.cuda_fallback_to_cpu = True
    config.model.cuda_warn_on_fallback = True
    
    # Save
    result = config.save_config()
    assert result, "Config save should return True"
    print("✓ Configuration saved")
    
    # Check file exists
    config_file = config._config_file
    assert config_file.exists(), "Config file should exist after save"
    print(f"✓ Config file created: {config_file}")
    
    # Reset and reload (simulate new session)
    config._initialized = False
    config.__init__()
    
    print(f"✓ Language after reload: {config.model.language}")
    print(f"✓ VAD enabled after reload: {config.audio.vad_enabled}")
    print(f"✓ use_cuda after reload: {config.model.use_cuda}")
    print(f"✓ cuda_device after reload: {config.model.cuda_device}")
    
    # Restore original
    config.model.language = original_lang
    config.model.use_cuda = original_use_cuda
    config.audio.vad_enabled = True
    config.save_config()
    print("✓ Configuration restored and saved")
    
    return True

def run_all_tests():
    """Run all configuration tests."""
    print("\n" + "="*60)
    print("CONFIGURATION MODULE TEST SUITE")
    print("="*60)
    
    tests = [
        test_config_singleton,
        test_audio_settings,
        test_model_settings,
        test_ui_settings,
        test_model_paths,
        test_config_save_load,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"\n✅ {test.__name__} PASSED")
            else:
                failed += 1
                print(f"\n❌ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test.__name__} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
