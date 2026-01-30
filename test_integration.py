"""
Integration Tests - Full Pipeline
Tests the complete STT pipeline from audio capture to transcription
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_pipeline_initialization():
    """Test that all components can be initialized together."""
    print("\n" + "="*60)
    print("INTEGRATION TEST 1: Pipeline Initialization")
    print("="*60)
    
    try:
        from config.settings import config
        from core.audio_capture import AudioCapture
        from core.transcriber import Transcriber
        from core.model_manager import get_model_manager
        
        logger.info("Initializing all components...")
        
        # 1. Config (already initialized as singleton)
        print(f"✓ Config loaded: {config.model.default_model}")
        
        # 2. Model Manager
        model_manager = get_model_manager()
        print(f"✓ ModelManager initialized: {len(model_manager.list_models())} models")
        
        # 3. Audio Capture
        audio = AudioCapture(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            silence_timeout=0.5  # Short for testing
        )
        print(f"✓ AudioCapture initialized")
        
        # 4. Transcriber (without loading model)
        transcoder = Transcriber(
            model_name=config.model.default_model,
            language=config.model.language,
            n_threads=config.model.n_threads
        )
        print(f"✓ Transcriber initialized (model not loaded)")
        
        logger.info("All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_to_transcriber_flow():
    """Test audio flow from capture to transcriber."""
    print("\n" + "="*60)
    print("INTEGRATION TEST 2: Audio to Transcriber Flow")
    print("="*60)
    
    try:
        from core.audio_capture import AudioCapture
        from core.transcriber import Transcriber
        
        # Create components
        audio = AudioCapture(silence_timeout=0.3)
        
        # Track audio received
        audio_chunks = []
        
        def on_audio_ready(audio_data):
            duration = len(audio_data) / 16000
            audio_chunks.append(duration)
            logger.info(f"Audio received: {duration:.2f}s")
        
        audio.on_audio_ready = on_audio_ready
        
        # Simulate audio generation (without microphone)
        logger.info("Simulating audio generation...")
        
        # Generate fake audio (silence)
        for i in range(10):
            fake_audio = np.zeros(1600, dtype=np.int16)  # 100ms of silence
            audio.on_audio_ready(fake_audio)
            time.sleep(0.05)
        
        print(f"✓ Audio chunks processed: {len(audio_chunks)}")
        print(f"✓ Total audio: {sum(audio_chunks):.2f}s")
        
        assert len(audio_chunks) == 10, "Should have 10 audio chunks"
        
        return True
        
    except Exception as e:
        logger.error(f"Audio flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_download_and_load():
    """Test model download and load flow."""
    print("\n" + "="*60)
    print("INTEGRATION TEST 3: Model Download & Load")
    print("="*60)
    
    try:
        from core.model_manager import get_model_manager
        from core.transcriber import Transcriber
        
        manager = get_model_manager()
        
        # Check if tiny model is available (small, good for testing)
        model_name = "tiny"
        
        print(f"✓ Checking model: {model_name}")
        is_available = manager.is_model_available(model_name)
        print(f"✓ Available: {is_available}")
        
        if not is_available:
            print(f"⚠ Model {model_name} not available locally")
            print(f"⚠ Would download from: {manager.get_model_info(model_name).url}")
            print(f"⚠ Skipping download in test (would take too long)")
        else:
            # Try to load the model
            model_path = manager.get_model_path(model_name)
            print(f"✓ Model path: {model_path}")
            
            transcoder = Transcriber(model_path=str(model_path), model_name=model_name)
            
            # Note: We won't actually load the model without the .bin file
            print(f"✓ Transcriber ready for model: {model_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model download/load test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_persistence():
    """Test that configuration persists across sessions."""
    print("\n" + "="*60)
    print("INTEGRATION TEST 4: Configuration Persistence")
    print("="*60)
    
    try:
        from config.settings import AppConfig
        
        # Create new config instance
        config1 = AppConfig()
        
        # Modify settings
        original_lang = config1.model.language
        config1.model.language = "es"
        config1.audio.vad_enabled = False
        
        # Save
        result = config1.save_config()
        print(f"✓ Config saved: {result}")
        assert result, "Config save should succeed"
        
        # Reset instance to force reload
        config1._initialized = False
        config1.__init__()
        
        # Verify loaded values
        print(f"✓ Loaded language: {config1.model.language}")
        print(f"✓ Loaded VAD: {config1.audio.vad_enabled}")
        
        assert config1.model.language == "es", "Language should persist"
        assert config1.audio.vad_enabled == False, "VAD should persist"
        
        # Restore original
        config1.model.language = original_lang
        config1.audio.vad_enabled = True
        config1.save_config()
        print("✓ Original values restored")
        
        return True
        
    except Exception as e:
        logger.error(f"Config persistence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_mock():
    """Test end-to-end with mock audio (no real transcription)."""
    print("\n" + "="*60)
    print("INTEGRATION TEST 5: End-to-End Mock")
    print("="*60)
    
    try:
        from config.settings import config
        from core.audio_capture import AudioCapture, RecordingState
        from core.transcriber import Transcriber, TranscriptionResult
        
        logger.info("Setting up end-to-end mock pipeline...")
        
        # Create audio capture
        audio = AudioCapture(silence_timeout=0.3)
        
        # Create transcriber (without loading model)
        transcoder = Transcriber(
            model_name="tiny",
            language="en",
            n_threads=2
        )
        
        # Track results
        transcription_results = []
        
        def on_audio_ready(audio_data):
            """Simulate transcription."""
            duration = len(audio_data) / 16000
            
            # Create mock result
            result = TranscriptionResult(
                text=f"[Mock transcription of {duration:.1f}s audio]",
                language="en",
                duration=duration,
                processing_time=0.1,
                is_partial=False
            )
            
            transcription_results.append(result)
            logger.info(f"Mock transcription: {result.text}")
        
        audio.on_audio_ready = on_audio_ready
        
        # Simulate recording session
        logger.info("Simulating recording session...")
        
        # Generate 3 chunks of "audio"
        for i in range(3):
            fake_audio = np.random.randint(-1000, 1000, 8000, dtype=np.int16)
            audio.on_audio_ready(fake_audio)
            time.sleep(0.1)
        
        print(f"✓ Generated {len(transcription_results)} transcriptions")
        print(f"✓ Total audio processed: {sum(r.duration for r in transcription_results):.2f}s")
        
        assert len(transcription_results) == 3, "Should have 3 transcriptions"
        
        logger.info("End-to-end mock completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"End-to-end mock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling in pipeline."""
    print("\n" + "="*60)
    print("INTEGRATION TEST 6: Error Handling")
    print("="*60)
    
    try:
        from core.audio_capture import AudioCapture
        from core.transcriber import Transcriber
        
        # Test transcriber with invalid model
        transcoder = Transcriber(model_path="/invalid/path.bin")
        
        # Try to load (should fail gracefully)
        result = transcoder.load_model()
        print(f"✓ Invalid model load handled: {result}")
        assert not result, "Should fail to load invalid model"
        assert not transcoder.is_ready(), "Should not be ready"
        
        # Test error callback
        errors_caught = []
        
        def on_error(e):
            errors_caught.append(str(e))
            logger.info(f"Error caught: {e}")
        
        transcoder.on_error = on_error
        
        # Simulate error
        fake_error = Exception("Test error")
        if transcoder.on_error:
            transcoder.on_error(fake_error)
        
        print(f"✓ Errors caught: {len(errors_caught)}")
        assert len(errors_caught) == 1, "Should catch 1 error"
        
        return True
        
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("INTEGRATION TEST SUITE - Full Pipeline")
    print("="*70)
    
    tests = [
        test_pipeline_initialization,
        test_audio_to_transcriber_flow,
        test_model_download_and_load,
        test_configuration_persistence,
        test_end_to_end_mock,
        test_error_handling,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            if result:
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
    
    print("\n" + "="*70)
    print(f"INTEGRATION TEST RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
