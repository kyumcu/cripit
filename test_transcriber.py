"""
Test suite for transcriber module
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np

# Setup logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*50)
    print("TEST 1: Module Imports")
    print("="*50)
    
    try:
        from core.transcriber import Transcriber, TranscriptionResult
        from core.transcriber import MultiModelTranscriber, create_transcriber
        print("✓ All transcriber classes imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_transcription_result():
    """Test TranscriptionResult dataclass."""
    print("\n" + "="*50)
    print("TEST 2: TranscriptionResult Dataclass")
    print("="*50)
    
    from core.transcriber import TranscriptionResult
    
    # Create result
    result = TranscriptionResult(
        text="Hello world",
        language="en",
        confidence=0.95,
        duration=2.5,
        processing_time=0.3,
        segments=[{'text': 'Hello world', 'language': 'en'}],
        is_partial=False
    )
    
    print(f"✓ Text: {result.text}")
    print(f"✓ Language: {result.language}")
    print(f"✓ Duration: {result.duration}s")
    print(f"✓ Processing time: {result.processing_time}s")
    print(f"✓ RTF: {result.processing_time / result.duration:.2f}x")
    
    assert result.text == "Hello world"
    assert result.language == "en"
    assert result.duration == 2.5
    
    return True


def test_transcriber_initialization():
    """Test Transcriber initialization."""
    print("\n" + "="*50)
    print("TEST 3: Transcriber Initialization")
    print("="*50)
    
    from core.transcriber import Transcriber
    
    try:
        transcoder = Transcriber(
            model_path="/fake/path/model.bin",
            model_name="large-v3-turbo",
            language="en",
            n_threads=4,
            translate=False
        )
        
        print(f"✓ Transcriber initialized")
        print(f"✓ Model name: {transcoder.model_name}")
        print(f"✓ Language: {transcoder.language}")
        print(f"✓ Threads: {transcoder.n_threads}")
        print(f"✓ Ready: {transcoder.is_ready()}")
        
        assert transcoder.model_name == "large-v3-turbo"
        assert transcoder.language == "en"
        assert transcoder.n_threads == 4
        assert not transcoder.is_ready()  # Not loaded yet
        
        return True
    except Exception as e:
        print(f"❌ Transcriber initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transcriber_stats():
    """Test transcriber statistics."""
    print("\n" + "="*50)
    print("TEST 4: Transcriber Statistics")
    print("="*50)
    
    from core.transcriber import Transcriber
    
    try:
        transcoder = Transcriber(model_name="large-v3-turbo")
        
        # Get initial stats
        stats = transcoder.get_stats()
        
        print(f"✓ Initial stats:")
        print(f"  - Total transcriptions: {stats['total_transcriptions']}")
        print(f"  - Total audio seconds: {stats['total_audio_seconds']:.2f}s")
        print(f"  - Total processing seconds: {stats['total_processing_seconds']:.2f}s")
        print(f"  - Model loaded: {stats['model_loaded']}")
        print(f"  - Model name: {stats['model_name']}")
        
        assert stats['total_transcriptions'] == 0
        assert stats['model_loaded'] == False
        assert stats['model_name'] == "large-v3-turbo"
        
        return True
    except Exception as e:
        print(f"❌ Stats test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_model_transcriber():
    """Test MultiModelTranscriber."""
    print("\n" + "="*50)
    print("TEST 5: MultiModelTranscriber")
    print("="*50)
    
    from core.transcriber import MultiModelTranscriber
    
    try:
        # Create without config
        mm_transcoder = MultiModelTranscriber(config=None)
        
        print(f"✓ MultiModelTranscriber initialized")
        print(f"✓ Available models: {mm_transcoder.list_models()}")
        print(f"✓ Current model: {mm_transcoder.get_current_model()}")
        
        assert len(mm_transcoder.list_models()) == 0
        assert mm_transcoder.get_current_model() is None
        
        # Try to add model (will fail because file doesn't exist)
        result = mm_transcoder.add_model("test", "/fake/path.bin")
        print(f"✓ Add model result (expected False): {result}")
        
        return True
    except Exception as e:
        print(f"❌ MultiModelTranscriber test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_factory():
    """Test creating Transcriber from config."""
    print("\n" + "="*50)
    print("TEST 6: Config Factory")
    print("="*50)
    
    try:
        from core.transcriber import create_transcriber
        from config.settings import config
        
        # Create from config
        transcoder = create_transcriber(config)
        
        print(f"✓ Transcriber created from config")
        print(f"✓ Model name: {transcoder.model_name}")
        print(f"✓ Language: {transcoder.language}")
        print(f"✓ Threads: {transcoder.n_threads}")
        print(f"✓ Translate: {transcoder.translate}")
        
        # Verify config values applied
        assert transcoder.model_name == config.model.default_model
        assert transcoder.language == config.model.language
        assert transcoder.n_threads == config.model.n_threads
        assert transcoder.translate == config.model.translate
        
        return True
    except Exception as e:
        print(f"❌ Config factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_manager():
    """Test transcriber context manager."""
    print("\n" + "="*50)
    print("TEST 7: Context Manager")
    print("="*50)
    
    from core.transcriber import Transcriber
    
    try:
        with Transcriber(model_name="base") as transcoder:
            print(f"✓ Entered context with model: {transcoder.model_name}")
            assert transcoder.model_name == "base"
        
        print(f"✓ Exited context successfully")
        
        return True
    except Exception as e:
        print(f"❌ Context manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all transcriber tests."""
    print("\n" + "="*60)
    print("TRANSCRIBER MODULE TEST SUITE")
    print("="*60)
    
    tests = [
        test_imports,
        test_transcription_result,
        test_transcriber_initialization,
        test_transcriber_stats,
        test_multi_model_transcriber,
        test_config_factory,
        test_context_manager,
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
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
