"""
Test suite for transcriber module (updated for WhisperX support)
Tests both whisper.cpp and WhisperX transcribers
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


def test_base_transcriber_imports():
    """Test that base transcriber can be imported."""
    print("\n" + "="*50)
    print("TEST 1: Base Transcriber Imports")
    print("="*50)
    
    try:
        from core.base_transcriber import BaseTranscriber, TranscriptionResult
        from core.base_transcriber import EngineNotAvailableError, ModelLoadError, TranscriptionError
        print("✓ Base transcriber classes imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_transcription_result_base():
    """Test TranscriptionResult dataclass from base."""
    print("\n" + "="*50)
    print("TEST 2: TranscriptionResult (Base)")
    print("="*50)
    
    from core.base_transcriber import TranscriptionResult
    
    # Create result with all fields including speakers
    result = TranscriptionResult(
        text="Hello world",
        language="en",
        confidence=0.95,
        duration=2.5,
        processing_time=0.3,
        segments=[{'text': 'Hello world', 'language': 'en'}],
        is_partial=False,
        speakers=[{'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 2.5, 'text': 'Hello world'}]
    )
    
    print(f"✓ Text: {result.text}")
    print(f"✓ Language: {result.language}")
    print(f"✓ Duration: {result.duration}s")
    print(f"✓ Processing time: {result.processing_time}s")
    print(f"✓ Speakers: {len(result.speakers) if result.speakers else 0} speakers")
    
    assert result.text == "Hello world"
    assert result.language == "en"
    assert result.duration == 2.5
    assert result.speakers is not None
    assert len(result.speakers) == 1
    
    return True


def test_whispercpp_transcriber():
    """Test WhisperCppTranscriber."""
    print("\n" + "="*50)
    print("TEST 3: WhisperCppTranscriber")
    print("="*50)
    
    from core.whispercpp_transcriber import WhisperCppTranscriber, PYWHISPERCPP_AVAILABLE
    
    if not PYWHISPERCPP_AVAILABLE:
        print("⚠️ pywhispercpp not installed, skipping detailed tests")
        # Still test initialization
        try:
            transcoder = WhisperCppTranscriber(
                model_path="/fake/path/model.bin",
                model_name="large-v3-turbo",
                language="en",
                n_threads=4,
            )
            print("✓ WhisperCppTranscriber initialized (without pywhispercpp)")
            return True
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    try:
        transcoder = WhisperCppTranscriber(
            model_path="/fake/path/model.bin",
            model_name="large-v3-turbo",
            language="en",
            n_threads=4,
            translate=False,
            use_cuda=True,
            cuda_device=0,
        )
        
        print(f"✓ WhisperCppTranscriber initialized")
        print(f"✓ Model name: {transcoder.model_name}")
        print(f"✓ Language: {transcoder.language}")
        print(f"✓ Threads: {transcoder.n_threads}")
        print(f"✓ Ready: {transcoder.is_ready()}")
        
        assert transcoder.model_name == "large-v3-turbo"
        assert not transcoder.is_ready()  # Not loaded yet
        
        # Test stats
        stats = transcoder.get_stats()
        assert 'total_transcriptions' in stats
        assert 'model_name' in stats
        assert stats['model_name'] == "large-v3-turbo"
        
        print(f"✓ Stats working correctly")
        
        return True
    except Exception as e:
        print(f"❌ WhisperCppTranscriber test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_whisperx_transcriber():
    """Test WhisperXTranscriber."""
    print("\n" + "="*50)
    print("TEST 4: WhisperXTranscriber")
    print("="*50)
    
    from core.whisperx_transcriber import WhisperXTranscriber, WHISPERX_AVAILABLE
    
    try:
        transcoder = WhisperXTranscriber(
            model_name="base",
            device="cpu",
            compute_type="int8",
            language="en",
            enable_diarization=False,
            hf_token=None,
        )
        
        print(f"✓ WhisperXTranscriber initialized")
        print(f"✓ Model name: {transcoder.model_name}")
        print(f"✓ Device: {transcoder.device}")
        print(f"✓ Compute type: {transcoder.compute_type}")
        print(f"✓ Language: {transcoder.language}")
        print(f"✓ Diarization: {transcoder.enable_diarization}")
        print(f"✓ Ready: {transcoder.is_ready()}")
        
        assert transcoder.model_name == "base"
        assert transcoder.device == "cpu"
        assert not transcoder.is_ready()  # Not loaded yet
        
        # Test stats
        stats = transcoder.get_stats()
        assert 'total_transcriptions' in stats
        assert 'model_name' in stats
        assert 'diarization_enabled' in stats
        assert stats['model_name'] == "base"
        assert stats['diarization_enabled'] == False
        
        print(f"✓ Stats working correctly")
        print(f"  - Device: {stats['device']}")
        print(f"  - Using GPU: {stats['using_gpu']}")
        
        return True
    except Exception as e:
        print(f"❌ WhisperXTranscriber test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transcriber_factory():
    """Test transcriber factory."""
    print("\n" + "="*50)
    print("TEST 5: Transcriber Factory")
    print("="*50)
    
    from core.transcriber_factory import create_transcriber, check_engine_availability, reload_transcriber
    from core.base_transcriber import BaseTranscriber
    
    try:
        # Check engine availability
        engines = check_engine_availability()
        print(f"✓ Engine availability checked")
        print(f"  - whispercpp: {engines.get('whispercpp', False)}")
        print(f"  - whisperx: {engines.get('whisperx', False)}")
        
        assert 'whispercpp' in engines
        assert 'whisperx' in engines
        
        # Try to create default transcriber
        transcoder = create_transcriber()
        if transcoder:
            print(f"✓ Default transcriber created: {type(transcoder).__name__}")
            assert isinstance(transcoder, BaseTranscriber)
            
            # Test stats interface
            stats = transcoder.get_stats()
            assert isinstance(stats, dict)
            print(f"✓ Transcriber implements BaseTranscriber interface")
        else:
            print("⚠️ No transcriber created (no engines available)")
        
        return True
    except Exception as e:
        print(f"❌ Factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transcriber_compatibility():
    """Test that both transcribers have compatible interfaces."""
    print("\n" + "="*50)
    print("TEST 6: Transcriber Interface Compatibility")
    print("="*50)
    
    from core.base_transcriber import BaseTranscriber
    from core.whispercpp_transcriber import WhisperCppTranscriber
    from core.whisperx_transcriber import WhisperXTranscriber
    
    try:
        # Create instances
        cpp_transcoder = WhisperCppTranscriber(model_name="base")
        x_transcoder = WhisperXTranscriber(model_name="base", device="cpu")
        
        # Test that both inherit from BaseTranscriber
        assert isinstance(cpp_transcoder, BaseTranscriber), "WhisperCppTranscriber should inherit from BaseTranscriber"
        assert isinstance(x_transcoder, BaseTranscriber), "WhisperXTranscriber should inherit from BaseTranscriber"
        
        print("✓ Both transcribers inherit from BaseTranscriber")
        
        # Test common interface methods
        for transcoder in [cpp_transcoder, x_transcoder]:
            assert hasattr(transcoder, 'load_model')
            assert hasattr(transcoder, 'unload_model')
            assert hasattr(transcoder, 'transcribe')
            assert hasattr(transcoder, 'is_ready')
            assert hasattr(transcoder, 'get_stats')
            assert hasattr(transcoder, 'get_device_name')
            
            # Test method signatures (they should be callable)
            assert callable(transcoder.load_model)
            assert callable(transcoder.unload_model)
            assert callable(transcoder.transcribe)
            assert callable(transcoder.is_ready)
            assert callable(transcoder.get_stats)
            assert callable(transcoder.get_device_name)
        
        print("✓ Both transcribers implement required interface methods")
        
        # Test stats return type
        for transcoder in [cpp_transcoder, x_transcoder]:
            stats = transcoder.get_stats()
            assert isinstance(stats, dict)
            assert 'model_name' in stats
        
        print("✓ Stats interface consistent between implementations")
        
        return True
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transcriber_with_audio():
    """Test transcription with dummy audio data."""
    print("\n" + "="*50)
    print("TEST 7: Transcription with Audio Data")
    print("="*50)
    
    from core.transcriber_factory import create_transcriber
    
    try:
        transcoder = create_transcriber()
        if not transcoder:
            print("⚠️ No transcriber available, skipping audio test")
            return True
        
        # Create dummy audio data (1 second of silence)
        sample_rate = 16000
        duration = 1.0
        audio_data = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        print(f"✓ Created {duration}s of dummy audio")
        
        # Note: We don't actually transcribe because models aren't loaded
        # Just verify the transcribe method exists and accepts the right args
        import inspect
        sig = inspect.signature(transcoder.transcribe)
        params = list(sig.parameters.keys())
        
        assert 'audio_data' in params
        assert 'is_partial' in params
        
        print(f"✓ Transcribe method accepts correct parameters: {params}")
        
        return True
    except Exception as e:
        print(f"❌ Audio transcription test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backwards_compatibility():
    """Test that old imports still work."""
    print("\n" + "="*50)
    print("TEST 8: Backwards Compatibility")
    print("="*50)
    
    try:
        # Old imports should still work
        from core.transcriber import Transcriber, TranscriptionResult
        from core.transcriber import create_transcriber, MultiModelTranscriber
        from core.transcriber import PYWHISPERCPP_AVAILABLE, SAMPLE_RATE
        
        print("✓ Old import paths work")
        
        # Transcriber should be an alias for WhisperCppTranscriber
        from core.whispercpp_transcriber import WhisperCppTranscriber
        assert Transcriber is WhisperCppTranscriber, "Transcriber should be alias for WhisperCppTranscriber"
        
        print("✓ Transcriber is alias for WhisperCppTranscriber")
        
        # Can still create using old interface
        transcoder = Transcriber(model_name="base")
        assert transcoder.model_name == "base"
        
        print("✓ Can create transcribers using old interface")
        
        return True
    except Exception as e:
        print(f"❌ Backwards compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all transcriber tests."""
    print("\n" + "="*60)
    print("TRANSCRIBER MODULE TEST SUITE (WhisperX Support)")
    print("="*60)
    
    tests = [
        test_base_transcriber_imports,
        test_transcription_result_base,
        test_whispercpp_transcriber,
        test_whisperx_transcriber,
        test_transcriber_factory,
        test_transcriber_compatibility,
        test_transcriber_with_audio,
        test_backwards_compatibility,
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
