"""
Test suite for transcriber factory
Tests engine checking and transcriber creation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def test_engine_availability_check():
    """Test engine availability checking."""
    print("\n" + "="*50)
    print("TEST 1: Engine Availability Check")
    print("="*50)
    
    from core.transcriber_factory import check_engine_availability
    
    try:
        engines = check_engine_availability()
        
        print(f"✓ Engine availability checked successfully")
        print(f"  - whispercpp: {engines.get('whispercpp', False)}")
        print(f"  - whisperx: {engines.get('whisperx', False)}")
        
        # Should return dict with both keys
        assert isinstance(engines, dict)
        assert 'whispercpp' in engines
        assert 'whisperx' in engines
        
        # Values should be booleans
        assert isinstance(engines['whispercpp'], bool)
        assert isinstance(engines['whisperx'], bool)
        
        return True
    except Exception as e:
        print(f"❌ Engine check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_create_whispercpp_transcriber():
    """Test creating whisper.cpp transcriber."""
    print("\n" + "="*50)
    print("TEST 2: Create WhisperCpp Transcriber")
    print("="*50)
    
    from core.transcriber_factory import create_transcriber, check_engine_availability
    from core.whispercpp_transcriber import WhisperCppTranscriber
    
    engines = check_engine_availability()
    
    try:
        # Create whispercpp transcriber explicitly
        transcoder = create_transcriber("whispercpp")
        
        if engines['whispercpp']:
            assert transcoder is not None
            assert isinstance(transcoder, WhisperCppTranscriber)
            print(f"✓ WhisperCpp transcriber created successfully")
            print(f"  - Model: {transcoder.model_name}")
            print(f"  - Type: {type(transcoder).__name__}")
        else:
            # If whispercpp not available, should return None or fallback
            print(f"⚠️ whispercpp not available, result: {transcoder}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create whispercpp transcriber: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_create_whisperx_transcriber():
    """Test creating WhisperX transcriber."""
    print("\n" + "="*50)
    print("TEST 3: Create WhisperX Transcriber")
    print("="*50)
    
    from core.transcriber_factory import create_transcriber, check_engine_availability
    from core.whisperx_transcriber import WhisperXTranscriber
    
    engines = check_engine_availability()
    
    try:
        # Create whisperx transcriber explicitly
        transcoder = create_transcriber("whisperx")
        
        if engines['whisperx']:
            assert transcoder is not None
            assert isinstance(transcoder, WhisperXTranscriber)
            print(f"✓ WhisperX transcriber created successfully")
            print(f"  - Model: {transcoder.model_name}")
            print(f"  - Device: {transcoder.device}")
            print(f"  - Type: {type(transcoder).__name__}")
        else:
            # If whisperx not available, should return None
            assert transcoder is None, "Should return None when WhisperX not available"
            print(f"⚠️ whisperx not available (expected)")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create whisperx transcriber: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_default_transcriber():
    """Test creating default transcriber from config."""
    print("\n" + "="*50)
    print("TEST 4: Default Transcriber from Config")
    print("="*50)
    
    from core.transcriber_factory import create_transcriber
    from config.settings import config
    
    try:
        # Create default transcriber (uses config)
        transcoder = create_transcriber()
        
        if transcoder:
            print(f"✓ Default transcriber created")
            print(f"  - Type: {type(transcoder).__name__}")
            print(f"  - Engine from config: {config.model.asr_engine}")
            
            # Verify it's the right type based on config
            if config.model.asr_engine == "whisperx":
                from core.whisperx_transcriber import WhisperXTranscriber
                assert isinstance(transcoder, WhisperXTranscriber)
            else:
                from core.whispercpp_transcriber import WhisperCppTranscriber
                assert isinstance(transcoder, WhisperCppTranscriber)
            
            print(f"✓ Transcriber type matches config")
        else:
            print(f"⚠️ No transcriber created (no engines available)")
        
        return True
    except Exception as e:
        print(f"❌ Default transcriber test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transcriber_reload():
    """Test transcriber hot-swap."""
    print("\n" + "="*50)
    print("TEST 5: Transcriber Hot-Swap")
    print("="*50)
    
    from core.transcriber_factory import create_transcriber, reload_transcriber
    
    try:
        # Create initial transcriber
        old_transcoder = create_transcriber()
        
        if not old_transcoder:
            print("⚠️ No transcriber available, skipping reload test")
            return True
        
        old_type = type(old_transcoder).__name__
        old_stats = old_transcoder.get_stats()
        
        print(f"✓ Initial transcriber: {old_type}")
        
        # Try to reload to the other engine
        new_engine = "whisperx" if old_type == "WhisperCppTranscriber" else "whispercpp"
        
        new_transcoder = reload_transcriber(old_transcoder, new_engine)
        
        if new_transcoder:
            new_type = type(new_transcoder).__name__
            print(f"✓ Reloaded transcriber: {new_type}")
            
            # Verify old one was unloaded
            assert not old_transcoder.is_ready(), "Old transcriber should be unloaded"
            print(f"✓ Old transcriber properly unloaded")
            
            # Verify config was updated
            from config.settings import config
            assert config.model.asr_engine == new_engine
            print(f"✓ Config updated to: {new_engine}")
        else:
            print(f"⚠️ Could not reload to {new_engine} (engine may not be available)")
        
        return True
    except Exception as e:
        print(f"❌ Transcriber reload test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_integration():
    """Test factory integration with config."""
    print("\n" + "="*50)
    print("TEST 6: Config Integration")
    print("="*50)
    
    from core.transcriber_factory import create_transcriber
    from config.settings import config
    
    try:
        # Save original engine
        original_engine = config.model.asr_engine
        
        # Test whispercpp config
        config.model.asr_engine = "whispercpp"
        transcoder = create_transcriber()
        
        if transcoder:
            from core.whispercpp_transcriber import WhisperCppTranscriber
            assert isinstance(transcoder, WhisperCppTranscriber)
            print(f"✓ whispercpp config creates WhisperCppTranscriber")
        
        # Test whisperx config (only if available)
        config.model.asr_engine = "whisperx"
        transcoder = create_transcriber()
        
        if transcoder:
            from core.whisperx_transcriber import WhisperXTranscriber
            assert isinstance(transcoder, WhisperXTranscriber)
            print(f"✓ whisperx config creates WhisperXTranscriber")
        
        # Restore original
        config.model.asr_engine = original_engine
        print(f"✓ Config integration working correctly")
        
        return True
    except Exception as e:
        print(f"❌ Config integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all factory tests."""
    print("\n" + "="*60)
    print("TRANSCRIBER FACTORY TEST SUITE")
    print("="*60)
    
    tests = [
        test_engine_availability_check,
        test_create_whispercpp_transcriber,
        test_create_whisperx_transcriber,
        test_default_transcriber,
        test_transcriber_reload,
        test_config_integration,
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
