"""
Test suite for audio capture module
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
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
        from core.audio_capture import AudioCapture, RecordingState
        from core.audio_capture import WebRTCVAD, SileroVAD, BaseVAD
        print("✓ All audio capture classes imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_vad_initialization():
    """Test VAD initialization."""
    print("\n" + "="*50)
    print("TEST 2: VAD Initialization")
    print("="*50)
    
    from core.audio_capture import WebRTCVAD, WEBRTC_AVAILABLE
    
    if not WEBRTC_AVAILABLE:
        print("⚠ WebRTC VAD not available, skipping test")
        return True
    
    try:
        # Test WebRTC VAD
        vad = WebRTCVAD(sample_rate=16000, aggressiveness=2)
        print(f"✓ WebRTC VAD initialized: sample_rate=16000, aggressiveness=2")
        
        # Test with dummy audio frame
        frame_size = int(16000 * 30 / 1000)  # 30ms at 16kHz
        dummy_frame = b'\x00' * (frame_size * 2)  # 16-bit = 2 bytes per sample
        
        is_speech = vad.is_speech(dummy_frame)
        print(f"✓ VAD test on silence: is_speech={is_speech}")
        
        return True
    except Exception as e:
        print(f"❌ VAD initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_capture_initialization():
    """Test AudioCapture class initialization."""
    print("\n" + "="*50)
    print("TEST 3: AudioCapture Initialization")
    print("="*50)
    
    from core.audio_capture import AudioCapture, RecordingState, PYAUDIO_AVAILABLE
    
    try:
        # Create instance
        capture = AudioCapture(
            sample_rate=16000,
            channels=1,
            chunk_size=1024,
            vad_type="webrtc",
            vad_aggressiveness=2,
            silence_timeout=1.5
        )
        
        print(f"✓ AudioCapture initialized")
        print(f"✓ State: {capture.get_state().name}")
        print(f"✓ Recording: {capture.is_recording()}")
        print(f"✓ Buffered duration: {capture.get_buffered_duration():.3f}s")
        
        # Verify initial state
        assert capture.get_state() == RecordingState.IDLE, "Initial state should be IDLE"
        assert not capture.is_recording(), "Should not be recording initially"
        assert capture.get_buffered_duration() == 0.0, "Buffer should be empty"
        
        return True
    except Exception as e:
        print(f"❌ AudioCapture initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_list_devices():
    """Test listing audio devices."""
    print("\n" + "="*50)
    print("TEST 4: List Audio Devices")
    print("="*50)
    
    from core.audio_capture import AudioCapture, PYAUDIO_AVAILABLE
    
    if not PYAUDIO_AVAILABLE:
        print("⚠ PyAudio not available, skipping test")
        return True
    
    try:
        devices = AudioCapture.list_devices()
        
        if devices:
            print(f"✓ Found {len(devices)} input device(s):")
            for dev in devices:
                default_marker = " (DEFAULT)" if dev['default'] else ""
                print(f"  [{dev['index']}] {dev['name']}{default_marker}")
                print(f"      Channels: {dev['channels']}, Sample rate: {dev['sample_rate']} Hz")
        else:
            print("⚠ No input devices found")
        
        return True
    except Exception as e:
        print(f"❌ Failed to list devices: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_transitions():
    """Test recording state transitions."""
    print("\n" + "="*50)
    print("TEST 5: State Transitions")
    print("="*50)
    
    from core.audio_capture import AudioCapture, RecordingState, PYAUDIO_AVAILABLE
    
    if not PYAUDIO_AVAILABLE:
        print("⚠ PyAudio not available, skipping test")
        return True
    
    try:
        capture = AudioCapture()
        
        # Track state changes
        state_changes = []
        def on_state_change(state):
            state_changes.append(state)
            print(f"  State changed to: {state.name}")
        
        capture.on_state_change = on_state_change
        
        # Initial state
        print(f"✓ Initial state: {capture.get_state().name}")
        assert capture.get_state() == RecordingState.IDLE
        
        # Start capture
        print("\n  Starting capture...")
        result = capture.start()
        
        if result:
            time.sleep(0.1)  # Brief pause
            print(f"✓ Started successfully, state: {capture.get_state().name}")
            
            # Stop capture
            print("\n  Stopping capture...")
            capture.stop()
            print(f"✓ Stopped, final state: {capture.get_state().name}")
            
            assert capture.get_state() == RecordingState.IDLE, "Final state should be IDLE"
            
            if state_changes:
                print(f"\n✓ State transition history:")
                for i, state in enumerate(state_changes):
                    print(f"    {i+1}. {state.name}")
            
            return True
        else:
            print("⚠ Failed to start capture (may be no microphone available)")
            return True  # Don't fail if hardware not available
            
    except Exception as e:
        print(f"❌ State transition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_callbacks():
    """Test audio capture callbacks."""
    print("\n" + "="*50)
    print("TEST 6: Audio Callbacks")
    print("="*50)
    
    from core.audio_capture import AudioCapture, PYAUDIO_AVAILABLE
    
    if not PYAUDIO_AVAILABLE:
        print("⚠ PyAudio not available, skipping test")
        return True
    
    try:
        capture = AudioCapture(silence_timeout=0.5)  # Short timeout for testing
        
        # Track callbacks
        audio_received = []
        speech_events = []
        
        def on_audio_ready(audio_data):
            duration = len(audio_data) / 16000
            audio_received.append(duration)
            print(f"  ✓ Audio received: {duration:.2f}s")
        
        def on_speech_detected(is_speech):
            speech_events.append(is_speech)
        
        capture.on_audio_ready = on_audio_ready
        capture.on_speech_detected = on_speech_detected
        
        # Start capture
        print("  Starting capture (speak or stay silent for 2 seconds)...")
        result = capture.start()
        
        if result:
            # Record for 2 seconds
            time.sleep(2.0)
            
            # Stop
            capture.stop()
            
            print(f"\n✓ Capture stopped")
            print(f"✓ Audio segments received: {len(audio_received)}")
            if audio_received:
                total_duration = sum(audio_received)
                print(f"✓ Total audio duration: {total_duration:.2f}s")
            print(f"✓ Speech detection events: {len(speech_events)}")
            
            return True
        else:
            print("⚠ Failed to start capture (may be no microphone available)")
            return True
            
    except Exception as e:
        print(f"❌ Audio callback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_factory():
    """Test creating AudioCapture from config."""
    print("\n" + "="*50)
    print("TEST 7: Config Factory")
    print("="*50)
    
    try:
        from core.audio_capture import create_audio_capture
        from config.settings import config
        
        # Create from config
        capture = create_audio_capture(config)
        
        print(f"✓ AudioCapture created from config")
        print(f"✓ Sample rate: {capture.sample_rate} Hz")
        print(f"✓ Channels: {capture.channels}")
        print(f"✓ VAD enabled: {capture.vad is not None}")
        
        # Verify config values applied
        assert capture.sample_rate == config.audio.sample_rate
        assert capture.channels == config.audio.channels
        assert capture.silence_timeout == config.audio.silence_timeout
        
        return True
    except Exception as e:
        print(f"❌ Config factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all audio capture tests."""
    print("\n" + "="*60)
    print("AUDIO CAPTURE MODULE TEST SUITE")
    print("="*60)
    
    tests = [
        test_imports,
        test_vad_initialization,
        test_audio_capture_initialization,
        test_list_devices,
        test_state_transitions,
        test_audio_callbacks,
        test_config_factory,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
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
    print(f"TEST RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
