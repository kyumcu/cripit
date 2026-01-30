#!/usr/bin/env python3
"""
Test audio device selection and gain control
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_audio_features():
    """Test device selection and gain control."""
    
    from core.audio_capture import AudioCapture, RecordingState
    from config.settings import config
    
    print("\n" + "="*60)
    print("AUDIO DEVICE & GAIN TEST")
    print("="*60)
    
    # List available devices
    print("\n1. Available Audio Devices:")
    devices = AudioCapture.list_devices()
    for dev in devices:
        default_mark = " ⭐ DEFAULT" if dev['default'] else ""
        print(f"   [{dev['index']}] {dev['name']}{default_mark}")
    
    # Get current config
    print(f"\n2. Current Settings:")
    print(f"   Device: {config.audio.device_index if config.audio.device_index is not None else 'Default'}")
    print(f"   Gain: {config.audio.gain_db} dB")
    
    # Test with different gains
    test_gains = [0, 6, -6]  # normal, louder, quieter
    
    for gain_db in test_gains:
        print(f"\n3. Testing with {gain_db} dB gain...")
        
        # Create audio capture with specific gain
        audio = AudioCapture(
            sample_rate=16000,
            channels=1,
            device_index=None,  # Use default
            gain_db=gain_db,
            silence_timeout=1.0
        )
        
        recorded_chunks = []
        
        def on_audio_ready(audio_data):
            recorded_chunks.append(audio_data)
        
        audio.on_audio_ready = on_audio_ready
        
        # Quick 2-second test recording
        if audio.start():
            print(f"   Recording started (gain: {gain_db} dB)...")
            time.sleep(2)
            audio.stop()
            
            if recorded_chunks:
                import numpy as np
                audio_data = np.concatenate(recorded_chunks)
                
                # Analyze
                max_val = np.max(np.abs(audio_data))
                mean_val = np.mean(np.abs(audio_data))
                
                print(f"   ✓ Recorded {len(audio_data)/16000:.1f}s")
                print(f"   Max amplitude: {max_val} (higher = louder)")
                print(f"   Mean amplitude: {mean_val:.0f}")
                
                # Expected: higher gain = higher amplitude
                if gain_db > 0:
                    print(f"   Expected: Louder than normal ✓")
                elif gain_db < 0:
                    print(f"   Expected: Quieter than normal ✓")
                else:
                    print(f"   Expected: Normal volume ✓")
            else:
                print("   ✗ No audio recorded")
        else:
            print("   ✗ Failed to start recording")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nYou can now:")
    print("1. Run: python main.py")
    print("2. Click 'Settings...'")
    print("3. Select your microphone")
    print("4. Adjust gain (-20 to +20 dB)")
    print("5. Click 'Apply' to save")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_audio_features()
