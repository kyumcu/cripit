#!/usr/bin/env python3
"""
Test recording script - Records audio and saves to samples directory
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import time
import wave
import numpy as np
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def record_and_save_sample(duration=5, filename=None):
    """Record audio and save to samples directory."""
    
    from core.audio_capture import AudioCapture, RecordingState
    
    # Create samples directory
    samples_dir = os.path.join(os.path.dirname(__file__), 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"sample_recording_{timestamp}.wav"
    
    filepath = os.path.join(samples_dir, filename)
    
    print(f"\n{'='*60}")
    print("TEST RECORDING - Save to Samples")
    print(f"{'='*60}")
    print(f"Duration: {duration} seconds")
    print(f"Output: {filepath}")
    print(f"{'='*60}\n")
    
    # Initialize audio capture
    audio = AudioCapture(
        sample_rate=16000,
        channels=1,
        silence_timeout=2.0
    )
    
    # Collect audio chunks
    recorded_chunks = []
    
    def on_audio_ready(audio_data):
        """Callback for recorded audio."""
        recorded_chunks.append(audio_data)
        duration_so_far = sum(len(chunk) for chunk in recorded_chunks) / 16000
        print(f"  ✓ Received chunk: {len(audio_data)/16000:.1f}s (total: {duration_so_far:.1f}s)")
    
    def on_state_change(state):
        """Callback for state changes."""
        print(f"  → State: {state.name}")
    
    def on_speech_detected(is_speech):
        """Callback for speech detection."""
        if is_speech:
            print("  🎤 Speech detected!")
    
    audio.on_audio_ready = on_audio_ready
    audio.on_state_change = on_state_change
    audio.on_speech_detected = on_speech_detected
    
    # Start recording
    print("Starting recording...")
    print("Please speak into your microphone now!\n")
    
    if not audio.start():
        print("❌ Failed to start audio capture!")
        return False
    
    # Record for specified duration
    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            time.sleep(0.1)
            
            # Show progress
            elapsed = time.time() - start_time
            if int(elapsed) % 2 == 0 and elapsed % 1 < 0.2:  # Every 2 seconds
                buffered = audio.get_buffered_duration()
                print(f"  ⏱️  Recording... {elapsed:.1f}s (buffered: {buffered:.1f}s)")
    
    except KeyboardInterrupt:
        print("\n⚠️ Recording interrupted by user")
    
    finally:
        # Stop recording
        print("\nStopping recording...")
        audio.stop()
    
    # Save to WAV file
    if recorded_chunks:
        print(f"\n💾 Saving to {filename}...")
        
        # Concatenate all chunks
        audio_data = np.concatenate(recorded_chunks)
        
        # Save as WAV
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_data.tobytes())
        
        # Get file size
        file_size = os.path.getsize(filepath)
        duration = len(audio_data) / 16000
        
        print(f"\n{'='*60}")
        print("✅ RECORDING SAVED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"File: {filepath}")
        print(f"Size: {file_size / 1024:.1f} KB")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Sample rate: 16000 Hz")
        print(f"Channels: 1 (mono)")
        print(f"Format: 16-bit PCM")
        print(f"{'='*60}\n")
        
        return True
    else:
        print("❌ No audio recorded!")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Record audio sample')
    parser.add_argument('--duration', type=int, default=5, help='Recording duration in seconds')
    parser.add_argument('--filename', type=str, help='Output filename (default: auto-generated)')
    
    args = parser.parse_args()
    
    success = record_and_save_sample(args.duration, args.filename)
    sys.exit(0 if success else 1)
