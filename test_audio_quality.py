#!/usr/bin/env python3
"""
Audio Quality Test - Records and analyzes audio to verify quality
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

def analyze_audio_quality(audio_data: np.ndarray) -> dict:
    """Analyze audio data for quality metrics."""
    
    # Basic statistics
    mean_val = np.mean(audio_data)
    std_val = np.std(audio_data)
    max_val = np.max(audio_data)
    min_val = np.min(audio_data)
    
    # Check for clipping (values at max int16 range)
    clip_threshold = 32700
    clipping_count = np.sum(np.abs(audio_data) > clip_threshold)
    clipping_percent = (clipping_count / len(audio_data)) * 100
    
    # Dynamic range (ratio of signal to noise)
    # Assuming quietest non-zero samples are noise floor
    non_zero = audio_data[audio_data != 0]
    if len(non_zero) > 0:
        noise_floor = np.percentile(np.abs(non_zero), 10)
        signal_peak = np.max(np.abs(audio_data))
        dynamic_range_db = 20 * np.log10(signal_peak / (noise_floor + 1))
    else:
        dynamic_range_db = 0
    
    # Check for DC offset (should be close to 0 for proper audio)
    dc_offset = mean_val
    
    return {
        'mean': mean_val,
        'std': std_val,
        'max': max_val,
        'min': min_val,
        'clipping_percent': clipping_percent,
        'dynamic_range_db': dynamic_range_db,
        'dc_offset': dc_offset,
        'samples': len(audio_data),
        'duration': len(audio_data) / 16000
    }

def record_with_quality_check(duration=5, filename=None):
    """Record audio and check quality."""
    
    from core.audio_capture import AudioCapture, RecordingState
    
    # Create samples directory
    samples_dir = os.path.join(os.path.dirname(__file__), 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"quality_test_{timestamp}.wav"
    
    filepath = os.path.join(samples_dir, filename)
    
    print(f"\n{'='*60}")
    print("AUDIO QUALITY TEST")
    print(f"{'='*60}")
    print(f"Duration: {duration} seconds")
    print(f"Output: {filepath}")
    print(f"\nPlease speak clearly into your microphone...")
    print(f"{'='*60}\n")
    
    # Initialize audio capture
    audio = AudioCapture(
        sample_rate=16000,
        channels=1,
        silence_timeout=3.0  # Longer timeout to capture more
    )
    
    # Collect audio chunks
    recorded_chunks = []
    state_history = []
    
    def on_audio_ready(audio_data):
        """Callback for recorded audio."""
        recorded_chunks.append(audio_data.copy())
        print(f"  ✓ Received: {len(audio_data)/16000:.1f}s (total: {sum(len(c) for c in recorded_chunks)/16000:.1f}s)")
    
    def on_state_change(state):
        """Callback for state changes."""
        state_history.append((time.time(), state.name))
        print(f"  → State: {state.name}")
    
    audio.on_audio_ready = on_audio_ready
    audio.on_state_change = on_state_change
    
    # Start recording
    if not audio.start():
        print("❌ Failed to start audio capture!")
        return False
    
    # Record for specified duration
    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n⚠️ Recording interrupted")
    
    finally:
        audio.stop()
    
    # Analyze recorded audio
    if recorded_chunks:
        print(f"\n{'='*60}")
        print("ANALYZING AUDIO QUALITY...")
        print(f"{'='*60}")
        
        # Concatenate all chunks
        audio_data = np.concatenate(recorded_chunks)
        
        # Analyze quality
        metrics = analyze_audio_quality(audio_data)
        
        print(f"\n📊 QUALITY METRICS:")
        print(f"  Duration:        {metrics['duration']:.1f} seconds")
        print(f"  Samples:         {metrics['samples']:,}")
        print(f"  DC Offset:       {metrics['dc_offset']:.2f} (should be near 0)")
        print(f"  Standard Dev:    {metrics['std']:.2f} (higher = more audio activity)")
        print(f"  Min/Max:         {metrics['min']:,} / {metrics['max']:,}")
        print(f"  Clipping:        {metrics['clipping_percent']:.2f}% (should be < 1%)")
        print(f"  Dynamic Range:   {metrics['dynamic_range_db']:.1f} dB (higher = better)")
        
        # Quality assessment
        print(f"\n✅ QUALITY ASSESSMENT:")
        
        # Check for white noise issue
        if metrics['std'] < 100:
            print("  ⚠️  WARNING: Very low audio variance - possible white noise or no input!")
            print("      (Expected: > 500 for normal speech)")
        elif metrics['std'] > 500:
            print("  ✓ Good audio activity detected")
        
        # Check DC offset
        if abs(metrics['dc_offset']) > 500:
            print(f"  ⚠️  WARNING: High DC offset ({metrics['dc_offset']:.0f})")
        else:
            print("  ✓ DC offset normal")
        
        # Check clipping
        if metrics['clipping_percent'] > 1.0:
            print(f"  ⚠️  WARNING: High clipping ({metrics['clipping_percent']:.1f}%)")
            print("      Recording may be too loud - speak softer or move mic further")
        else:
            print("  ✓ No significant clipping")
        
        # Save to WAV file
        print(f"\n💾 Saving to {filename}...")
        
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_data.tobytes())
        
        file_size = os.path.getsize(filepath)
        
        print(f"\n{'='*60}")
        print("✅ RECORDING SAVED")
        print(f"{'='*60}")
        print(f"File: {filepath}")
        print(f"Size: {file_size / 1024:.1f} KB")
        print(f"{'='*60}\n")
        
        return metrics
    else:
        print("❌ No audio recorded!")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test audio recording quality')
    parser.add_argument('--duration', type=int, default=5, help='Recording duration in seconds')
    parser.add_argument('--filename', type=str, help='Output filename (default: auto-generated)')
    
    args = parser.parse_args()
    
    metrics = record_with_quality_check(args.duration, args.filename)
    
    if metrics:
        # Determine if quality is acceptable
        is_good = (
            metrics['std'] > 500 and  # Good audio variance
            abs(metrics['dc_offset']) < 500 and  # Low DC offset
            metrics['clipping_percent'] < 1.0  # No significant clipping
        )
        
        if is_good:
            print("🎉 AUDIO QUALITY CHECK PASSED! Your recording should sound good.")
            sys.exit(0)
        else:
            print("⚠️  AUDIO QUALITY ISSUES DETECTED - Please review warnings above")
            sys.exit(1)
    else:
        sys.exit(1)
