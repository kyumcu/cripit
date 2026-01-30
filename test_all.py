"""
Final Comprehensive Test Suite
Runs all tests to verify the complete application
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging

# Reduce noise during tests
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s'
)

def run_tests():
    """Run all test suites."""
    print("\n" + "="*70)
    print("CRIPIT COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Test 1: Configuration
    print("\n[1/5] Running Configuration Tests...")
    try:
        import test_config
        success = test_config.run_all_tests()
        results['config'] = 'PASS' if success else 'FAIL'
    except Exception as e:
        print(f"  Error: {e}")
        results['config'] = 'ERROR'
    
    # Test 2: Audio Capture
    print("\n[2/5] Running Audio Capture Tests...")
    try:
        import test_audio
        success = test_audio.run_all_tests()
        results['audio'] = 'PASS' if success else 'FAIL'
    except Exception as e:
        print(f"  Error: {e}")
        results['audio'] = 'ERROR'
    
    # Test 3: Transcriber
    print("\n[3/5] Running Transcriber Tests...")
    try:
        import test_transcriber
        success = test_transcriber.run_all_tests()
        results['transcriber'] = 'PASS' if success else 'FAIL'
    except Exception as e:
        print(f"  Error: {e}")
        results['transcriber'] = 'ERROR'
    
    # Test 4: Model Manager
    print("\n[4/5] Running Model Manager Tests...")
    try:
        import test_model_manager
        success = test_model_manager.run_all_tests()
        results['model_manager'] = 'PASS' if success else 'FAIL'
    except Exception as e:
        print(f"  Error: {e}")
        results['model_manager'] = 'ERROR'
    
    # Test 5: Integration
    print("\n[5/5] Running Integration Tests...")
    try:
        import test_integration
        success = test_integration.run_all_tests()
        results['integration'] = 'PASS' if success else 'FAIL'
    except Exception as e:
        print(f"  Error: {e}")
        results['integration'] = 'ERROR'
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for module, result in results.items():
        status_icon = "✅" if result == "PASS" else "❌" if result == "FAIL" else "⚠️"
        print(f"  {status_icon} {module:20s}: {result}")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r == 'PASS')
    
    print("="*70)
    print(f"OVERALL: {passed}/{total} test suites passed")
    print("="*70)
    
    # Check dependencies
    print("\n" + "="*70)
    print("DEPENDENCY STATUS")
    print("="*70)
    
    try:
        from core.audio_capture import SOUNDDEVICE_AVAILABLE, WEBRTC_AVAILABLE
        from core.transcriber import PYWHISPERCPP_AVAILABLE
        
        deps = {
            'PyQt6': True,
            'sounddevice': SOUNDDEVICE_AVAILABLE,
            'WebRTC VAD': WEBRTC_AVAILABLE,
            'pywhispercpp': PYWHISPERCPP_AVAILABLE,
        }
        
        for dep, available in deps.items():
            icon = "✅" if available else "❌"
            print(f"  {icon} {dep}")
        
        print("="*70)
    except Exception as e:
        print(f"  Error checking dependencies: {e}")
    
    # Final verdict
    all_passed = all(r == 'PASS' for r in results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Application is ready to use.")
    else:
        print("⚠️  Some tests failed. Check output above for details.")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
