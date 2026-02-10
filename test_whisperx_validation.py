"""
Validation tests for WhisperX implementation
Checks file structure, imports, and code without requiring dependencies
"""

import sys
import os
import ast
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_file_structure():
    """Test that all required files exist."""
    print("\n" + "="*60)
    print("TEST 1: File Structure")
    print("="*60)
    
    required_files = [
        'core/base_transcriber.py',
        'core/whispercpp_transcriber.py',
        'core/whisperx_transcriber.py',
        'core/transcriber_factory.py',
        'core/transcriber.py',  # Compatibility shim
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} NOT FOUND")
            return False
    
    return True


def test_python_syntax():
    """Test that all Python files have valid syntax."""
    print("\n" + "="*60)
    print("TEST 2: Python Syntax")
    print("="*60)
    
    files_to_check = [
        'core/base_transcriber.py',
        'core/whispercpp_transcriber.py',
        'core/whisperx_transcriber.py',
        'core/transcriber_factory.py',
        'core/transcriber.py',
        'test_transcriber.py',
        'test_transcriber_factory.py',
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for file_path in files_to_check:
        full_path = os.path.join(base_dir, file_path)
        if not os.path.exists(full_path):
            print(f"⚠️ {file_path} not found, skipping")
            continue
            
        try:
            with open(full_path, 'r') as f:
                source = f.read()
            ast.parse(source)
            print(f"✓ {file_path} - valid syntax")
        except SyntaxError as e:
            print(f"❌ {file_path} - syntax error: {e}")
            return False
    
    return True


def test_base_transcriber_structure():
    """Test that base_transcriber.py has correct structure."""
    print("\n" + "="*60)
    print("TEST 3: Base Transcriber Structure")
    print("="*60)
    
    with open('core/base_transcriber.py', 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    # Check for required classes
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    required_classes = ['BaseTranscriber', 'TranscriptionResult']
    for cls in required_classes:
        if cls in classes:
            print(f"✓ Class {cls} defined")
        else:
            print(f"❌ Class {cls} NOT FOUND")
            return False
    
    # Check for required exceptions
    exceptions = ['EngineNotAvailableError', 'ModelLoadError', 'TranscriptionError']
    for exc in exceptions:
        if exc in classes:
            print(f"✓ Exception {exc} defined")
        else:
            print(f"❌ Exception {exc} NOT FOUND")
            return False
    
    # Check BaseTranscriber has required methods
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'BaseTranscriber':
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            required_methods = ['load_model', 'unload_model', 'transcribe', 
                              'is_ready', 'get_stats', 'get_device_name']
            for method in required_methods:
                if method in methods:
                    print(f"✓ BaseTranscriber.{method}() defined")
                else:
                    print(f"❌ BaseTranscriber.{method}() NOT FOUND")
                    return False
    
    return True


def test_whisperx_transcriber_structure():
    """Test that whisperx_transcriber.py has correct structure."""
    print("\n" + "="*60)
    print("TEST 4: WhisperX Transcriber Structure")
    print("="*60)
    
    with open('core/whisperx_transcriber.py', 'r') as f:
        source = f.read()
    
    # Check for required imports
    if 'from core.base_transcriber import' in source:
        print("✓ Imports from base_transcriber")
    else:
        print("❌ Missing import from base_transcriber")
        return False
    
    if 'class WhisperXTranscriber(BaseTranscriber)' in source:
        print("✓ WhisperXTranscriber inherits from BaseTranscriber")
    else:
        print("❌ WhisperXTranscriber does not inherit from BaseTranscriber")
        return False
    
    # Check for lazy import pattern
    if 'WHISPERX_AVAILABLE = False' in source:
        print("✓ Lazy import pattern for WhisperX")
    else:
        print("❌ Missing lazy import pattern")
        return False
    
    return True


def test_factory_structure():
    """Test that transcriber_factory.py has correct structure."""
    print("\n" + "="*60)
    print("TEST 5: Transcriber Factory Structure")
    print("="*60)
    
    with open('core/transcriber_factory.py', 'r') as f:
        source = f.read()
    
    # Check for required functions
    required_funcs = ['check_engine_availability', 'create_transcriber', 'reload_transcriber']
    for func in required_funcs:
        if f'def {func}(' in source:
            print(f"✓ Function {func}() defined")
        else:
            print(f"❌ Function {func}() NOT FOUND")
            return False
    
    # Check that it handles both engines
    if 'whispercpp' in source and 'whisperx' in source:
        print("✓ Handles both whispercpp and whisperx engines")
    else:
        print("❌ Missing engine handling")
        return False
    
    return True


def test_config_settings():
    """Test that config/settings.py has WhisperX settings."""
    print("\n" + "="*60)
    print("TEST 6: Config Settings")
    print("="*60)
    
    with open('config/settings.py', 'r') as f:
        source = f.read()
    
    # Check for WhisperX settings
    whisperx_settings = [
        'asr_engine',
        'whisperx_model',
        'whisperx_compute_type',
        'whisperx_diarize',
        'whisperx_hf_token',
        'whisperx_device',
    ]
    
    for setting in whisperx_settings:
        if setting in source:
            print(f"✓ Setting {setting} present")
        else:
            print(f"❌ Setting {setting} NOT FOUND")
            return False
    
    # Check default values
    if 'asr_engine: str = "whispercpp"' in source:
        print("✓ Default engine is whispercpp")
    else:
        print("❌ Default engine not set correctly")
        return False
    
    return True


def test_backwards_compatibility():
    """Test that core/transcriber.py maintains backwards compatibility."""
    print("\n" + "="*60)
    print("TEST 7: Backwards Compatibility")
    print("="*60)
    
    with open('core/transcriber.py', 'r') as f:
        source = f.read()
    
    # Check that it re-exports from whispercpp_transcriber
    if 'from core.whispercpp_transcriber import' in source:
        print("✓ Re-exports from whispercpp_transcriber")
    else:
        print("❌ Missing re-exports")
        return False
    
    # Check for Transcriber alias
    if 'Transcriber = WhisperCppTranscriber' in source:
        print("✓ Transcriber alias defined")
    else:
        print("❌ Transcriber alias NOT FOUND")
        return False
    
    return True


def test_requirements():
    """Test that requirements.txt includes WhisperX dependencies."""
    print("\n" + "="*60)
    print("TEST 8: Requirements")
    print("="*60)
    
    with open('requirements.txt', 'r') as f:
        requirements = f.read()
    
    # Check for WhisperX-related dependencies
    deps = ['whisperx', 'torch', 'torchaudio']
    
    for dep in deps:
        if dep in requirements.lower():
            print(f"✓ {dep} in requirements")
        else:
            print(f"❌ {dep} NOT in requirements")
            return False
    
    return True


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("WHISPERX IMPLEMENTATION VALIDATION SUITE")
    print("="*70)
    
    tests = [
        test_file_structure,
        test_python_syntax,
        test_base_transcriber_structure,
        test_whisperx_transcriber_structure,
        test_factory_structure,
        test_config_settings,
        test_backwards_compatibility,
        test_requirements,
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
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
