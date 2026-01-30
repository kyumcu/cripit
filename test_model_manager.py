"""
Test suite for model manager module
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path

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
        from core.model_manager import ModelManager, ModelInfo
        from core.model_manager import get_model_manager
        print("✓ All model manager classes imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_model_manager_initialization():
    """Test ModelManager initialization."""
    print("\n" + "="*50)
    print("TEST 2: ModelManager Initialization")
    print("="*50)
    
    from core.model_manager import ModelManager
    
    try:
        # Create manager
        manager = ModelManager()
        
        print(f"✓ ModelManager initialized")
        print(f"✓ Models directory: {manager.models_dir}")
        print(f"✓ Total available models: {len(manager.list_models())}")
        
        assert manager.models_dir.exists(), "Models directory should exist"
        assert len(manager.list_models()) > 0, "Should have models defined"
        
        return True
    except Exception as e:
        print(f"❌ ModelManager initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_info():
    """Test model information retrieval."""
    print("\n" + "="*50)
    print("TEST 3: Model Information")
    print("="*50)
    
    from core.model_manager import ModelManager
    
    try:
        manager = ModelManager()
        
        # Test getting model info
        info = manager.get_model_info("large-v3-turbo")
        
        print(f"✓ Model: {info.name}")
        print(f"✓ File: {info.file}")
        print(f"✓ Size: {info.size_mb} MB")
        print(f"✓ Parameters: {info.params}")
        print(f"✓ URL: {info.url[:50]}...")
        print(f"✓ Description: {info.description}")
        
        assert info.name == "large-v3-turbo"
        assert info.size_mb == 1500
        assert info.params == "809M"
        
        # Test unknown model
        unknown = manager.get_model_info("nonexistent")
        assert unknown is None, "Unknown model should return None"
        print("✓ Unknown model correctly returns None")
        
        return True
    except Exception as e:
        print(f"❌ Model info test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_list_models():
    """Test listing models."""
    print("\n" + "="*50)
    print("TEST 4: List Models")
    print("="*50)
    
    from core.model_manager import ModelManager
    
    try:
        manager = ModelManager()
        
        all_models = manager.list_models()
        print(f"✓ All models ({len(all_models)}):")
        
        # Show first 5 models
        for model_name in all_models[:5]:
            info = manager.get_model_info(model_name)
            print(f"  - {model_name}: {info.params}, {info.size_mb} MB")
        
        if len(all_models) > 5:
            print(f"  ... and {len(all_models) - 5} more")
        
        assert "large-v3-turbo" in all_models, "Should have large-v3-turbo"
        assert "tiny" in all_models, "Should have tiny"
        assert "base" in all_models, "Should have base"
        
        return True
    except Exception as e:
        print(f"❌ List models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_availability():
    """Test checking model availability."""
    print("\n" + "="*50)
    print("TEST 5: Model Availability")
    print("="*50)
    
    from core.model_manager import ModelManager
    
    try:
        manager = ModelManager()
        
        # Check availability (none should be available in test environment)
        available = manager.get_available_models()
        missing = manager.get_missing_models()
        
        print(f"✓ Available models: {len(available)}")
        if available:
            for model in available:
                size = manager.get_model_size(model)
                print(f"  - {model}: {size / (1024**2):.1f} MB")
        
        print(f"✓ Missing models: {len(missing)}")
        print(f"  First 3: {missing[:3]}")
        
        # Verify consistency
        assert len(available) + len(missing) == len(manager.list_models())
        
        return True
    except Exception as e:
        print(f"❌ Model availability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_model_path():
    """Test getting model paths."""
    print("\n" + "="*50)
    print("TEST 6: Get Model Path")
    print("="*50)
    
    from core.model_manager import ModelManager
    
    try:
        manager = ModelManager()
        
        # Get path for unavailable model (should return None)
        path = manager.get_model_path("large-v3-turbo")
        print(f"✓ Path for unavailable model: {path}")
        
        # Test with unknown model
        unknown_path = manager.get_model_path("nonexistent")
        assert unknown_path is None
        print("✓ Unknown model path correctly returns None")
        
        return True
    except Exception as e:
        print(f"❌ Get model path test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_singleton():
    """Test ModelManager singleton."""
    print("\n" + "="*50)
    print("TEST 7: Singleton Pattern")
    print("="*50)
    
    from core.model_manager import get_model_manager, ModelManager
    
    try:
        # Get singleton instance twice
        manager1 = get_model_manager()
        manager2 = get_model_manager()
        
        print(f"✓ Singleton instance 1: {id(manager1)}")
        print(f"✓ Singleton instance 2: {id(manager2)}")
        
        assert manager1 is manager2, "Should be same instance"
        print("✓ Both references point to same instance")
        
        return True
    except Exception as e:
        print(f"❌ Singleton test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_descriptions():
    """Test that all models have proper descriptions."""
    print("\n" + "="*50)
    print("TEST 8: Model Descriptions")
    print("="*50)
    
    from core.model_manager import ModelManager
    
    try:
        manager = ModelManager()
        
        print("✓ Checking all models have descriptions...")
        
        for model_name in manager.list_models():
            info = manager.get_model_info(model_name)
            assert info.name, f"{model_name} missing name"
            assert info.file, f"{model_name} missing file"
            assert info.size_mb > 0, f"{model_name} missing size"
            assert info.params, f"{model_name} missing params"
            assert info.url, f"{model_name} missing URL"
            assert info.description, f"{model_name} missing description"
        
        print(f"✓ All {len(manager.list_models())} models have complete info")
        
        return True
    except Exception as e:
        print(f"❌ Model descriptions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all model manager tests."""
    print("\n" + "="*60)
    print("MODEL MANAGER MODULE TEST SUITE")
    print("="*60)
    
    tests = [
        test_imports,
        test_model_manager_initialization,
        test_model_info,
        test_list_models,
        test_model_availability,
        test_get_model_path,
        test_singleton,
        test_model_descriptions,
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
