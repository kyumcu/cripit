"""
Unit tests for transcriber module (no external dependencies required)
Tests structure, imports, and interfaces without requiring numpy or models
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestBaseTranscriber(unittest.TestCase):
    """Test base transcriber classes."""
    
    def test_import_base_transcriber(self):
        """Test that base transcriber can be imported."""
        from core.base_transcriber import BaseTranscriber, TranscriptionResult
        from core.base_transcriber import EngineNotAvailableError, ModelLoadError, TranscriptionError
        self.assertTrue(True)
    
    def test_transcription_result_creation(self):
        """Test TranscriptionResult dataclass."""
        from core.base_transcriber import TranscriptionResult
        
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            confidence=0.95,
            duration=2.5,
            processing_time=0.3,
            segments=[{'text': 'Hello world'}],
            is_partial=False,
            speakers=[]
        )
        
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.language, "en")
        self.assertEqual(result.duration, 2.5)
        self.assertEqual(result.speakers, [])
    
    def test_transcription_result_with_speakers(self):
        """Test TranscriptionResult with speaker diarization."""
        from core.base_transcriber import TranscriptionResult
        
        speakers = [
            {'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 2.0, 'text': 'Hello'},
            {'speaker': 'SPEAKER_01', 'start': 2.5, 'end': 4.0, 'text': 'World'}
        ]
        
        result = TranscriptionResult(
            text="Hello World",
            speakers=speakers
        )
        
        self.assertEqual(len(result.speakers), 2)
        self.assertEqual(result.speakers[0]['speaker'], 'SPEAKER_00')


class TestTranscriberFactory(unittest.TestCase):
    """Test transcriber factory functions."""
    
    def test_import_factory(self):
        """Test that factory can be imported."""
        from core.transcriber_factory import create_transcriber, check_engine_availability
        from core.transcriber_factory import reload_transcriber
        self.assertTrue(True)
    
    def test_check_engine_availability_structure(self):
        """Test that check_engine_availability returns correct structure."""
        from core.transcriber_factory import check_engine_availability
        
        engines = check_engine_availability()
        
        self.assertIsInstance(engines, dict)
        self.assertIn('whispercpp', engines)
        self.assertIn('whisperx', engines)
        self.assertIsInstance(engines['whispercpp'], bool)
        self.assertIsInstance(engines['whisperx'], bool)


class TestWhisperCppTranscriber(unittest.TestCase):
    """Test WhisperCppTranscriber class."""
    
    @patch('core.whispercpp_transcriber.PYWHISPERCPP_AVAILABLE', False)
    def test_initialization_without_pywhispercpp(self):
        """Test that transcriber can be initialized even without pywhispercpp."""
        from core.whispercpp_transcriber import WhisperCppTranscriber
        
        transcoder = WhisperCppTranscriber(
            model_path="/fake/path/model.bin",
            model_name="large-v3-turbo",
            language="en",
            n_threads=4,
        )
        
        self.assertEqual(transcoder.model_name, "large-v3-turbo")
        self.assertEqual(transcoder.language, "en")
        self.assertEqual(transcoder.n_threads, 4)
        self.assertFalse(transcoder.is_ready())
    
    def test_stats_structure(self):
        """Test that get_stats returns expected structure."""
        from core.whispercpp_transcriber import WhisperCppTranscriber
        
        transcoder = WhisperCppTranscriber(model_name="base")
        stats = transcoder.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_transcriptions', stats)
        self.assertIn('model_name', stats)
        self.assertIn('using_gpu', stats)
        self.assertEqual(stats['model_name'], "base")


class TestWhisperXTranscriber(unittest.TestCase):
    """Test WhisperXTranscriber class."""
    
    def test_initialization(self):
        """Test WhisperXTranscriber initialization."""
        from core.whisperx_transcriber import WhisperXTranscriber
        
        transcoder = WhisperXTranscriber(
            model_name="base",
            device="cpu",
            compute_type="int8",
            language="en",
            enable_diarization=False,
            hf_token=None,
        )
        
        self.assertEqual(transcoder.model_name, "base")
        self.assertEqual(transcoder.device, "cpu")
        self.assertEqual(transcoder.compute_type, "int8")
        self.assertEqual(transcoder.language, "en")
        self.assertFalse(transcoder.enable_diarization)
        self.assertFalse(transcoder.is_ready())
    
    def test_stats_structure(self):
        """Test that get_stats returns expected structure."""
        from core.whisperx_transcriber import WhisperXTranscriber
        
        transcoder = WhisperXTranscriber(model_name="base", device="cpu")
        stats = transcoder.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_transcriptions', stats)
        self.assertIn('model_name', stats)
        self.assertIn('diarization_enabled', stats)
        self.assertIn('using_gpu', stats)
        self.assertEqual(stats['model_name'], "base")
        self.assertEqual(stats['diarization_enabled'], False)
    
    def test_get_device_name_cpu(self):
        """Test get_device_name returns CPU when device is cpu."""
        from core.whisperx_transcriber import WhisperXTranscriber
        
        transcoder = WhisperXTranscriber(device="cpu")
        device_name = transcoder.get_device_name()
        
        self.assertEqual(device_name, "CPU")


class TestTranscriberCompatibility(unittest.TestCase):
    """Test that both transcribers implement the same interface."""
    
    def test_both_inherit_from_base(self):
        """Test that both transcribers inherit from BaseTranscriber."""
        from core.base_transcriber import BaseTranscriber
        from core.whispercpp_transcriber import WhisperCppTranscriber
        from core.whisperx_transcriber import WhisperXTranscriber
        
        cpp = WhisperCppTranscriber(model_name="base")
        wx = WhisperXTranscriber(model_name="base", device="cpu")
        
        self.assertIsInstance(cpp, BaseTranscriber)
        self.assertIsInstance(wx, BaseTranscriber)
    
    def test_common_interface_methods(self):
        """Test that both transcribers have the same interface methods."""
        from core.whispercpp_transcriber import WhisperCppTranscriber
        from core.whisperx_transcriber import WhisperXTranscriber
        
        cpp = WhisperCppTranscriber(model_name="base")
        wx = WhisperXTranscriber(model_name="base", device="cpu")
        
        required_methods = ['load_model', 'unload_model', 'transcribe', 
                           'is_ready', 'get_stats', 'get_device_name']
        
        for method in required_methods:
            self.assertTrue(hasattr(cpp, method), f"WhisperCppTranscriber missing {method}")
            self.assertTrue(hasattr(wx, method), f"WhisperXTranscriber missing {method}")
            self.assertTrue(callable(getattr(cpp, method)))
            self.assertTrue(callable(getattr(wx, method)))


class TestBackwardsCompatibility(unittest.TestCase):
    """Test backwards compatibility with old imports."""
    
    def test_old_import_paths(self):
        """Test that old import paths still work."""
        from core.transcriber import Transcriber, TranscriptionResult
        from core.transcriber import create_transcriber, MultiModelTranscriber
        from core.transcriber import PYWHISPERCPP_AVAILABLE, SAMPLE_RATE
        
        self.assertTrue(True)
    
    def test_transcriber_is_whispercpp(self):
        """Test that Transcriber is an alias for WhisperCppTranscriber."""
        from core.transcriber import Transcriber
        from core.whispercpp_transcriber import WhisperCppTranscriber
        
        self.assertIs(Transcriber, WhisperCppTranscriber)
    
    def test_old_interface_works(self):
        """Test that old interface still works."""
        from core.transcriber import Transcriber
        
        transcoder = Transcriber(model_name="base")
        self.assertEqual(transcoder.model_name, "base")


class TestConfigSettings(unittest.TestCase):
    """Test configuration settings."""
    
    def test_whisperx_settings_exist(self):
        """Test that WhisperX settings exist in config."""
        from config.settings import ModelSettings
        
        # Check that attributes exist
        self.assertTrue(hasattr(ModelSettings, '__dataclass_fields__'))
        
        settings = ModelSettings()
        
        # Check WhisperX settings
        self.assertTrue(hasattr(settings, 'asr_engine'))
        self.assertTrue(hasattr(settings, 'whisperx_model'))
        self.assertTrue(hasattr(settings, 'whisperx_compute_type'))
        self.assertTrue(hasattr(settings, 'whisperx_diarize'))
        self.assertTrue(hasattr(settings, 'whisperx_hf_token'))
        self.assertTrue(hasattr(settings, 'whisperx_device'))
        
        # Check defaults
        self.assertEqual(settings.asr_engine, "whispercpp")
        self.assertEqual(settings.whisperx_model, "large-v3")
        self.assertEqual(settings.whisperx_compute_type, "int8")
        self.assertFalse(settings.whisperx_diarize)
        self.assertEqual(settings.whisperx_device, "cuda")


def run_tests():
    """Run all tests and return success status."""
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBaseTranscriber))
    suite.addTests(loader.loadTestsFromTestCase(TestTranscriberFactory))
    suite.addTests(loader.loadTestsFromTestCase(TestWhisperCppTranscriber))
    suite.addTests(loader.loadTestsFromTestCase(TestWhisperXTranscriber))
    suite.addTests(loader.loadTestsFromTestCase(TestTranscriberCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestBackwardsCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigSettings))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
