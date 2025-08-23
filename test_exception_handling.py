#!/usr/bin/env python3
"""
Unit tests for exception handling in the STT service.
Tests specifically focus on the fixed bare exception handlers.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import logging
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stts.base_engine import BaseSTTEngine
from stts.engine import SpeechToTextEngine
from stts.engine_manager import STTEngineManager


class TestEngine(BaseSTTEngine):
    """Test implementation of BaseSTTEngine for testing"""
    
    def initialize(self):
        self.model = Mock()
    
    def transcribe_raw(self, audio_data, sample_rate=16000):
        return "test transcription"
    
    def _check_availability(self):
        # This can be overridden in tests to raise specific exceptions
        return True


class TestBaseEngineExceptionHandling(unittest.TestCase):
    """Test exception handling in BaseSTTEngine"""
    
    def setUp(self):
        # Configure logging to capture messages
        self.log_capture = []
        self.handler = logging.Handler()
        self.handler.emit = lambda record: self.log_capture.append(record)
        logging.getLogger('stts.base_engine').addHandler(self.handler)
        logging.getLogger('stts.base_engine').setLevel(logging.DEBUG)
    
    def tearDown(self):
        logging.getLogger('stts.base_engine').removeHandler(self.handler)
    
    def test_is_available_import_error(self):
        """Test handling of ImportError in is_available property"""
        engine = TestEngine()
        engine._check_availability = Mock(side_effect=ImportError("test module not found"))
        
        result = engine.is_available
        
        self.assertFalse(result)
        # Check that debug log was written
        self.assertTrue(any("dependency not available" in str(record.getMessage()) 
                           for record in self.log_capture))
    
    def test_is_available_module_not_found_error(self):
        """Test handling of ModuleNotFoundError in is_available property"""
        engine = TestEngine()
        engine._check_availability = Mock(side_effect=ModuleNotFoundError("whisper"))
        
        result = engine.is_available
        
        self.assertFalse(result)
        self.assertTrue(any("dependency not available" in str(record.getMessage()) 
                           for record in self.log_capture))
    
    def test_is_available_file_not_found_error(self):
        """Test handling of FileNotFoundError in is_available property"""
        engine = TestEngine()
        engine._check_availability = Mock(side_effect=FileNotFoundError("model.pb"))
        
        result = engine.is_available
        
        self.assertFalse(result)
        self.assertTrue(any("model file error" in str(record.getMessage()) 
                           for record in self.log_capture))
    
    def test_is_available_os_error(self):
        """Test handling of OSError in is_available property"""
        engine = TestEngine()
        engine._check_availability = Mock(side_effect=OSError("Permission denied"))
        
        result = engine.is_available
        
        self.assertFalse(result)
        self.assertTrue(any("model file error" in str(record.getMessage()) 
                           for record in self.log_capture))
    
    def test_is_available_attribute_error(self):
        """Test handling of AttributeError in is_available property"""
        engine = TestEngine()
        engine._check_availability = Mock(side_effect=AttributeError("'NoneType' object has no attribute 'model'"))
        
        result = engine.is_available
        
        self.assertFalse(result)
        self.assertTrue(any("configuration error" in str(record.getMessage()) 
                           for record in self.log_capture))
    
    def test_is_available_type_error(self):
        """Test handling of TypeError in is_available property"""
        engine = TestEngine()
        engine._check_availability = Mock(side_effect=TypeError("Expected str, got int"))
        
        result = engine.is_available
        
        self.assertFalse(result)
        self.assertTrue(any("configuration error" in str(record.getMessage()) 
                           for record in self.log_capture))
    
    def test_is_available_value_error(self):
        """Test handling of ValueError in is_available property"""
        engine = TestEngine()
        engine._check_availability = Mock(side_effect=ValueError("Invalid model size"))
        
        result = engine.is_available
        
        self.assertFalse(result)
        self.assertTrue(any("configuration error" in str(record.getMessage()) 
                           for record in self.log_capture))
    
    def test_is_available_runtime_error(self):
        """Test handling of RuntimeError in is_available property"""
        engine = TestEngine()
        engine._check_availability = Mock(side_effect=RuntimeError("CUDA out of memory"))
        
        result = engine.is_available
        
        self.assertFalse(result)
        self.assertTrue(any("runtime error" in str(record.getMessage()) 
                           for record in self.log_capture))
    
    def test_is_available_unexpected_exception(self):
        """Test handling of unexpected exceptions in is_available property"""
        class CustomException(Exception):
            pass
        
        engine = TestEngine()
        engine._check_availability = Mock(side_effect=CustomException("Unexpected error"))
        
        result = engine.is_available
        
        self.assertFalse(result)
        # Should log as warning for unexpected exceptions
        self.assertTrue(any(record.levelname == "WARNING" and "unexpected error" in str(record.getMessage()).lower()
                           for record in self.log_capture))
        self.assertTrue(any("CustomException" in str(record.getMessage())
                           for record in self.log_capture))
    
    def test_is_available_success(self):
        """Test successful availability check"""
        engine = TestEngine()
        engine._check_availability = Mock(return_value=True)
        
        result = engine.is_available
        
        self.assertTrue(result)
        # No error logs should be written
        self.assertFalse(any(record.levelname in ["ERROR", "WARNING"] 
                            for record in self.log_capture))


class TestSpeechToTextEngineExceptionHandling(unittest.TestCase):
    """Test exception handling in SpeechToTextEngine"""
    
    def setUp(self):
        # Configure logging to capture messages
        self.log_capture = []
        self.handler = logging.Handler()
        self.handler.emit = lambda record: self.log_capture.append(record)
        logging.getLogger('stts.engine').addHandler(self.handler)
        logging.getLogger('stts.engine').setLevel(logging.DEBUG)
    
    def tearDown(self):
        logging.getLogger('stts.engine').removeHandler(self.handler)
    
    @patch('stts.engine.STTEngineManager')
    def test_setup_legacy_support_value_error(self, mock_manager_class):
        """Test handling of ValueError in _setup_legacy_support"""
        mock_manager = Mock()
        mock_manager.get_engine.side_effect = ValueError("Unknown engine: invalid")
        mock_manager_class.return_value = mock_manager
        
        # Create engine - should handle the exception in _setup_legacy_support
        engine = SpeechToTextEngine(engine_name='invalid')
        
        # Should not have model attribute
        self.assertFalse(hasattr(engine, 'model'))
        # Check that debug log was written
        self.assertTrue(any("Could not setup legacy model attribute" in str(record.getMessage()) 
                           for record in self.log_capture))
    
    @patch('stts.engine.STTEngineManager')
    def test_setup_legacy_support_attribute_error(self, mock_manager_class):
        """Test handling of AttributeError in _setup_legacy_support"""
        mock_manager = Mock()
        mock_engine = Mock(spec=[])  # No 'model' attribute
        mock_manager.get_engine.return_value = mock_engine
        mock_manager_class.return_value = mock_manager
        
        # Create engine - should handle the exception in _setup_legacy_support
        engine = SpeechToTextEngine()
        
        # Should not have model attribute
        self.assertFalse(hasattr(engine, 'model'))
        # Check that debug log was written
        self.assertTrue(any("doesn't have model attribute" in str(record.getMessage()) 
                           for record in self.log_capture))
    
    @patch('stts.engine.STTEngineManager')
    def test_setup_legacy_support_unexpected_exception(self, mock_manager_class):
        """Test handling of unexpected exceptions in _setup_legacy_support"""
        class CustomException(Exception):
            pass
        
        mock_manager = Mock()
        mock_manager.get_engine.side_effect = CustomException("Unexpected error")
        mock_manager_class.return_value = mock_manager
        
        # Create engine - should handle the exception in _setup_legacy_support
        engine = SpeechToTextEngine()
        
        # Should not have model attribute
        self.assertFalse(hasattr(engine, 'model'))
        # Check that debug log was written
        self.assertTrue(any("Unexpected error during legacy support setup" in str(record.getMessage()) 
                           for record in self.log_capture))
        self.assertTrue(any("CustomException" in str(record.getMessage())
                           for record in self.log_capture))
    
    @patch('stts.engine.STTEngineManager')
    def test_setup_legacy_support_success(self, mock_manager_class):
        """Test successful legacy support setup"""
        mock_manager = Mock()
        mock_engine = Mock()
        mock_engine.model = Mock()
        mock_manager.get_engine.return_value = mock_engine
        mock_manager_class.return_value = mock_manager
        
        # Create engine - should successfully set up legacy support
        engine = SpeechToTextEngine()
        
        # Should have model attribute
        self.assertTrue(hasattr(engine, 'model'))
        self.assertEqual(engine.model, mock_engine.model)
        # No error logs should be written
        self.assertFalse(any(record.levelname in ["ERROR", "WARNING"] 
                            for record in self.log_capture))


class TestFailureScenarios(unittest.TestCase):
    """Test various failure scenarios with the fixed exception handling"""
    
    def test_corrupted_audio_handling(self):
        """Test handling of corrupted audio data"""
        engine = TestEngine()
        
        # Mock normalize_audio to simulate corrupted audio
        with patch.object(engine, 'normalize_audio', side_effect=Exception("Invalid audio format")):
            with self.assertRaises(Exception) as context:
                engine.transcribe(b"corrupted_data")
            
            self.assertIn("Invalid audio format", str(context.exception))
    
    @patch('stts.engine.Path')
    def test_missing_model_file_handling(self, mock_path):
        """Test handling of missing model files"""
        # Simulate no model files existing
        mock_path.return_value.exists.return_value = False
        
        with patch('stts.engine.STTEngineManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.get_engine.side_effect = ValueError("No model file found")
            mock_manager_class.return_value = mock_manager
            
            # Should handle the missing model gracefully
            engine = SpeechToTextEngine()
            self.assertFalse(hasattr(engine, 'model'))
    
    def test_memory_error_handling(self):
        """Test handling of memory errors during model loading"""
        engine = TestEngine()
        engine._check_availability = Mock(side_effect=MemoryError("Out of memory"))
        
        result = engine.is_available
        
        self.assertFalse(result)
    
    def test_permission_error_handling(self):
        """Test handling of permission errors"""
        engine = TestEngine()
        engine._check_availability = Mock(side_effect=PermissionError("Access denied"))
        
        result = engine.is_available
        
        self.assertFalse(result)
    
    def test_keyboard_interrupt_not_caught(self):
        """Test that KeyboardInterrupt is not caught (system exceptions should propagate)"""
        engine = TestEngine()
        engine._check_availability = Mock(side_effect=KeyboardInterrupt())
        
        with self.assertRaises(KeyboardInterrupt):
            _ = engine.is_available
    
    def test_system_exit_not_caught(self):
        """Test that SystemExit is not caught (system exceptions should propagate)"""
        engine = TestEngine()
        engine._check_availability = Mock(side_effect=SystemExit())
        
        with self.assertRaises(SystemExit):
            _ = engine.is_available


class TestEngineManagerIntegration(unittest.TestCase):
    """Integration tests for engine manager with exception handling"""
    
    @patch('stts.engine_manager.DeepSpeechEngine')
    def test_engine_initialization_failure_handling(self, mock_engine_class):
        """Test that engine initialization failures are properly handled"""
        # Simulate engine initialization failure
        mock_engine_class.side_effect = ImportError("stt module not found")
        
        # Manager should handle the failure gracefully
        manager = STTEngineManager(default_engine='deepspeech', config={'deepspeech': {}})
        
        # Engine should not be available
        self.assertNotIn('deepspeech', manager.list_available_engines())
    
    @patch('stts.engine_manager.WhisperEngine')
    def test_fallback_on_engine_failure(self, mock_whisper_class):
        """Test fallback mechanism when primary engine fails"""
        # Create a working test engine
        test_engine = TestEngine()
        
        # Create manager with test engine as fallback
        manager = STTEngineManager(default_engine='whisper')
        manager.add_engine('test', test_engine)
        
        # Mock whisper to fail
        mock_whisper = Mock()
        mock_whisper.transcribe.side_effect = Exception("Whisper failed")
        mock_whisper.name = "Whisper"
        mock_whisper.is_available = True
        mock_whisper_class.return_value = mock_whisper
        manager.engines['whisper'] = mock_whisper
        
        # Transcription should fall back to test engine
        result = manager.transcribe(b"test_audio")
        
        self.assertEqual(result['engine'], 'Test')
        self.assertTrue(result.get('fallback', False))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)