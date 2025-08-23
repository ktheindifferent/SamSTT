#!/usr/bin/env python3
"""
Simplified unit tests for exception handling that can run without external dependencies.
Tests specifically focus on the fixed bare exception handlers.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import logging

# Mock external dependencies before importing
sys.modules['numpy'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['ffmpeg'] = MagicMock()
sys.modules['sanic'] = MagicMock()
sys.modules['sanic.exceptions'] = MagicMock()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stts.base_engine import BaseSTTEngine
from stts.engine import SpeechToTextEngine


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
        self.original_logger = logging.getLogger('stts.base_engine')
        self.original_level = self.original_logger.level
        self.original_logger.setLevel(logging.DEBUG)
        
        # Create custom handler to capture logs
        class LogCapture(logging.Handler):
            def __init__(self, capture_list):
                super().__init__()
                self.capture_list = capture_list
            
            def emit(self, record):
                self.capture_list.append(record)
        
        self.handler = LogCapture(self.log_capture)
        self.original_logger.addHandler(self.handler)
    
    def tearDown(self):
        self.original_logger.removeHandler(self.handler)
        self.original_logger.setLevel(self.original_level)
    
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
        self.original_logger = logging.getLogger('stts.engine')
        self.original_level = self.original_logger.level
        self.original_logger.setLevel(logging.DEBUG)
        
        # Create custom handler to capture logs
        class LogCapture(logging.Handler):
            def __init__(self, capture_list):
                super().__init__()
                self.capture_list = capture_list
            
            def emit(self, record):
                self.capture_list.append(record)
        
        self.handler = LogCapture(self.log_capture)
        self.original_logger.addHandler(self.handler)
    
    def tearDown(self):
        self.original_logger.removeHandler(self.handler)
        self.original_logger.setLevel(self.original_level)
    
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
        # No AttributeError should be logged for this case since hasattr handles it
    
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


class TestFailureScenarios(unittest.TestCase):
    """Test various failure scenarios with the fixed exception handling"""
    
    def test_memory_error_handling(self):
        """Test handling of memory errors during model loading"""
        engine = TestEngine()
        engine._check_availability = Mock(side_effect=MemoryError("Out of memory"))
        
        result = engine.is_available
        
        # MemoryError should be caught by the generic Exception handler
        self.assertFalse(result)
    
    def test_permission_error_handling(self):
        """Test handling of permission errors"""
        engine = TestEngine()
        engine._check_availability = Mock(side_effect=PermissionError("Access denied"))
        
        result = engine.is_available
        
        # PermissionError is a subclass of OSError, should be caught
        self.assertFalse(result)
    
    def test_keyboard_interrupt_not_caught(self):
        """Test that KeyboardInterrupt is not caught (system exceptions should propagate)"""
        engine = TestEngine()
        engine._check_availability = Mock(side_effect=KeyboardInterrupt())
        
        # KeyboardInterrupt should NOT be caught as it derives from BaseException, not Exception
        with self.assertRaises(KeyboardInterrupt):
            _ = engine.is_available
    
    def test_system_exit_not_caught(self):
        """Test that SystemExit is not caught (system exceptions should propagate)"""
        engine = TestEngine()
        engine._check_availability = Mock(side_effect=SystemExit())
        
        # SystemExit should NOT be caught as it derives from BaseException, not Exception
        with self.assertRaises(SystemExit):
            _ = engine.is_available


if __name__ == '__main__':
    # Run tests with verbose output
    print("Running exception handling tests...")
    print("=" * 70)
    unittest.main(verbosity=2)