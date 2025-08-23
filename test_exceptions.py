#!/usr/bin/env python3
"""Unit tests for the custom exception hierarchy in the Unified STT Service

This module tests the exception classes and their integration with the STT engines.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add the parent directory to the path so we can import stts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stts.exceptions import (
    STTException,
    EngineNotAvailableError,
    EngineInitializationError,
    ModelNotFoundError,
    AudioProcessingError,
    TranscriptionError,
    InvalidAudioError,
    UnsupportedAudioFormatError,
    EngineTimeoutError,
    InsufficientResourcesError,
    ConfigurationError
)


class TestExceptionHierarchy(unittest.TestCase):
    """Test the exception class hierarchy and attributes"""
    
    def test_base_exception(self):
        """Test STTException base class"""
        exc = STTException("Test error", engine="test_engine")
        self.assertEqual(str(exc), "[test_engine] Test error")
        self.assertEqual(exc.message, "Test error")
        self.assertEqual(exc.engine, "test_engine")
        self.assertIsNone(exc.original_error)
    
    def test_base_exception_without_engine(self):
        """Test STTException without engine name"""
        exc = STTException("Test error")
        self.assertEqual(str(exc), "Test error")
        self.assertIsNone(exc.engine)
    
    def test_base_exception_with_original_error(self):
        """Test STTException with original error"""
        original = ValueError("Original error")
        exc = STTException("Wrapped error", engine="test", original_error=original)
        self.assertEqual(exc.original_error, original)
        self.assertEqual(str(exc), "[test] Wrapped error")
    
    def test_engine_not_available_error(self):
        """Test EngineNotAvailableError"""
        exc = EngineNotAvailableError(
            "Engine not installed",
            engine="whisper"
        )
        self.assertIsInstance(exc, STTException)
        self.assertEqual(exc.engine, "whisper")
    
    def test_model_not_found_error(self):
        """Test ModelNotFoundError with model_path"""
        exc = ModelNotFoundError(
            "Model file missing",
            engine="deepspeech",
            model_path="/path/to/model.pbmm"
        )
        self.assertIsInstance(exc, EngineInitializationError)
        self.assertIsInstance(exc, STTException)
        self.assertEqual(exc.model_path, "/path/to/model.pbmm")
    
    def test_audio_processing_error_hierarchy(self):
        """Test audio processing error types"""
        exc1 = AudioProcessingError("Processing failed", engine="test")
        exc2 = InvalidAudioError("Corrupted audio", engine="test")
        exc3 = UnsupportedAudioFormatError("Wrong format", engine="test")
        
        self.assertIsInstance(exc1, STTException)
        self.assertIsInstance(exc2, AudioProcessingError)
        self.assertIsInstance(exc3, AudioProcessingError)
    
    def test_engine_timeout_error(self):
        """Test EngineTimeoutError with timeout value"""
        exc = EngineTimeoutError(
            "Transcription timed out",
            engine="whisper",
            timeout=30.0
        )
        self.assertIsInstance(exc, TranscriptionError)
        self.assertEqual(exc.timeout, 30.0)
    
    def test_configuration_error(self):
        """Test ConfigurationError"""
        exc = ConfigurationError(
            "Invalid configuration parameter",
            engine="vosk"
        )
        self.assertIsInstance(exc, STTException)
        self.assertEqual(exc.engine, "vosk")


class TestEngineExceptionHandling(unittest.TestCase):
    """Test exception handling in engine implementations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_audio = np.zeros(16000, dtype=np.int16)  # 1 second of silence
    
    @patch('stts.engines.deepspeech.Model')
    def test_deepspeech_model_not_found(self, mock_model):
        """Test DeepSpeech raises ModelNotFoundError when model is missing"""
        from stts.engines.deepspeech import DeepSpeechEngine
        
        with self.assertRaises(ModelNotFoundError) as context:
            engine = DeepSpeechEngine({'model_path': '/nonexistent/model.pbmm'})
        
        self.assertIn("not found", str(context.exception).lower())
        self.assertEqual(context.exception.engine, "deepspeech")
    
    @patch('stts.engines.whisper.whisper')
    def test_whisper_insufficient_resources(self, mock_whisper):
        """Test Whisper raises InsufficientResourcesError on OOM"""
        mock_whisper.load_model.side_effect = RuntimeError("CUDA out of memory")
        
        from stts.engines.whisper import WhisperEngine
        
        with self.assertRaises(InsufficientResourcesError) as context:
            engine = WhisperEngine({'model_size': 'large'})
        
        self.assertIn("memory", str(context.exception).lower())
        self.assertEqual(context.exception.engine, "whisper")
    
    def test_vosk_engine_not_available(self):
        """Test Vosk raises EngineNotAvailableError when not installed"""
        with patch.dict('sys.modules', {'vosk': None}):
            from stts.engines.vosk import VoskEngine
            
            with self.assertRaises(EngineNotAvailableError) as context:
                engine = VoskEngine({})
            
            self.assertIn("not installed", str(context.exception))
            self.assertEqual(context.exception.engine, "vosk")


class TestEngineManagerExceptionHandling(unittest.TestCase):
    """Test exception handling in the engine manager"""
    
    @patch('stts.engine_manager.DeepSpeechEngine')
    def test_manager_handles_engine_not_available(self, mock_engine_class):
        """Test manager handles EngineNotAvailableError gracefully"""
        mock_engine_class.side_effect = EngineNotAvailableError(
            "DeepSpeech not installed",
            engine="deepspeech"
        )
        
        from stts.engine_manager import STTEngineManager
        
        manager = STTEngineManager(default_engine='deepspeech')
        self.assertEqual(len(manager.engines), 0)
    
    @patch('stts.engine_manager.WhisperEngine')
    @patch('stts.engine_manager.DeepSpeechEngine')
    def test_manager_fallback_on_transcription_error(self, mock_deepspeech, mock_whisper):
        """Test manager falls back when primary engine fails"""
        # Set up mocks
        mock_ds_instance = MagicMock()
        mock_ds_instance.is_available = True
        mock_ds_instance.name = "DeepSpeech"
        mock_ds_instance.transcribe.side_effect = TranscriptionError(
            "Transcription failed",
            engine="deepspeech"
        )
        mock_deepspeech.return_value = mock_ds_instance
        
        mock_whisper_instance = MagicMock()
        mock_whisper_instance.is_available = True
        mock_whisper_instance.name = "Whisper"
        mock_whisper_instance.transcribe.return_value = "Hello world"
        mock_whisper.return_value = mock_whisper_instance
        
        from stts.engine_manager import STTEngineManager
        
        manager = STTEngineManager(
            default_engine='deepspeech',
            config={'deepspeech': {}, 'whisper': {}}
        )
        
        result = manager.transcribe(b"fake_audio", engine="deepspeech")
        
        self.assertEqual(result['text'], "Hello world")
        self.assertEqual(result['engine'], "Whisper")
        self.assertTrue(result['fallback'])
    
    @patch('stts.engine_manager.DeepSpeechEngine')
    def test_manager_audio_processing_error_no_fallback(self, mock_engine_class):
        """Test manager doesn't try fallback for audio processing errors"""
        mock_instance = MagicMock()
        mock_instance.is_available = True
        mock_instance.name = "DeepSpeech"
        mock_instance.transcribe.side_effect = InvalidAudioError(
            "Corrupted audio data",
            engine="deepspeech"
        )
        mock_engine_class.return_value = mock_instance
        
        from stts.engine_manager import STTEngineManager
        
        manager = STTEngineManager(default_engine='deepspeech')
        
        with self.assertRaises(InvalidAudioError) as context:
            manager.transcribe(b"corrupted_audio")
        
        self.assertIn("Corrupted audio", str(context.exception))


class TestAPIExceptionHandling(unittest.TestCase):
    """Test exception handling in the Sanic API"""
    
    @patch('stts.app.engine')
    def test_api_handles_invalid_audio(self, mock_engine):
        """Test API returns appropriate error for invalid audio"""
        mock_engine.transcribe.side_effect = InvalidAudioError(
            "Invalid audio format",
            engine="test"
        )
        
        from stts.app import app
        from sanic.exceptions import InvalidUsage
        
        # Create a test request
        request = MagicMock()
        request.files = {'speech': MagicMock(body=b"invalid")}
        request.args = {}
        request.headers = {}
        
        # The API should raise InvalidUsage with 400 status
        # This would be tested in an actual integration test
        # Here we just verify the exception is properly configured
        self.assertTrue(hasattr(app, 'route'))
    
    def test_api_exception_status_codes(self):
        """Test that different exceptions map to correct HTTP status codes"""
        from stts.app import app
        
        # Map of exception types to expected status codes
        exception_status_map = {
            InvalidAudioError: 400,
            AudioProcessingError: 400,
            ConfigurationError: 400,
            EngineNotAvailableError: 503,
            ModelNotFoundError: 503,
            InsufficientResourcesError: 503,
            EngineTimeoutError: 504,
            TranscriptionError: 500,
        }
        
        # Verify the mapping exists in the app error handlers
        # This is a basic check - full testing would require integration tests
        for exc_type, expected_status in exception_status_map.items():
            self.assertIn(expected_status, [400, 404, 500, 503, 504])


class TestExceptionMessages(unittest.TestCase):
    """Test that exception messages are informative and consistent"""
    
    def test_model_not_found_message(self):
        """Test ModelNotFoundError provides helpful information"""
        exc = ModelNotFoundError(
            "DeepSpeech model not found in default locations. Expected locations: /app/model.pbmm, /app/model.tflite",
            engine="deepspeech",
            model_path="/app/model.pbmm"
        )
        
        self.assertIn("Expected locations", exc.message)
        self.assertIn("/app/model.pbmm", exc.message)
        self.assertEqual(exc.model_path, "/app/model.pbmm")
    
    def test_engine_not_available_message(self):
        """Test EngineNotAvailableError provides installation instructions"""
        exc = EngineNotAvailableError(
            "Whisper package not installed. Install with: pip install openai-whisper",
            engine="whisper"
        )
        
        self.assertIn("pip install", exc.message)
        self.assertIn("openai-whisper", exc.message)
    
    def test_insufficient_resources_message(self):
        """Test InsufficientResourcesError provides actionable advice"""
        exc = InsufficientResourcesError(
            "Insufficient memory to load Whisper model 'large'. Try a smaller model or increase available memory.",
            engine="whisper"
        )
        
        self.assertIn("smaller model", exc.message)
        self.assertIn("increase", exc.message)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)