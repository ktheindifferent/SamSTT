#!/usr/bin/env python3
"""Basic tests for the custom exception hierarchy - no external dependencies"""

import sys
import os

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


def test_exception_hierarchy():
    """Test the exception class hierarchy"""
    print("Testing exception hierarchy...")
    
    # Test base exception
    exc = STTException("Test error", engine="test_engine")
    assert str(exc) == "[test_engine] Test error"
    assert exc.message == "Test error"
    assert exc.engine == "test_engine"
    assert exc.original_error is None
    print("✓ Base STTException works correctly")
    
    # Test exception without engine
    exc = STTException("Test error")
    assert str(exc) == "Test error"
    assert exc.engine is None
    print("✓ STTException without engine works")
    
    # Test with original error
    original = ValueError("Original error")
    exc = STTException("Wrapped error", engine="test", original_error=original)
    assert exc.original_error == original
    print("✓ Original error preservation works")
    
    # Test inheritance
    exc = EngineNotAvailableError("Not available", engine="test")
    assert isinstance(exc, STTException)
    print("✓ EngineNotAvailableError inherits from STTException")
    
    exc = ModelNotFoundError("Model missing", model_path="/path/to/model", engine="test")
    assert isinstance(exc, EngineInitializationError)
    assert isinstance(exc, STTException)
    assert exc.model_path == "/path/to/model"
    print("✓ ModelNotFoundError hierarchy and attributes work")
    
    # Test audio error hierarchy
    exc1 = InvalidAudioError("Bad audio", engine="test")
    exc2 = UnsupportedAudioFormatError("Wrong format", engine="test")
    assert isinstance(exc1, AudioProcessingError)
    assert isinstance(exc2, AudioProcessingError)
    assert isinstance(exc1, STTException)
    print("✓ Audio processing error hierarchy works")
    
    # Test timeout error
    exc = EngineTimeoutError("Timeout", engine="test", timeout=30.0)
    assert isinstance(exc, TranscriptionError)
    assert exc.timeout == 30.0
    print("✓ EngineTimeoutError works with timeout attribute")
    
    # Test configuration error
    exc = ConfigurationError("Bad config", engine="test")
    assert isinstance(exc, STTException)
    print("✓ ConfigurationError works")
    
    # Test insufficient resources
    exc = InsufficientResourcesError("Out of memory", engine="test")
    assert isinstance(exc, STTException)
    print("✓ InsufficientResourcesError works")
    
    print("\n✅ All exception hierarchy tests passed!")


def test_exception_messages():
    """Test that exception messages are informative"""
    print("\nTesting exception messages...")
    
    # Test model not found message
    exc = ModelNotFoundError(
        "Model not found in: /app/model.pbmm, /app/model.tflite",
        engine="deepspeech",
        model_path="/app/model.pbmm"
    )
    assert "/app/model.pbmm" in exc.message
    assert exc.model_path == "/app/model.pbmm"
    print("✓ ModelNotFoundError message is informative")
    
    # Test engine not available message
    exc = EngineNotAvailableError(
        "Whisper not installed. Install with: pip install openai-whisper",
        engine="whisper"
    )
    assert "pip install" in exc.message
    assert "openai-whisper" in exc.message
    print("✓ EngineNotAvailableError provides installation instructions")
    
    # Test insufficient resources message
    exc = InsufficientResourcesError(
        "Insufficient memory for model 'large'. Try a smaller model.",
        engine="whisper"
    )
    assert "smaller model" in exc.message
    print("✓ InsufficientResourcesError provides actionable advice")
    
    print("\n✅ All message tests passed!")


def test_exception_string_representation():
    """Test string representation of exceptions"""
    print("\nTesting string representations...")
    
    # With engine
    exc = TranscriptionError("Failed to transcribe", engine="whisper")
    assert str(exc) == "[whisper] Failed to transcribe"
    print("✓ String representation with engine works")
    
    # Without engine
    exc = AudioProcessingError("Audio error")
    assert str(exc) == "Audio error"
    print("✓ String representation without engine works")
    
    # Nested with engine
    original = RuntimeError("CUDA OOM")
    exc = InsufficientResourcesError(
        "Out of GPU memory",
        engine="whisper",
        original_error=original
    )
    assert str(exc) == "[whisper] Out of GPU memory"
    assert exc.original_error == original
    print("✓ Nested exception representation works")
    
    print("\n✅ All string representation tests passed!")


def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING CUSTOM EXCEPTION HIERARCHY")
    print("=" * 60)
    
    try:
        test_exception_hierarchy()
        test_exception_messages()
        test_exception_string_representation()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())