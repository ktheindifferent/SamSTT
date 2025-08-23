#!/usr/bin/env python3
"""
Test script to verify exception handling with various failure scenarios.
This script simulates different types of failures to ensure the system handles them gracefully.
"""

import sys
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

# Mock external dependencies
sys.modules['numpy'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['ffmpeg'] = MagicMock()
sys.modules['sanic'] = MagicMock()
sys.modules['sanic.exceptions'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torchaudio'] = MagicMock()
sys.modules['omegaconf'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['librosa'] = MagicMock()
sys.modules['speechbrain'] = MagicMock()
sys.modules['nemo_toolkit'] = MagicMock()
sys.modules['pocketsphinx'] = MagicMock()
sys.modules['vosk'] = MagicMock()
sys.modules['stt'] = MagicMock()
sys.modules['STT'] = MagicMock()
sys.modules['whisper'] = MagicMock()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stts.base_engine import BaseSTTEngine
from stts.engine import SpeechToTextEngine
from stts.engine_manager import STTEngineManager


def test_corrupted_audio():
    """Test handling of corrupted audio data"""
    print("\n1. Testing corrupted audio handling...")
    
    class TestEngine(BaseSTTEngine):
        def initialize(self):
            pass
        
        def transcribe_raw(self, audio_data, sample_rate=16000):
            return "test"
    
    engine = TestEngine()
    
    # Mock ffmpeg to fail
    with patch('stts.base_engine.ffmpeg') as mock_ffmpeg:
        mock_ffmpeg.input.side_effect = Exception("Corrupted audio format")
        
        try:
            engine.transcribe(b"corrupted_data")
            print("   ❌ Failed: Should have raised exception for corrupted audio")
        except Exception as e:
            if "Corrupted audio format" in str(e):
                print("   ✅ Passed: Corrupted audio properly raised exception")
            else:
                print(f"   ❌ Failed: Unexpected exception: {e}")


def test_missing_model_files():
    """Test handling when model files are missing"""
    print("\n2. Testing missing model files...")
    
    # Create a temporary config with non-existent model paths
    config = {
        'deepspeech': {'model_path': '/non/existent/model.pb'},
        'vosk': {'model_path': '/non/existent/vosk_model'},
        'coqui': {'model_path': '/non/existent/coqui_model.pb'}
    }
    
    with patch('stts.engine.Path') as mock_path:
        mock_path.return_value.exists.return_value = False
        
        try:
            engine = SpeechToTextEngine(config=config)
            
            # Check that engine was created but model attribute is not set
            if not hasattr(engine, 'model'):
                print("   ✅ Passed: Engine created without model attribute when files missing")
            else:
                print("   ❌ Failed: Engine should not have model attribute")
                
        except Exception as e:
            print(f"   ❌ Failed: Should handle missing models gracefully: {e}")


def test_import_errors():
    """Test handling when dependencies are not installed"""
    print("\n3. Testing import error handling...")
    
    class TestEngineWithImportError(BaseSTTEngine):
        def initialize(self):
            import non_existent_module  # This will raise ImportError
        
        def transcribe_raw(self, audio_data, sample_rate=16000):
            return "test"
        
        def _check_availability(self):
            import non_existent_module  # This will raise ImportError
            return True
    
    try:
        engine = TestEngineWithImportError()
        is_available = engine.is_available
        
        if not is_available:
            print("   ✅ Passed: Import errors properly handled in availability check")
        else:
            print("   ❌ Failed: Should return False when imports fail")
            
    except ImportError:
        print("   ❌ Failed: ImportError should be caught, not propagated")


def test_memory_errors():
    """Test handling of memory errors"""
    print("\n4. Testing memory error handling...")
    
    class TestEngineWithMemoryError(BaseSTTEngine):
        def initialize(self):
            pass
        
        def transcribe_raw(self, audio_data, sample_rate=16000):
            raise MemoryError("Out of memory")
        
        def _check_availability(self):
            raise MemoryError("Out of memory during initialization")
    
    engine = TestEngineWithMemoryError()
    
    # Test availability check
    is_available = engine.is_available
    if not is_available:
        print("   ✅ Passed: Memory errors handled in availability check")
    else:
        print("   ❌ Failed: Should return False on memory error")


def test_permission_errors():
    """Test handling of permission errors"""
    print("\n5. Testing permission error handling...")
    
    class TestEngineWithPermissionError(BaseSTTEngine):
        def initialize(self):
            pass
        
        def transcribe_raw(self, audio_data, sample_rate=16000):
            return "test"
        
        def _check_availability(self):
            raise PermissionError("Access denied to model file")
    
    engine = TestEngineWithPermissionError()
    is_available = engine.is_available
    
    if not is_available:
        print("   ✅ Passed: Permission errors properly handled")
    else:
        print("   ❌ Failed: Should return False on permission error")


def test_runtime_errors():
    """Test handling of runtime errors (e.g., CUDA issues)"""
    print("\n6. Testing runtime error handling...")
    
    class TestEngineWithRuntimeError(BaseSTTEngine):
        def initialize(self):
            pass
        
        def transcribe_raw(self, audio_data, sample_rate=16000):
            return "test"
        
        def _check_availability(self):
            raise RuntimeError("CUDA out of memory")
    
    engine = TestEngineWithRuntimeError()
    is_available = engine.is_available
    
    if not is_available:
        print("   ✅ Passed: Runtime errors properly handled")
    else:
        print("   ❌ Failed: Should return False on runtime error")


def test_configuration_errors():
    """Test handling of configuration errors"""
    print("\n7. Testing configuration error handling...")
    
    test_errors = [
        AttributeError("'NoneType' object has no attribute 'model'"),
        TypeError("Expected string, got int"),
        ValueError("Invalid model size: extra-large")
    ]
    
    all_passed = True
    for error in test_errors:
        class TestEngineWithConfigError(BaseSTTEngine):
            def initialize(self):
                pass
            
            def transcribe_raw(self, audio_data, sample_rate=16000):
                return "test"
            
            def _check_availability(self):
                raise error
        
        engine = TestEngineWithConfigError()
        is_available = engine.is_available
        
        if not is_available:
            print(f"   ✅ Passed: {error.__class__.__name__} properly handled")
        else:
            print(f"   ❌ Failed: Should return False on {error.__class__.__name__}")
            all_passed = False
    
    return all_passed


def test_unexpected_exceptions():
    """Test handling of unexpected exceptions"""
    print("\n8. Testing unexpected exception handling...")
    
    class UnexpectedException(Exception):
        """Custom unexpected exception"""
        pass
    
    class TestEngineWithUnexpectedError(BaseSTTEngine):
        def initialize(self):
            pass
        
        def transcribe_raw(self, audio_data, sample_rate=16000):
            return "test"
        
        def _check_availability(self):
            raise UnexpectedException("Something unexpected happened")
    
    engine = TestEngineWithUnexpectedError()
    is_available = engine.is_available
    
    if not is_available:
        print("   ✅ Passed: Unexpected exceptions properly handled")
    else:
        print("   ❌ Failed: Should return False on unexpected exception")


def test_system_exceptions_propagation():
    """Test that system exceptions (KeyboardInterrupt, SystemExit) are NOT caught"""
    print("\n9. Testing system exception propagation...")
    
    # Test KeyboardInterrupt
    class TestEngineWithKeyboardInterrupt(BaseSTTEngine):
        def initialize(self):
            pass
        
        def transcribe_raw(self, audio_data, sample_rate=16000):
            return "test"
        
        def _check_availability(self):
            raise KeyboardInterrupt()
    
    engine = TestEngineWithKeyboardInterrupt()
    
    try:
        _ = engine.is_available
        print("   ❌ Failed: KeyboardInterrupt should NOT be caught")
    except KeyboardInterrupt:
        print("   ✅ Passed: KeyboardInterrupt properly propagated")
    
    # Test SystemExit
    class TestEngineWithSystemExit(BaseSTTEngine):
        def initialize(self):
            pass
        
        def transcribe_raw(self, audio_data, sample_rate=16000):
            return "test"
        
        def _check_availability(self):
            raise SystemExit()
    
    engine = TestEngineWithSystemExit()
    
    try:
        _ = engine.is_available
        print("   ❌ Failed: SystemExit should NOT be caught")
    except SystemExit:
        print("   ✅ Passed: SystemExit properly propagated")


def test_legacy_support_errors():
    """Test error handling in legacy support setup"""
    print("\n10. Testing legacy support error handling...")
    
    with patch('stts.engine.STTEngineManager') as mock_manager_class:
        # Test ValueError
        mock_manager = Mock()
        mock_manager.get_engine.side_effect = ValueError("Unknown engine")
        mock_manager_class.return_value = mock_manager
        
        try:
            engine = SpeechToTextEngine()
            if not hasattr(engine, 'model'):
                print("   ✅ Passed: ValueError in legacy support handled gracefully")
            else:
                print("   ❌ Failed: Should not have model attribute after ValueError")
        except Exception as e:
            print(f"   ❌ Failed: Should handle ValueError gracefully: {e}")
        
        # Test unexpected exception
        class CustomError(Exception):
            pass
        
        mock_manager.get_engine.side_effect = CustomError("Unexpected")
        mock_manager_class.return_value = mock_manager
        
        try:
            engine = SpeechToTextEngine()
            if not hasattr(engine, 'model'):
                print("   ✅ Passed: Unexpected exception in legacy support handled gracefully")
            else:
                print("   ❌ Failed: Should not have model attribute after unexpected exception")
        except Exception as e:
            print(f"   ❌ Failed: Should handle unexpected exceptions gracefully: {e}")


def main():
    """Run all failure scenario tests"""
    print("=" * 70)
    print("Testing Exception Handling with Various Failure Scenarios")
    print("=" * 70)
    
    # Run all tests
    test_corrupted_audio()
    test_missing_model_files()
    test_import_errors()
    test_memory_errors()
    test_permission_errors()
    test_runtime_errors()
    test_configuration_errors()
    test_unexpected_exceptions()
    test_system_exceptions_propagation()
    test_legacy_support_errors()
    
    print("\n" + "=" * 70)
    print("✅ All failure scenario tests completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()