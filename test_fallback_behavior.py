#!/usr/bin/env python3
"""Test script to verify fallback behavior with new exception handling"""

import sys
import os
from unittest.mock import MagicMock, patch

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stts.exceptions import (
    TranscriptionError,
    AudioProcessingError,
    InvalidAudioError,
    EngineNotAvailableError,
    ModelNotFoundError
)


def test_fallback_on_transcription_error():
    """Test that fallback works when primary engine fails with TranscriptionError"""
    print("Testing fallback on TranscriptionError...")
    
    with patch('stts.engine_manager.DeepSpeechEngine') as MockDS, \
         patch('stts.engine_manager.WhisperEngine') as MockWhisper:
        
        # Setup DeepSpeech to fail
        mock_ds = MagicMock()
        mock_ds.is_available = True
        mock_ds.name = "DeepSpeech"
        mock_ds.transcribe.side_effect = TranscriptionError(
            "DeepSpeech failed", engine="deepspeech"
        )
        MockDS.return_value = mock_ds
        
        # Setup Whisper to succeed
        mock_whisper = MagicMock()
        mock_whisper.is_available = True
        mock_whisper.name = "Whisper"
        mock_whisper.transcribe.return_value = "Hello from fallback"
        MockWhisper.return_value = mock_whisper
        
        from stts.engine_manager import STTEngineManager
        
        manager = STTEngineManager(
            default_engine='deepspeech',
            config={'deepspeech': {}, 'whisper': {}}
        )
        
        result = manager.transcribe(b"test_audio")
        
        assert result['text'] == "Hello from fallback"
        assert result['engine'] == "Whisper"
        assert result['fallback'] == True
        print("✓ Fallback to Whisper succeeded after DeepSpeech TranscriptionError")


def test_no_fallback_on_audio_error():
    """Test that fallback is not attempted for AudioProcessingError"""
    print("\nTesting no fallback on AudioProcessingError...")
    
    with patch('stts.engine_manager.DeepSpeechEngine') as MockDS, \
         patch('stts.engine_manager.WhisperEngine') as MockWhisper:
        
        # Setup DeepSpeech to fail with audio error
        mock_ds = MagicMock()
        mock_ds.is_available = True
        mock_ds.name = "DeepSpeech"
        mock_ds.transcribe.side_effect = InvalidAudioError(
            "Corrupted audio", engine="deepspeech"
        )
        MockDS.return_value = mock_ds
        
        # Setup Whisper (should not be called)
        mock_whisper = MagicMock()
        mock_whisper.is_available = True
        mock_whisper.name = "Whisper"
        mock_whisper.transcribe.return_value = "Should not be called"
        MockWhisper.return_value = mock_whisper
        
        from stts.engine_manager import STTEngineManager
        
        manager = STTEngineManager(
            default_engine='deepspeech',
            config={'deepspeech': {}, 'whisper': {}}
        )
        
        try:
            result = manager.transcribe(b"corrupted_audio")
            assert False, "Should have raised InvalidAudioError"
        except InvalidAudioError as e:
            assert "Corrupted audio" in str(e)
            # Verify Whisper was not called for fallback
            mock_whisper.transcribe.assert_not_called()
            print("✓ No fallback attempted for InvalidAudioError (as expected)")


def test_fallback_chain():
    """Test fallback through multiple engines"""
    print("\nTesting fallback chain through multiple engines...")
    
    with patch('stts.engine_manager.DeepSpeechEngine') as MockDS, \
         patch('stts.engine_manager.WhisperEngine') as MockWhisper, \
         patch('stts.engine_manager.VoskEngine') as MockVosk:
        
        # Setup all engines
        mock_ds = MagicMock()
        mock_ds.is_available = True
        mock_ds.name = "DeepSpeech"
        mock_ds.transcribe.side_effect = TranscriptionError(
            "DeepSpeech failed", engine="deepspeech"
        )
        MockDS.return_value = mock_ds
        
        mock_whisper = MagicMock()
        mock_whisper.is_available = True
        mock_whisper.name = "Whisper"
        mock_whisper.transcribe.side_effect = TranscriptionError(
            "Whisper also failed", engine="whisper"
        )
        MockWhisper.return_value = mock_whisper
        
        mock_vosk = MagicMock()
        mock_vosk.is_available = True
        mock_vosk.name = "Vosk"
        mock_vosk.transcribe.return_value = "Success from Vosk"
        MockVosk.return_value = mock_vosk
        
        from stts.engine_manager import STTEngineManager
        
        manager = STTEngineManager(
            default_engine='deepspeech',
            config={'deepspeech': {}, 'whisper': {}, 'vosk': {}}
        )
        
        result = manager.transcribe(b"test_audio")
        
        assert result['text'] == "Success from Vosk"
        assert result['engine'] == "Vosk"
        assert result['fallback'] == True
        print("✓ Fallback chain worked: DeepSpeech → Whisper → Vosk (success)")


def test_all_engines_fail():
    """Test error message when all engines fail"""
    print("\nTesting error when all engines fail...")
    
    with patch('stts.engine_manager.DeepSpeechEngine') as MockDS, \
         patch('stts.engine_manager.WhisperEngine') as MockWhisper:
        
        mock_ds = MagicMock()
        mock_ds.is_available = True
        mock_ds.name = "DeepSpeech"
        mock_ds.transcribe.side_effect = TranscriptionError(
            "DeepSpeech failed", engine="deepspeech"
        )
        MockDS.return_value = mock_ds
        
        mock_whisper = MagicMock()
        mock_whisper.is_available = True
        mock_whisper.name = "Whisper"
        mock_whisper.transcribe.side_effect = TranscriptionError(
            "Whisper also failed", engine="whisper"
        )
        MockWhisper.return_value = mock_whisper
        
        from stts.engine_manager import STTEngineManager
        
        manager = STTEngineManager(
            default_engine='deepspeech',
            config={'deepspeech': {}, 'whisper': {}}
        )
        
        try:
            result = manager.transcribe(b"test_audio")
            assert False, "Should have raised TranscriptionError"
        except TranscriptionError as e:
            assert "All STT engines failed" in str(e)
            assert "DeepSpeech failed" in str(e)
            assert "Whisper also failed" in str(e)
            print("✓ Comprehensive error message when all engines fail")


def test_engine_not_available_fallback():
    """Test fallback when requested engine is not available"""
    print("\nTesting fallback when requested engine not available...")
    
    with patch('stts.engine_manager.DeepSpeechEngine') as MockDS, \
         patch('stts.engine_manager.WhisperEngine') as MockWhisper:
        
        # DeepSpeech not available
        MockDS.side_effect = EngineNotAvailableError(
            "DeepSpeech not installed", engine="deepspeech"
        )
        
        # Whisper available
        mock_whisper = MagicMock()
        mock_whisper.is_available = True
        mock_whisper.name = "Whisper"
        mock_whisper.transcribe.return_value = "Whisper worked"
        MockWhisper.return_value = mock_whisper
        
        from stts.engine_manager import STTEngineManager
        
        manager = STTEngineManager(
            default_engine='deepspeech',
            config={'deepspeech': {}, 'whisper': {}}
        )
        
        # Should only have Whisper available
        assert len(manager.engines) == 1
        assert 'whisper' in manager.engines
        print("✓ Manager initialized with only available engines")
        
        # Trying to use unavailable engine should raise error
        try:
            engine = manager.get_engine('deepspeech')
            assert False, "Should have raised EngineNotAvailableError"
        except EngineNotAvailableError as e:
            assert "not available" in str(e)
            print("✓ Appropriate error when requesting unavailable engine")


def test_model_not_found_handling():
    """Test handling of ModelNotFoundError"""
    print("\nTesting ModelNotFoundError handling...")
    
    with patch('stts.engine_manager.DeepSpeechEngine') as MockDS:
        
        MockDS.side_effect = ModelNotFoundError(
            "Model file not found: /app/model.pbmm",
            engine="deepspeech",
            model_path="/app/model.pbmm"
        )
        
        from stts.engine_manager import STTEngineManager
        
        manager = STTEngineManager(
            default_engine='deepspeech',
            config={'deepspeech': {'model_path': '/app/model.pbmm'}}
        )
        
        # Engine should not be available
        assert 'deepspeech' not in manager.engines
        print("✓ Engine with missing model not initialized")
        
        # Trying to get the engine should provide helpful error
        try:
            engine = manager.get_engine('deepspeech')
            assert False, "Should have raised error"
        except ModelNotFoundError as e:
            assert "/app/model.pbmm" in str(e)
            print("✓ ModelNotFoundError preserves model path information")


def main():
    """Run all fallback behavior tests"""
    print("=" * 60)
    print("TESTING FALLBACK BEHAVIOR WITH NEW EXCEPTIONS")
    print("=" * 60)
    
    try:
        test_fallback_on_transcription_error()
        test_no_fallback_on_audio_error()
        test_fallback_chain()
        test_all_engines_fail()
        test_engine_not_available_fallback()
        test_model_not_found_handling()
        
        print("\n" + "=" * 60)
        print("🎉 ALL FALLBACK TESTS PASSED!")
        print("=" * 60)
        print("\nSummary of improvements:")
        print("✅ Fallback works correctly for recoverable errors")
        print("✅ Audio errors don't trigger unnecessary fallbacks")
        print("✅ Error messages are informative and include context")
        print("✅ Engine availability is properly tracked")
        print("✅ Model path information is preserved in errors")
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())