#!/usr/bin/env python3
"""
Test audio duration limit configuration
"""
import unittest
import os
import sys
import wave
import struct
from io import BytesIO
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress log output during tests
import logging
logging.basicConfig(level=logging.CRITICAL)


def create_wav_data(duration_seconds=1, sample_rate=16000):
    """Create WAV data with specified duration"""
    num_samples = int(duration_seconds * sample_rate)
    
    # Create silent audio data (zeros)
    audio_data = struct.pack('<' + 'h' * num_samples, *([0] * num_samples))
    
    # Create WAV file in memory
    wav_buffer = BytesIO()
    with wave.open(wav_buffer, 'wb') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data)
    
    return wav_buffer.getvalue()


class TestAudioDurationLimit(unittest.TestCase):
    """Test audio duration limit enforcement"""
    
    def setUp(self):
        """Set up test environment"""
        # Store original value
        self.original_duration = os.environ.get('MAX_AUDIO_DURATION')
        
    def tearDown(self):
        """Restore original environment"""
        if self.original_duration is not None:
            os.environ['MAX_AUDIO_DURATION'] = self.original_duration
        elif 'MAX_AUDIO_DURATION' in os.environ:
            del os.environ['MAX_AUDIO_DURATION']
    
    def test_default_duration_limit(self):
        """Test default 10-minute duration limit"""
        from stts.config import SecurityConfig
        from stts.base_engine import BaseSTTEngine
        
        # Create a dummy engine implementation for testing
        class TestEngine(BaseSTTEngine):
            def initialize(self):
                pass
            
            def transcribe_raw(self, audio_data, sample_rate=16000):
                return "test"
        
        engine = TestEngine()
        
        # Test audio just under the limit (9 minutes 59 seconds)
        wav_data = create_wav_data(duration_seconds=599)
        try:
            # This should succeed
            result = engine.normalize_audio(wav_data)
            self.assertIsNotNone(result)
        except ValueError as e:
            self.fail(f"Audio under limit was rejected: {e}")
        
        # Test audio just over the limit (10 minutes 1 second)
        wav_data = create_wav_data(duration_seconds=601)
        with self.assertRaises(ValueError) as context:
            engine.normalize_audio(wav_data)
        
        self.assertIn("exceeds maximum", str(context.exception))
    
    def test_custom_duration_limit(self):
        """Test custom duration limit via environment variable"""
        # Set a custom 5-minute limit
        os.environ['MAX_AUDIO_DURATION'] = '300'
        
        # Reload the config module to pick up the new value
        import importlib
        import stts.config
        importlib.reload(stts.config)
        
        # Also reload base_engine to use the new config
        import stts.base_engine
        importlib.reload(stts.base_engine)
        
        from stts.config import SecurityConfig
        from stts.base_engine import BaseSTTEngine
        
        # Verify the config was updated
        self.assertEqual(SecurityConfig.MAX_AUDIO_DURATION, 300)
        
        # Create a dummy engine implementation
        class TestEngine(BaseSTTEngine):
            def initialize(self):
                pass
            
            def transcribe_raw(self, audio_data, sample_rate=16000):
                return "test"
        
        engine = TestEngine()
        
        # Test audio just under the new limit (4 minutes 59 seconds)
        wav_data = create_wav_data(duration_seconds=299)
        try:
            result = engine.normalize_audio(wav_data)
            self.assertIsNotNone(result)
        except ValueError as e:
            self.fail(f"Audio under custom limit was rejected: {e}")
        
        # Test audio just over the new limit (5 minutes 1 second)
        wav_data = create_wav_data(duration_seconds=301)
        with self.assertRaises(ValueError) as context:
            engine.normalize_audio(wav_data)
        
        self.assertIn("exceeds maximum", str(context.exception))
        self.assertIn("300", str(context.exception))  # Should mention the limit
    
    def test_duration_limit_bounds(self):
        """Test that duration limits are properly bounded"""
        # Try to set an unreasonably low limit
        os.environ['MAX_AUDIO_DURATION'] = '5'  # 5 seconds
        
        import importlib
        import stts.config
        importlib.reload(stts.config)
        from stts.config import SecurityConfig
        
        # After applying safe defaults, should be at minimum
        self.assertGreaterEqual(SecurityConfig.MAX_AUDIO_DURATION, 
                               SecurityConfig.MIN_AUDIO_DURATION_LIMIT)
        
        # Try to set an unreasonably high limit
        os.environ['MAX_AUDIO_DURATION'] = '7200'  # 2 hours
        
        importlib.reload(stts.config)
        from stts.config import SecurityConfig
        
        # After applying safe defaults, should be at maximum
        self.assertLessEqual(SecurityConfig.MAX_AUDIO_DURATION,
                            SecurityConfig.MAX_AUDIO_DURATION_LIMIT)
    
    def test_duration_validation_performance(self):
        """Test that duration validation doesn't impact performance"""
        from stts.base_engine import BaseSTTEngine
        import time
        
        class TestEngine(BaseSTTEngine):
            def initialize(self):
                pass
            
            def transcribe_raw(self, audio_data, sample_rate=16000):
                return "test"
        
        engine = TestEngine()
        
        # Create 1-second audio
        wav_data = create_wav_data(duration_seconds=1)
        
        # Measure time for normalization with duration check
        start = time.time()
        for _ in range(10):
            try:
                engine.normalize_audio(wav_data)
            except:
                pass
        elapsed = time.time() - start
        
        # Should be fast (less than 1 second for 10 iterations)
        self.assertLess(elapsed, 1.0, 
                       f"Duration validation is too slow: {elapsed:.3f}s for 10 iterations")
    
    def test_duration_error_message(self):
        """Test that duration limit error messages are informative"""
        from stts.base_engine import BaseSTTEngine
        from stts.config import SecurityConfig
        
        class TestEngine(BaseSTTEngine):
            def initialize(self):
                pass
            
            def transcribe_raw(self, audio_data, sample_rate=16000):
                return "test"
        
        engine = TestEngine()
        
        # Create audio over the limit
        wav_data = create_wav_data(duration_seconds=700)  # ~11.7 minutes
        
        with self.assertRaises(ValueError) as context:
            engine.normalize_audio(wav_data)
        
        error_msg = str(context.exception)
        
        # Check error message contains useful information
        self.assertIn("700", error_msg)  # Actual duration
        self.assertIn(str(SecurityConfig.MAX_AUDIO_DURATION), error_msg)  # Limit
        self.assertIn("exceeds", error_msg.lower())  # Clear indication of problem


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)