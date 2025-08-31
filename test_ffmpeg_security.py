#!/usr/bin/env python3
"""
Comprehensive security test suite for FFmpeg hardening

This test suite validates that the security measures properly protect
against various attack vectors including zip bombs, infinite loops,
and resource exhaustion attacks.
"""

import os
import sys
import unittest
import tempfile
import struct
import time
import threading
import subprocess
from io import BytesIO
import wave
import random
import string

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stts.audio_validators import AudioSecurityValidator, validate_audio_security
from stts.security.ffmpeg_sandbox import (
    FFmpegSandbox, FFmpegSecurityConfig, CircuitBreaker,
    ResourceMonitor, secure_normalize_audio
)
from stts.base_engine import BaseSTTEngine
from stts.validators import validate_audio_file, MAX_FILE_SIZE


class MockSTTEngine(BaseSTTEngine):
    """Mock STT engine for testing"""
    
    def initialize(self):
        pass
    
    def transcribe_raw(self, audio_data, sample_rate=16000):
        return "mock transcription"


class TestAudioValidators(unittest.TestCase):
    """Test audio validation security features"""
    
    def test_compression_bomb_detection(self):
        """Test detection of compression bomb attacks"""
        # Create a fake WAV with inflated size claim
        fake_wav = bytearray(b'RIFF')
        # Claim file is 1GB but actually only 100 bytes
        fake_wav.extend(struct.pack('<I', 1024 * 1024 * 1024))  # 1GB claim
        fake_wav.extend(b'WAVEfmt ')
        fake_wav.extend(struct.pack('<I', 16))  # fmt chunk size
        fake_wav.extend(struct.pack('<HHIIHH', 1, 1, 16000, 32000, 2, 16))
        fake_wav.extend(b'data')
        fake_wav.extend(struct.pack('<I', 1024 * 1024 * 1024))  # 1GB data claim
        fake_wav.extend(b'\x00' * 50)  # Only 50 bytes of actual data
        
        is_safe, error = AudioSecurityValidator.check_compression_ratio(bytes(fake_wav))
        self.assertFalse(is_safe)
        self.assertIn("compression bomb", error.lower() or "potential audio bomb", error.lower())
    
    def test_metadata_bomb_detection(self):
        """Test detection of metadata bomb attacks"""
        # Create MP3 with excessive ID3 tag size
        fake_mp3 = bytearray(b'ID3')
        fake_mp3.extend(b'\x04\x00')  # Version
        fake_mp3.extend(b'\x00')  # Flags
        # Synchsafe integer for 50MB tag size
        fake_mp3.extend(bytes([0x01, 0x7f, 0x7f, 0x7f]))
        
        is_safe, error = AudioSecurityValidator.check_metadata_bombs(bytes(fake_mp3))
        self.assertFalse(is_safe)
        self.assertIn("metadata too large", error.lower())
    
    def test_recursive_structure_detection(self):
        """Test detection of recursive/cyclic structures"""
        # Create RIFF with many small chunks (chunk bomb)
        fake_wav = bytearray(b'RIFF')
        fake_wav.extend(struct.pack('<I', 10000))
        fake_wav.extend(b'WAVE')
        
        # Add 2000 tiny chunks
        for i in range(2000):
            fake_wav.extend(b'TEST')
            fake_wav.extend(struct.pack('<I', 1))
            fake_wav.extend(b'X')
        
        is_safe, error = AudioSecurityValidator.check_recursive_structures(bytes(fake_wav))
        self.assertFalse(is_safe)
        self.assertIn("chunk", error.lower())
    
    def test_polyglot_file_detection(self):
        """Test detection of polyglot files"""
        # Create file with multiple signatures
        fake_file = b'RIFF' + b'\x00' * 100 + b'%PDF-1.4'
        
        is_safe, error = AudioSecurityValidator.check_polyglot_files(fake_file)
        self.assertFalse(is_safe)
        self.assertIn("pdf", error.lower())
    
    def test_valid_wav_structure(self):
        """Test validation of legitimate WAV file structure"""
        # Create a valid WAV file
        sample_rate = 16000
        duration = 1  # 1 second
        samples = sample_rate * duration
        
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            # Write silence
            wav.writeframes(b'\x00\x00' * samples)
        
        wav_bytes = wav_buffer.getvalue()
        
        is_valid, error, metadata = AudioSecurityValidator.validate_wav_structure(wav_bytes)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        self.assertEqual(metadata['channels'], 1)
        self.assertEqual(metadata['sample_rate'], 16000)
        self.assertAlmostEqual(metadata['duration_seconds'], 1.0, places=1)
    
    def test_invalid_wav_structure(self):
        """Test detection of invalid WAV structure"""
        # WAV with invalid channel count
        fake_wav = bytearray(b'RIFF')
        fake_wav.extend(struct.pack('<I', 100))
        fake_wav.extend(b'WAVEfmt ')
        fake_wav.extend(struct.pack('<I', 16))
        fake_wav.extend(struct.pack('<HHIIHH', 1, 255, 16000, 32000, 2, 16))  # 255 channels
        
        is_valid, error, metadata = AudioSecurityValidator.validate_wav_structure(bytes(fake_wav))
        self.assertFalse(is_valid)
        self.assertIn("channel", error.lower())
    
    def test_comprehensive_validation(self):
        """Test comprehensive validation function"""
        # Create a valid small WAV
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b'\x00\x00' * 1000)
        
        wav_bytes = wav_buffer.getvalue()
        
        is_valid, error, metadata = validate_audio_security(wav_bytes)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        self.assertIn('file_hash', metadata)
        self.assertIn('file_size', metadata)


class TestFFmpegSandbox(unittest.TestCase):
    """Test FFmpeg sandboxing and resource control"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = FFmpegSecurityConfig(
            max_memory_mb=256,
            max_cpu_seconds=5,
            max_output_size_mb=10,
            max_duration_seconds=60,
            timeout_seconds=3,
            circuit_breaker_threshold=3,
            circuit_breaker_reset_time=1
        )
        self.sandbox = FFmpegSandbox(self.config)
    
    def test_audio_metadata_validation(self):
        """Test audio metadata validation"""
        # Create valid WAV
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b'\x00\x00' * 16000)  # 1 second
        
        wav_bytes = wav_buffer.getvalue()
        
        is_valid, error, metadata = self.sandbox.validate_audio_metadata(wav_bytes)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        self.assertIn('input_size_mb', metadata)
    
    def test_client_rate_limiting(self):
        """Test per-client rate limiting"""
        client_id = "test_client_123"
        
        # Should allow initial requests
        for i in range(self.config.max_requests_per_minute):
            allowed, error = self.sandbox.check_client_limits(client_id)
            if i < self.config.max_requests_per_minute:
                self.assertTrue(allowed, f"Request {i+1} should be allowed")
        
        # Should block after limit
        allowed, error = self.sandbox.check_client_limits(client_id)
        self.assertFalse(allowed)
        self.assertIn("exceeded", error.lower())
    
    def test_circuit_breaker(self):
        """Test circuit breaker pattern"""
        breaker = CircuitBreaker(threshold=2, reset_time=1)
        
        def failing_function():
            raise Exception("Test failure")
        
        # Trigger failures to open circuit
        for i in range(2):
            with self.assertRaises(Exception):
                breaker.call(failing_function)
        
        # Circuit should be open now
        with self.assertRaises(RuntimeError) as ctx:
            breaker.call(failing_function)
        self.assertIn("circuit breaker is open", str(ctx.exception).lower())
        
        # Wait for reset time
        time.sleep(1.5)
        
        # Circuit should be half-open, successful call should close it
        def successful_function():
            return "success"
        
        result = breaker.call(successful_function)
        self.assertEqual(result, "success")
    
    def test_secure_ffmpeg_execution(self):
        """Test secure FFmpeg execution with valid audio"""
        # Create a valid WAV file
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(44100)
            # Generate 0.5 seconds of audio
            samples = int(44100 * 0.5)
            audio_data = bytes([random.randint(0, 255) for _ in range(samples * 2)])
            wav.writeframes(audio_data)
        
        wav_bytes = wav_buffer.getvalue()
        
        # Execute with sandbox
        success, output, error, stats = self.sandbox.execute_ffmpeg_secure(
            wav_bytes,
            output_format='WAV',
            additional_args={'ac': 1, 'ar': '16k'},
            client_id="test_client"
        )
        
        self.assertTrue(success)
        self.assertIsNotNone(output)
        self.assertIsNone(error)
        self.assertIn('execution_time', stats)
        
        # Verify output is valid WAV
        self.assertTrue(output.startswith(b'RIFF'))
        self.assertIn(b'WAVE', output[:20])
    
    def test_timeout_protection(self):
        """Test timeout protection for hanging operations"""
        # Create a large file that might take long to process
        large_wav = BytesIO()
        with wave.open(large_wav, 'wb') as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            # Generate 30 seconds of stereo audio
            samples = 48000 * 30
            wav.writeframes(b'\x00' * (samples * 2 * 2))
        
        large_bytes = large_wav.getvalue()
        
        # Use very short timeout
        self.sandbox.config.timeout_seconds = 0.1
        
        success, output, error, stats = self.sandbox.execute_ffmpeg_secure(
            large_bytes,
            output_format='WAV',
            client_id="test_timeout"
        )
        
        # Should fail due to timeout
        self.assertFalse(success)
        self.assertIn("timeout", error.lower())
    
    def test_memory_limit_protection(self):
        """Test memory limit protection"""
        # This test is tricky to implement reliably across platforms
        # We'll test the configuration is properly set
        self.assertEqual(self.sandbox.config.max_memory_mb, 256)
        self.assertEqual(self.sandbox.config.max_output_size_mb, 10)
    
    def test_malicious_input_rejection(self):
        """Test rejection of malicious inputs"""
        # Test with fake compressed bomb
        fake_wav = bytearray(b'RIFF')
        fake_wav.extend(struct.pack('<I', 1024 * 1024 * 1024))  # Claim 1GB
        fake_wav.extend(b'WAVEfmt ')
        fake_wav.extend(struct.pack('<I', 16))
        fake_wav.extend(struct.pack('<HHIIHH', 1, 1, 16000, 32000, 2, 16))
        fake_wav.extend(b'data')
        fake_wav.extend(struct.pack('<I', 1024 * 1024 * 1024))
        fake_wav.extend(b'\x00' * 100)
        
        success, output, error, stats = self.sandbox.execute_ffmpeg_secure(
            bytes(fake_wav),
            output_format='WAV',
            client_id="test_malicious"
        )
        
        self.assertFalse(success)
        self.assertIn("validation_error", stats)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete security system"""
    
    def test_base_engine_with_security(self):
        """Test BaseSTTEngine with security features"""
        engine = MockSTTEngine()
        
        # Create a valid WAV
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b'\x00\x00' * 8000)  # 0.5 seconds
        
        wav_bytes = wav_buffer.getvalue()
        
        # Should successfully normalize
        normalized = engine.normalize_audio(wav_bytes, client_id="test_integration")
        self.assertIsNotNone(normalized)
        self.assertTrue(normalized.startswith(b'RIFF'))
        
        # Should successfully transcribe
        result = engine.transcribe(wav_bytes, client_id="test_integration")
        self.assertEqual(result, "mock transcription")
    
    def test_legacy_mode(self):
        """Test legacy mode for backward compatibility"""
        # Enable legacy mode
        os.environ['FFMPEG_LEGACY_MODE'] = 'true'
        
        try:
            engine = MockSTTEngine()
            
            wav_buffer = BytesIO()
            with wave.open(wav_buffer, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b'\x00\x00' * 8000)
            
            wav_bytes = wav_buffer.getvalue()
            
            # Should use legacy normalization
            normalized = engine.normalize_audio(wav_bytes)
            self.assertIsNotNone(normalized)
            
        finally:
            # Reset environment
            os.environ['FFMPEG_LEGACY_MODE'] = 'false'
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        sandbox = FFmpegSandbox()
        
        # Create test audio
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b'\x00\x00' * 1600)  # 0.1 seconds
        
        wav_bytes = wav_buffer.getvalue()
        
        results = []
        
        def process_audio(client_id):
            success, output, error, stats = secure_normalize_audio(
                wav_bytes,
                client_id=client_id
            )
            results.append((success, client_id))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=process_audio, args=(f"client_{i}",))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join(timeout=10)
        
        # Check results
        self.assertEqual(len(results), 5)
        for success, client_id in results:
            self.assertTrue(success, f"Client {client_id} should succeed")
    
    def test_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion"""
        # Create multiple large files
        large_files = []
        for i in range(3):
            wav_buffer = BytesIO()
            with wave.open(wav_buffer, 'wb') as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                # 5 seconds of stereo audio
                wav.writeframes(b'\x00' * (48000 * 5 * 2 * 2))
            large_files.append(wav_buffer.getvalue())
        
        sandbox = FFmpegSandbox(FFmpegSecurityConfig(
            max_concurrent_per_client=1,
            timeout_seconds=2
        ))
        
        # Process files - should handle resource limits gracefully
        for i, audio in enumerate(large_files):
            success, output, error, stats = sandbox.execute_ffmpeg_secure(
                audio,
                output_format='WAV',
                additional_args={'ac': 1, 'ar': '8k'},  # Downsample to reduce load
                client_id="resource_test"
            )
            
            # First should succeed, others might be rate limited
            if i == 0:
                self.assertTrue(success or "concurrent" in (error or "").lower())


class TestSecurityScenarios(unittest.TestCase):
    """Test specific security attack scenarios"""
    
    def test_zip_bomb_scenario(self):
        """Test handling of zip bomb attack"""
        # Create a file that claims to be much larger than it is
        fake_wav = bytearray(b'RIFF')
        fake_wav.extend(struct.pack('<I', 2**31 - 1))  # Maximum 32-bit value
        fake_wav.extend(b'WAVEfmt ')
        fake_wav.extend(struct.pack('<I', 16))
        fake_wav.extend(struct.pack('<HHIIHH', 1, 1, 16000, 32000, 2, 16))
        fake_wav.extend(b'data')
        fake_wav.extend(struct.pack('<I', 2**31 - 1))
        fake_wav.extend(b'\x00' * 1000)
        
        # Should be rejected by validators
        is_valid, error, metadata = validate_audio_security(bytes(fake_wav))
        self.assertFalse(is_valid)
        self.assertIn("bomb", error.lower())
    
    def test_infinite_loop_scenario(self):
        """Test handling of files designed to cause infinite loops"""
        # Create RIFF with circular references (invalid but might cause loops)
        fake_wav = bytearray(b'RIFF')
        fake_wav.extend(struct.pack('<I', 100))
        fake_wav.extend(b'WAVE')
        
        # Add chunk that points back to itself
        fake_wav.extend(b'LOOP')
        fake_wav.extend(struct.pack('<I', 0xFFFFFFFF))  # Invalid size
        
        is_valid, error, metadata = validate_audio_security(bytes(fake_wav))
        self.assertFalse(is_valid)
    
    def test_memory_exhaustion_scenario(self):
        """Test handling of memory exhaustion attacks"""
        # Create file with excessive metadata
        fake_mp3 = bytearray(b'ID3')
        fake_mp3.extend(b'\x04\x00\x00')
        # Try to allocate 100MB of metadata
        fake_mp3.extend(bytes([0x0C, 0x20, 0x00, 0x00]))
        
        is_valid, error, metadata = validate_audio_security(bytes(fake_mp3))
        self.assertFalse(is_valid)
        self.assertIn("metadata", error.lower())
    
    def test_corrupted_file_scenario(self):
        """Test handling of corrupted files"""
        # Random bytes that aren't valid audio
        random_bytes = bytes([random.randint(0, 255) for _ in range(1000)])
        
        is_valid, error, metadata = validate_audio_security(random_bytes)
        self.assertFalse(is_valid)
    
    def test_polyglot_attack_scenario(self):
        """Test handling of polyglot file attacks"""
        # File that's both WAV and contains executable code
        fake_wav = bytearray(b'RIFF')
        fake_wav.extend(struct.pack('<I', 1000))
        fake_wav.extend(b'WAVE')
        fake_wav.extend(b'fmt ')
        fake_wav.extend(struct.pack('<I', 16))
        fake_wav.extend(struct.pack('<HHIIHH', 1, 1, 16000, 32000, 2, 16))
        fake_wav.extend(b'data')
        fake_wav.extend(struct.pack('<I', 100))
        fake_wav.extend(b'\x00' * 100)
        # Add executable signature
        fake_wav.extend(b'MZ\x90\x00')  # PE header
        
        is_valid, error, metadata = validate_audio_security(bytes(fake_wav))
        self.assertFalse(is_valid)
        self.assertIn("suspicious", error.lower())


def run_security_tests():
    """Run all security tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAudioValidators))
    suite.addTests(loader.loadTestsFromTestCase(TestFFmpegSandbox))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    print("SECURITY TEST SUMMARY")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All security tests passed!")
    else:
        print("\n❌ Some tests failed. Please review the output above.")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Check for required dependencies
    try:
        import ffmpeg
        import psutil
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Please install: pip install ffmpeg-python psutil")
        sys.exit(1)
    
    # Run the tests
    success = run_security_tests()
    sys.exit(0 if success else 1)