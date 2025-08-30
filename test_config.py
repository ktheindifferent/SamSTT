#!/usr/bin/env python3
"""
Test suite for security configuration module
"""
import unittest
import os
import sys
from unittest.mock import patch
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress log output during tests
logging.basicConfig(level=logging.CRITICAL)


class TestSecurityConfig(unittest.TestCase):
    """Test the SecurityConfig class and its validation"""
    
    def setUp(self):
        """Set up test environment"""
        # Store original env vars to restore later
        self.original_env = {}
        env_vars = [
            'MAX_AUDIO_DURATION',
            'MAX_FILE_SIZE',
            'MAX_REQUESTS_PER_MINUTE',
            'MAX_REQUESTS_PER_HOUR',
            'REQUEST_TIMEOUT',
            'MAX_ENGINE_WORKERS',
            'MAX_WAV_EXPANSION_FACTOR'
        ]
        for var in env_vars:
            self.original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Restore original environment"""
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]
    
    def test_default_values(self):
        """Test that default values are correctly set"""
        # Import fresh to get default values
        from stts.config import SecurityConfig
        
        self.assertEqual(SecurityConfig.MAX_AUDIO_DURATION, 600)  # 10 minutes
        self.assertEqual(SecurityConfig.MAX_FILE_SIZE, 50 * 1024 * 1024)  # 50MB
        self.assertEqual(SecurityConfig.MAX_REQUESTS_PER_MINUTE, 60)
        self.assertEqual(SecurityConfig.MAX_REQUESTS_PER_HOUR, 600)
        self.assertEqual(SecurityConfig.REQUEST_TIMEOUT, 60)
        self.assertEqual(SecurityConfig.MAX_ENGINE_WORKERS, 2)
        self.assertEqual(SecurityConfig.MAX_WAV_EXPANSION_FACTOR, 2.0)
    
    def test_env_override(self):
        """Test that environment variables override defaults"""
        os.environ['MAX_AUDIO_DURATION'] = '300'
        os.environ['MAX_FILE_SIZE'] = '104857600'  # 100MB
        os.environ['MAX_REQUESTS_PER_MINUTE'] = '30'
        os.environ['MAX_REQUESTS_PER_HOUR'] = '300'
        os.environ['REQUEST_TIMEOUT'] = '120'
        os.environ['MAX_ENGINE_WORKERS'] = '4'
        os.environ['MAX_WAV_EXPANSION_FACTOR'] = '3.5'
        
        # Reload module to pick up new env vars
        import importlib
        import stts.config
        importlib.reload(stts.config)
        from stts.config import SecurityConfig
        
        self.assertEqual(SecurityConfig.MAX_AUDIO_DURATION, 300)
        self.assertEqual(SecurityConfig.MAX_FILE_SIZE, 104857600)
        self.assertEqual(SecurityConfig.MAX_REQUESTS_PER_MINUTE, 30)
        self.assertEqual(SecurityConfig.MAX_REQUESTS_PER_HOUR, 300)
        self.assertEqual(SecurityConfig.REQUEST_TIMEOUT, 120)
        self.assertEqual(SecurityConfig.MAX_ENGINE_WORKERS, 4)
        self.assertEqual(SecurityConfig.MAX_WAV_EXPANSION_FACTOR, 3.5)
    
    def test_validation_audio_duration(self):
        """Test audio duration validation"""
        from stts.config import SecurityConfig
        
        # Test valid range
        SecurityConfig.MAX_AUDIO_DURATION = 300
        self.assertTrue(SecurityConfig.validate())
        
        # Test too low
        SecurityConfig.MAX_AUDIO_DURATION = 5
        self.assertFalse(SecurityConfig.validate())
        
        # Test too high
        SecurityConfig.MAX_AUDIO_DURATION = 7200
        self.assertFalse(SecurityConfig.validate())
        
        # Reset to default
        SecurityConfig.MAX_AUDIO_DURATION = 600
    
    def test_validation_file_size(self):
        """Test file size validation"""
        from stts.config import SecurityConfig
        
        # Test valid range
        SecurityConfig.MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        self.assertTrue(SecurityConfig.validate())
        
        # Test too low
        SecurityConfig.MAX_FILE_SIZE = 512  # 512 bytes
        self.assertFalse(SecurityConfig.validate())
        
        # Test too high
        SecurityConfig.MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB
        self.assertFalse(SecurityConfig.validate())
        
        # Reset to default
        SecurityConfig.MAX_FILE_SIZE = 50 * 1024 * 1024
    
    def test_validation_rate_limits(self):
        """Test rate limit validation"""
        from stts.config import SecurityConfig
        
        # Test valid configuration
        SecurityConfig.MAX_REQUESTS_PER_MINUTE = 30
        SecurityConfig.MAX_REQUESTS_PER_HOUR = 300
        self.assertTrue(SecurityConfig.validate())
        
        # Test inconsistent limits (hour < minute)
        SecurityConfig.MAX_REQUESTS_PER_MINUTE = 100
        SecurityConfig.MAX_REQUESTS_PER_HOUR = 50
        self.assertFalse(SecurityConfig.validate())
        
        # Test out of range
        SecurityConfig.MAX_REQUESTS_PER_MINUTE = 0
        self.assertFalse(SecurityConfig.validate())
        
        SecurityConfig.MAX_REQUESTS_PER_MINUTE = 20000
        self.assertFalse(SecurityConfig.validate())
        
        # Reset to defaults
        SecurityConfig.MAX_REQUESTS_PER_MINUTE = 60
        SecurityConfig.MAX_REQUESTS_PER_HOUR = 600
    
    def test_validation_timeout(self):
        """Test timeout validation"""
        from stts.config import SecurityConfig
        
        # Test valid range
        SecurityConfig.REQUEST_TIMEOUT = 30
        self.assertTrue(SecurityConfig.validate())
        
        # Test too low
        SecurityConfig.REQUEST_TIMEOUT = 0
        self.assertFalse(SecurityConfig.validate())
        
        # Test too high
        SecurityConfig.REQUEST_TIMEOUT = 1200
        self.assertFalse(SecurityConfig.validate())
        
        # Reset to default
        SecurityConfig.REQUEST_TIMEOUT = 60
    
    def test_validation_workers(self):
        """Test worker count validation"""
        from stts.config import SecurityConfig
        
        # Test valid range
        SecurityConfig.MAX_ENGINE_WORKERS = 4
        self.assertTrue(SecurityConfig.validate())
        
        # Test too low
        SecurityConfig.MAX_ENGINE_WORKERS = 0
        self.assertFalse(SecurityConfig.validate())
        
        # Test too high
        SecurityConfig.MAX_ENGINE_WORKERS = 200
        self.assertFalse(SecurityConfig.validate())
        
        # Reset to default
        SecurityConfig.MAX_ENGINE_WORKERS = 2
    
    def test_apply_safe_defaults(self):
        """Test that safe defaults are applied when values are invalid"""
        from stts.config import SecurityConfig
        
        # Set invalid values
        SecurityConfig.MAX_AUDIO_DURATION = 5  # Too low
        SecurityConfig.MAX_FILE_SIZE = 100  # Too low
        SecurityConfig.MAX_REQUESTS_PER_MINUTE = 0  # Too low
        SecurityConfig.REQUEST_TIMEOUT = 2000  # Too high
        SecurityConfig.MAX_ENGINE_WORKERS = 500  # Too high
        
        # Apply safe defaults
        SecurityConfig.apply_safe_defaults()
        
        # Check values are within valid ranges
        self.assertGreaterEqual(SecurityConfig.MAX_AUDIO_DURATION, SecurityConfig.MIN_AUDIO_DURATION_LIMIT)
        self.assertLessEqual(SecurityConfig.MAX_AUDIO_DURATION, SecurityConfig.MAX_AUDIO_DURATION_LIMIT)
        
        self.assertGreaterEqual(SecurityConfig.MAX_FILE_SIZE, SecurityConfig.MIN_FILE_SIZE_LIMIT)
        self.assertLessEqual(SecurityConfig.MAX_FILE_SIZE, SecurityConfig.MAX_FILE_SIZE_LIMIT)
        
        self.assertGreaterEqual(SecurityConfig.MAX_REQUESTS_PER_MINUTE, SecurityConfig.MIN_RATE_LIMIT)
        self.assertLessEqual(SecurityConfig.MAX_REQUESTS_PER_MINUTE, SecurityConfig.MAX_RATE_LIMIT)
        
        self.assertGreaterEqual(SecurityConfig.REQUEST_TIMEOUT, SecurityConfig.MIN_TIMEOUT)
        self.assertLessEqual(SecurityConfig.REQUEST_TIMEOUT, SecurityConfig.MAX_TIMEOUT)
        
        self.assertGreaterEqual(SecurityConfig.MAX_ENGINE_WORKERS, SecurityConfig.MIN_WORKERS)
        self.assertLessEqual(SecurityConfig.MAX_ENGINE_WORKERS, SecurityConfig.MAX_WORKERS)
    
    def test_config_summary(self):
        """Test configuration summary generation"""
        from stts.config import SecurityConfig
        
        summary = SecurityConfig.get_config_summary()
        
        # Check structure
        self.assertIn('audio', summary)
        self.assertIn('rate_limits', summary)
        self.assertIn('processing', summary)
        
        # Check audio settings
        self.assertIn('max_duration_seconds', summary['audio'])
        self.assertIn('max_file_size_mb', summary['audio'])
        self.assertIn('wav_expansion_factor', summary['audio'])
        
        # Check rate limits
        self.assertIn('per_minute', summary['rate_limits'])
        self.assertIn('per_hour', summary['rate_limits'])
        
        # Check processing settings
        self.assertIn('request_timeout_seconds', summary['processing'])
        self.assertIn('max_workers', summary['processing'])
        
        # Check values match
        self.assertEqual(summary['audio']['max_duration_seconds'], SecurityConfig.MAX_AUDIO_DURATION)
        self.assertEqual(summary['rate_limits']['per_minute'], SecurityConfig.MAX_REQUESTS_PER_MINUTE)
    
    def test_boundary_values(self):
        """Test configuration at boundary values"""
        from stts.config import SecurityConfig
        
        # Test minimum valid values
        SecurityConfig.MAX_AUDIO_DURATION = SecurityConfig.MIN_AUDIO_DURATION_LIMIT
        SecurityConfig.MAX_FILE_SIZE = SecurityConfig.MIN_FILE_SIZE_LIMIT
        SecurityConfig.MAX_REQUESTS_PER_MINUTE = SecurityConfig.MIN_RATE_LIMIT
        SecurityConfig.MAX_REQUESTS_PER_HOUR = SecurityConfig.MIN_RATE_LIMIT
        SecurityConfig.REQUEST_TIMEOUT = SecurityConfig.MIN_TIMEOUT
        SecurityConfig.MAX_ENGINE_WORKERS = SecurityConfig.MIN_WORKERS
        SecurityConfig.MAX_WAV_EXPANSION_FACTOR = 1.0
        
        self.assertTrue(SecurityConfig.validate())
        
        # Test maximum valid values
        SecurityConfig.MAX_AUDIO_DURATION = SecurityConfig.MAX_AUDIO_DURATION_LIMIT
        SecurityConfig.MAX_FILE_SIZE = SecurityConfig.MAX_FILE_SIZE_LIMIT
        SecurityConfig.MAX_REQUESTS_PER_MINUTE = SecurityConfig.MAX_RATE_LIMIT
        SecurityConfig.MAX_REQUESTS_PER_HOUR = SecurityConfig.MAX_RATE_LIMIT
        SecurityConfig.REQUEST_TIMEOUT = SecurityConfig.MAX_TIMEOUT
        SecurityConfig.MAX_ENGINE_WORKERS = SecurityConfig.MAX_WORKERS
        SecurityConfig.MAX_WAV_EXPANSION_FACTOR = 10.0
        
        self.assertTrue(SecurityConfig.validate())
    
    def test_rate_limit_consistency_fix(self):
        """Test that hour limit is adjusted to match minute limit if needed"""
        from stts.config import SecurityConfig
        
        # Set inconsistent limits
        SecurityConfig.MAX_REQUESTS_PER_MINUTE = 100
        SecurityConfig.MAX_REQUESTS_PER_HOUR = 50  # Less than per minute
        
        # Apply safe defaults should fix this
        SecurityConfig.apply_safe_defaults()
        
        # Hour limit should be at least minute limit
        self.assertGreaterEqual(SecurityConfig.MAX_REQUESTS_PER_HOUR, SecurityConfig.MAX_REQUESTS_PER_MINUTE)


class TestConfigIntegration(unittest.TestCase):
    """Test configuration integration with other modules"""
    
    def test_validators_import(self):
        """Test that validators module correctly imports from config"""
        from stts.validators import MAX_FILE_SIZE, MAX_REQUESTS_PER_MINUTE
        from stts.config import SecurityConfig
        
        # Values should match
        self.assertEqual(MAX_FILE_SIZE, SecurityConfig.MAX_FILE_SIZE)
        self.assertEqual(MAX_REQUESTS_PER_MINUTE, SecurityConfig.MAX_REQUESTS_PER_MINUTE)
    
    def test_base_engine_uses_config(self):
        """Test that base_engine uses configuration values"""
        import importlib
        import stts.base_engine
        importlib.reload(stts.base_engine)
        
        # Check that SecurityConfig is imported
        self.assertTrue(hasattr(stts.base_engine, 'SecurityConfig'))
    
    def test_app_uses_config(self):
        """Test that app module uses configuration values"""
        import importlib
        import stts.app
        importlib.reload(stts.app)
        
        # Check that SecurityConfig is imported
        self.assertTrue(hasattr(stts.app, 'SecurityConfig'))


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing code"""
    
    def test_validators_exports(self):
        """Test that validators still exports expected variables"""
        from stts import validators
        
        # Check that old variables are still accessible
        self.assertTrue(hasattr(validators, 'MAX_FILE_SIZE'))
        self.assertTrue(hasattr(validators, 'MAX_REQUESTS_PER_MINUTE'))
        self.assertTrue(hasattr(validators, 'MAX_REQUESTS_PER_HOUR'))
        self.assertTrue(hasattr(validators, 'REQUEST_TIMEOUT'))
    
    def test_env_var_compatibility(self):
        """Test that old environment variable names still work"""
        os.environ['MAX_FILE_SIZE'] = '104857600'  # 100MB
        os.environ['MAX_REQUESTS_PER_MINUTE'] = '120'
        
        # Reload modules
        import importlib
        import stts.config
        importlib.reload(stts.config)
        from stts.config import SecurityConfig
        
        # Check values are picked up
        self.assertEqual(SecurityConfig.MAX_FILE_SIZE, 104857600)
        self.assertEqual(SecurityConfig.MAX_REQUESTS_PER_MINUTE, 120)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)