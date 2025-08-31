#!/usr/bin/env python3
"""Test suite for configuration file handle management"""

import json
import os
import tempfile
import unittest
from pathlib import Path
import threading
import time
import subprocess
from unittest.mock import patch, mock_open, MagicMock

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from stts.config_manager import ConfigManager, get_config_manager
from stts.engine import SpeechToTextEngine


class TestConfigFileHandling(unittest.TestCase):
    """Test proper file handle management in configuration loading"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager()
        self.config_manager.clear_cache()
        
        # Create test configuration files
        self.valid_config = {
            "default_engine": "whisper",
            "whisper": {
                "model_size": "tiny",
                "device": "cpu"
            }
        }
        
        self.valid_config_path = Path(self.temp_dir) / "valid_config.json"
        with open(self.valid_config_path, 'w') as f:
            json.dump(self.valid_config, f)
        
        # Create malformed JSON file
        self.malformed_config_path = Path(self.temp_dir) / "malformed_config.json"
        with open(self.malformed_config_path, 'w') as f:
            f.write('{"invalid": json"syntax"}')
        
        # Create empty file
        self.empty_config_path = Path(self.temp_dir) / "empty_config.json"
        self.empty_config_path.touch()
        
        # Track initial file descriptor count
        self.initial_fd_count = self._get_open_file_descriptors()
    
    def tearDown(self):
        """Clean up test environment"""
        # Clean up temp files
        for file in Path(self.temp_dir).glob("*"):
            file.unlink()
        Path(self.temp_dir).rmdir()
        
        # Clear cache
        self.config_manager.clear_cache()
    
    def _get_open_file_descriptors(self):
        """Get count of open file descriptors for current process"""
        if HAS_PSUTIL:
            try:
                # Use psutil if available
                process = psutil.Process(os.getpid())
                return len(process.open_files())
            except:
                pass
        
        try:
            # Fallback to /proc on Linux
            pid = os.getpid()
            fd_dir = f"/proc/{pid}/fd"
            if os.path.exists(fd_dir):
                return len(os.listdir(fd_dir))
        except:
            pass
        return -1  # Unable to determine
    
    def test_valid_config_loading(self):
        """Test loading valid configuration file"""
        config = self.config_manager.load_json_config(self.valid_config_path)
        
        self.assertIsNotNone(config)
        self.assertEqual(config["default_engine"], "whisper")
        self.assertIn("whisper", config)
        
        # Verify file handle was closed
        current_fd_count = self._get_open_file_descriptors()
        if current_fd_count >= 0 and self.initial_fd_count >= 0:
            self.assertLessEqual(current_fd_count, self.initial_fd_count + 1,
                               "File descriptors may have leaked")
    
    def test_malformed_json_handling(self):
        """Test handling of malformed JSON without file handle leak"""
        # Load malformed config multiple times
        for _ in range(10):
            config = self.config_manager.load_json_config(self.malformed_config_path)
            self.assertIsNone(config)
        
        # Check file handle count didn't increase significantly
        current_fd_count = self._get_open_file_descriptors()
        if current_fd_count >= 0 and self.initial_fd_count >= 0:
            self.assertLessEqual(current_fd_count, self.initial_fd_count + 2,
                               "File descriptors leaked when handling malformed JSON")
        
        # Verify internal counter
        self.assertEqual(self.config_manager.get_file_handle_count(), 0,
                        "Internal file handle counter not zero")
    
    def test_empty_file_handling(self):
        """Test handling of empty configuration file"""
        config = self.config_manager.load_json_config(self.empty_config_path)
        self.assertIsNone(config)
        
        # Verify no file handle leak
        current_fd_count = self._get_open_file_descriptors()
        if current_fd_count >= 0 and self.initial_fd_count >= 0:
            self.assertLessEqual(current_fd_count, self.initial_fd_count + 1,
                               "File descriptors leaked with empty file")
    
    def test_nonexistent_file_handling(self):
        """Test handling of non-existent configuration file"""
        nonexistent_path = Path(self.temp_dir) / "nonexistent.json"
        config = self.config_manager.load_json_config(nonexistent_path)
        self.assertIsNone(config)
        
        # Verify no file handle was opened
        self.assertEqual(self.config_manager.get_file_handle_count(), 0)
    
    def test_permission_denied_handling(self):
        """Test handling of permission denied errors"""
        # Skip test if running as root (root can read any file)
        if os.getuid() == 0:
            self.skipTest("Cannot test permission denied as root user")
            
        # Create a file with no read permissions
        restricted_path = Path(self.temp_dir) / "restricted.json"
        with open(restricted_path, 'w') as f:
            json.dump(self.valid_config, f)
        
        # Remove read permissions
        os.chmod(restricted_path, 0o000)
        
        try:
            config = self.config_manager.load_json_config(restricted_path)
            self.assertIsNone(config)
            
            # Verify file handle was properly closed
            self.assertEqual(self.config_manager.get_file_handle_count(), 0)
        finally:
            # Restore permissions for cleanup
            os.chmod(restricted_path, 0o644)
    
    def test_concurrent_config_loading(self):
        """Test concurrent configuration loading for thread safety"""
        results = []
        errors = []
        
        def load_config():
            try:
                for _ in range(5):
                    # Mix valid and invalid configs
                    self.config_manager.load_json_config(self.valid_config_path)
                    self.config_manager.load_json_config(self.malformed_config_path)
                    results.append(True)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=load_config)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check no errors occurred
        self.assertEqual(len(errors), 0, f"Errors in concurrent loading: {errors}")
        
        # Verify file handles were cleaned up
        self.assertEqual(self.config_manager.get_file_handle_count(), 0,
                        "File handles leaked in concurrent access")
    
    def test_cache_functionality(self):
        """Test configuration caching reduces file I/O"""
        # First load should read from file
        config1 = self.config_manager.load_json_config(self.valid_config_path)
        self.assertIsNotNone(config1)
        
        # Second load should use cache (within TTL)
        with patch('builtins.open', mock_open()) as mock_file:
            config2 = self.config_manager.load_json_config(self.valid_config_path)
            # open() should not be called due to cache
            mock_file.assert_not_called()
        
        self.assertEqual(config1, config2)
    
    def test_cache_expiration(self):
        """Test cache expiration after TTL"""
        # Set short TTL for testing
        original_ttl = self.config_manager._cache_ttl
        self.config_manager._cache_ttl = 0.1  # 100ms
        
        try:
            # Load config
            config1 = self.config_manager.load_json_config(self.valid_config_path)
            
            # Wait for cache to expire
            time.sleep(0.2)
            
            # This should read from file again
            with patch('builtins.open', mock_open(read_data=json.dumps(self.valid_config))) as mock_file:
                config2 = self.config_manager.load_json_config(self.valid_config_path)
                # open() should be called after cache expiration
                mock_file.assert_called()
            
        finally:
            self.config_manager._cache_ttl = original_ttl
    
    def test_cache_clearing(self):
        """Test manual cache clearing"""
        # Load config (will be cached)
        config1 = self.config_manager.load_json_config(self.valid_config_path)
        
        # Clear cache
        self.config_manager.clear_cache()
        
        # Next load should read from file
        with patch('builtins.open', mock_open(read_data=json.dumps(self.valid_config))) as mock_file:
            config2 = self.config_manager.load_json_config(self.valid_config_path)
            mock_file.assert_called()
    
    def test_engine_initialization_with_config_manager(self):
        """Test SpeechToTextEngine uses ConfigManager properly"""
        # Test with valid config
        engine = SpeechToTextEngine(config_file=str(self.valid_config_path))
        self.assertEqual(engine.manager.default_engine_name, "whisper")
        
        # Test with malformed config (should fallback gracefully)
        engine2 = SpeechToTextEngine(config_file=str(self.malformed_config_path))
        self.assertIsNotNone(engine2.manager)
        
        # Verify no file handle leaks
        current_fd_count = self._get_open_file_descriptors()
        if current_fd_count >= 0 and self.initial_fd_count >= 0:
            self.assertLessEqual(current_fd_count, self.initial_fd_count + 3,
                               "File descriptors leaked in engine initialization")
    
    def test_stress_malformed_json(self):
        """Stress test with many malformed JSON loads"""
        initial_count = self.config_manager.get_file_handle_count()
        
        # Load malformed JSON many times
        for i in range(100):
            config = self.config_manager.load_json_config(self.malformed_config_path)
            self.assertIsNone(config)
            
            # Check counter stays at 0
            self.assertEqual(self.config_manager.get_file_handle_count(), initial_count,
                           f"File handle leak detected at iteration {i}")
        
        # Final check
        current_fd_count = self._get_open_file_descriptors()
        if current_fd_count >= 0 and self.initial_fd_count >= 0:
            # Allow small variance for system file descriptors
            self.assertLessEqual(current_fd_count, self.initial_fd_count + 5,
                               "File descriptors leaked during stress test")
    
    def test_file_handle_monitoring(self):
        """Test file handle monitoring and alerting"""
        # Mock the logger to capture warnings
        with patch('stts.config_manager.logger') as mock_logger:
            # Simulate high file handle count
            for _ in range(101):
                self.config_manager._increment_file_handle_count()
            
            # Should have triggered warning
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            self.assertIn("High file handle count", warning_msg)
        
        # Clean up
        for _ in range(101):
            self.config_manager._decrement_file_handle_count()
    
    def test_singleton_pattern(self):
        """Test ConfigManager singleton pattern"""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        # Should be the same instance
        self.assertIs(manager1, manager2)
        
        # Test thread safety of singleton
        managers = []
        def get_manager():
            managers.append(get_config_manager())
        
        threads = []
        for _ in range(10):
            t = threading.Thread(target=get_manager)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All should be the same instance
        for manager in managers:
            self.assertIs(manager, manager1)
    
    def test_load_config_fallback_chain(self):
        """Test the complete configuration loading fallback chain"""
        config_manager = get_config_manager()
        
        # Test with explicit config
        explicit_config = {"test": "explicit"}
        config, engine = config_manager.load_config(config=explicit_config)
        self.assertEqual(config["test"], "explicit")
        
        # Test with config file
        config, engine = config_manager.load_config(config_file=str(self.valid_config_path))
        self.assertEqual(engine, "whisper")
        
        # Test with environment variable
        os.environ['STT_ENGINE'] = 'vosk'
        config, engine = config_manager.load_config()
        self.assertEqual(engine, 'vosk')
        del os.environ['STT_ENGINE']
        
        # Test default fallback
        config, engine = config_manager.load_config()
        self.assertEqual(engine, 'deepspeech')


class TestFileHandleLeakIntegration(unittest.TestCase):
    """Integration tests for file handle leak detection"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up integration test environment"""
        for file in Path(self.temp_dir).glob("*"):
            file.unlink()
        Path(self.temp_dir).rmdir()
    
    def test_repeated_engine_creation(self):
        """Test repeated engine creation doesn't leak file handles"""
        initial_fd_count = self._get_open_file_descriptors()
        
        # Create invalid config that will fail to parse
        invalid_config_path = Path(self.temp_dir) / "invalid.json"
        with open(invalid_config_path, 'w') as f:
            f.write('{"bad json}')
        
        # Create engines repeatedly with invalid config
        engines = []
        for _ in range(10):
            try:
                engine = SpeechToTextEngine(config_file=str(invalid_config_path))
                engines.append(engine)
            except:
                pass
        
        # Check file descriptors
        current_fd_count = self._get_open_file_descriptors()
        if current_fd_count >= 0 and initial_fd_count >= 0:
            # Allow for some variance but should not grow linearly
            self.assertLessEqual(current_fd_count, initial_fd_count + 5,
                               "File descriptors leaked during repeated engine creation")
    
    def _get_open_file_descriptors(self):
        """Get count of open file descriptors for current process"""
        if HAS_PSUTIL:
            try:
                process = psutil.Process(os.getpid())
                return len(process.open_files())
            except:
                pass
        
        try:
            pid = os.getpid()
            fd_dir = f"/proc/{pid}/fd"
            if os.path.exists(fd_dir):
                return len(os.listdir(fd_dir))
        except:
            pass
        return -1


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)