#!/usr/bin/env python3
"""
Unit tests for NeMo engine resource leak fixes
Tests temporary file cleanup in various failure scenarios
"""

import unittest
import tempfile
import os
import glob
import time
import threading
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stts.engines.nemo import NeMoEngine


class TestNeMoTempFileHandling(unittest.TestCase):
    """Test NeMo engine temporary file handling and cleanup"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.gettempdir()
        self.engine = None
        
    def tearDown(self):
        """Clean up after tests"""
        if self.engine:
            self.engine._cleanup_orphaned_files()
    
    def get_nemo_temp_files(self):
        """Get list of NeMo temp files in temp directory"""
        pattern = os.path.join(self.temp_dir, 'nemo_*.wav')
        return glob.glob(pattern)
    
    def test_normal_transcription_cleanup(self):
        """Test that temp files are cleaned up after successful transcription"""
        with patch('stts.engines.nemo.NeMoEngine.initialize'):
            engine = NeMoEngine()
            engine.model = Mock()
            engine.model.transcribe = Mock(return_value=["test transcription"])
            
            # Count temp files before
            files_before = self.get_nemo_temp_files()
            
            # Mock audio data
            audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
            
            # Transcribe
            with patch('soundfile.write'):
                result = engine.transcribe_raw(audio_data)
            
            # Check result
            self.assertEqual(result, "test transcription")
            
            # Count temp files after - should be same as before
            files_after = self.get_nemo_temp_files()
            self.assertEqual(len(files_after), len(files_before),
                           "Temp file not cleaned up after successful transcription")
    
    def test_exception_during_transcription_cleanup(self):
        """Test that temp files are cleaned up even when transcription fails"""
        with patch('stts.engines.nemo.NeMoEngine.initialize'):
            engine = NeMoEngine()
            engine.model = Mock()
            engine.model.transcribe = Mock(side_effect=Exception("Transcription failed"))
            
            # Count temp files before
            files_before = self.get_nemo_temp_files()
            
            # Mock audio data
            audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
            
            # Transcribe should raise exception
            with patch('soundfile.write'):
                with self.assertRaises(Exception) as context:
                    engine.transcribe_raw(audio_data)
            
            self.assertIn("Transcription failed", str(context.exception))
            
            # Count temp files after - should be same as before
            files_after = self.get_nemo_temp_files()
            self.assertEqual(len(files_after), len(files_before),
                           "Temp file not cleaned up after transcription failure")
    
    def test_exception_during_file_write_cleanup(self):
        """Test cleanup when file write fails"""
        with patch('stts.engines.nemo.NeMoEngine.initialize'):
            engine = NeMoEngine()
            engine.model = Mock()
            
            # Count temp files before
            files_before = self.get_nemo_temp_files()
            
            # Mock audio data
            audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
            
            # Make soundfile.write fail
            with patch('soundfile.write', side_effect=Exception("Write failed")):
                with self.assertRaises(Exception) as context:
                    engine.transcribe_raw(audio_data)
            
            self.assertIn("Write failed", str(context.exception))
            
            # Count temp files after - should be same as before
            files_after = self.get_nemo_temp_files()
            self.assertEqual(len(files_after), len(files_before),
                           "Temp file not cleaned up after write failure")
    
    def test_orphaned_file_cleanup_on_init(self):
        """Test that orphaned files are cleaned up on initialization"""
        # Create some old orphaned files
        old_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(
                prefix='nemo_',
                suffix='.wav',
                dir=self.temp_dir,
                delete=False
            ) as f:
                old_files.append(f.name)
                # Make file appear old (2 hours ago)
                old_time = time.time() - 7200
                os.utime(f.name, (old_time, old_time))
        
        # Create a recent file (should not be deleted)
        with tempfile.NamedTemporaryFile(
            prefix='nemo_',
            suffix='.wav',
            dir=self.temp_dir,
            delete=False
        ) as f:
            recent_file = f.name
        
        # Initialize engine (should trigger cleanup)
        with patch('stts.engines.nemo.NeMoEngine.initialize'):
            engine = NeMoEngine()
        
        # Check that old files were deleted
        for old_file in old_files:
            self.assertFalse(os.path.exists(old_file),
                           f"Old orphaned file {old_file} not cleaned up")
        
        # Check that recent file was not deleted
        self.assertTrue(os.path.exists(recent_file),
                       "Recent file was incorrectly deleted")
        
        # Clean up recent file
        os.unlink(recent_file)
    
    def test_concurrent_transcriptions(self):
        """Test that concurrent transcriptions don't interfere with each other"""
        with patch('stts.engines.nemo.NeMoEngine.initialize'):
            engine = NeMoEngine()
            engine.model = Mock()
            
            # Track which files were used
            used_files = []
            files_lock = threading.Lock()
            
            def mock_transcribe(file_paths):
                """Mock transcribe that records file paths and delays"""
                with files_lock:
                    used_files.extend(file_paths)
                time.sleep(0.1)  # Simulate processing time
                return [f"transcription_{os.path.basename(fp)}" for fp in file_paths]
            
            engine.model.transcribe = mock_transcribe
            
            # Count temp files before
            files_before = self.get_nemo_temp_files()
            
            # Run concurrent transcriptions
            def transcribe_task(task_id):
                audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
                with patch('soundfile.write'):
                    result = engine.transcribe_raw(audio_data)
                return task_id, result
            
            num_concurrent = 10
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(transcribe_task, i) for i in range(num_concurrent)]
                results = [future.result() for future in as_completed(futures)]
            
            # All tasks should complete
            self.assertEqual(len(results), num_concurrent)
            
            # All temp files should be cleaned up
            files_after = self.get_nemo_temp_files()
            self.assertEqual(len(files_after), len(files_before),
                           f"Temp files not cleaned up after concurrent transcriptions. "
                           f"Before: {len(files_before)}, After: {len(files_after)}")
            
            # Each transcription should have used a unique file
            self.assertEqual(len(used_files), num_concurrent,
                           "Not all transcriptions used unique files")
            self.assertEqual(len(set(used_files)), num_concurrent,
                           "File name collision detected in concurrent transcriptions")
    
    def test_cleanup_with_permission_error(self):
        """Test graceful handling when file deletion fails"""
        with patch('stts.engines.nemo.NeMoEngine.initialize'):
            engine = NeMoEngine()
            engine.model = Mock()
            engine.model.transcribe = Mock(return_value=["test"])
            
            # Mock audio data
            audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
            
            # Make os.unlink fail
            with patch('soundfile.write'):
                with patch('os.unlink', side_effect=PermissionError("No permission")):
                    with patch('stts.engines.nemo.logger') as mock_logger:
                        result = engine.transcribe_raw(audio_data)
                        
                        # Should still return result
                        self.assertEqual(result, "test")
                        
                        # Should log warning about failed cleanup
                        mock_logger.warning.assert_called()
                        warning_msg = str(mock_logger.warning.call_args)
                        self.assertIn("Failed to delete temp file", warning_msg)
    
    def test_disk_space_monitoring(self):
        """Test that we can monitor disk usage in temp directory"""
        temp_dir = tempfile.gettempdir()
        
        # Get disk usage stats
        stat = os.statvfs(temp_dir)
        free_bytes = stat.f_bavail * stat.f_frsize
        total_bytes = stat.f_blocks * stat.f_frsize
        used_percent = ((total_bytes - free_bytes) / total_bytes) * 100
        
        # Just verify we can get stats (not testing specific values)
        self.assertGreaterEqual(free_bytes, 0)
        self.assertGreaterEqual(total_bytes, 0)
        self.assertGreaterEqual(used_percent, 0)
        self.assertLessEqual(used_percent, 100)
        
        print(f"Temp dir disk usage: {used_percent:.1f}% "
              f"({free_bytes / (1024**3):.2f} GB free)")


class TestNeMoEngineIntegration(unittest.TestCase):
    """Integration tests for NeMo engine (requires NeMo to be installed)"""
    
    @unittest.skipUnless(
        'INTEGRATION_TEST' in os.environ,
        "Set INTEGRATION_TEST env var to run integration tests"
    )
    def test_real_transcription_with_cleanup(self):
        """Test actual NeMo transcription with file cleanup"""
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            self.skipTest("NeMo not installed")
        
        engine = NeMoEngine()
        engine.initialize()
        
        # Generate test audio
        sample_rate = 16000
        duration = 2  # seconds
        frequency = 440  # Hz (A4 note)
        t = np.linspace(0, duration, sample_rate * duration)
        audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        
        # Count temp files before
        temp_dir = tempfile.gettempdir()
        pattern = os.path.join(temp_dir, 'nemo_*.wav')
        files_before = glob.glob(pattern)
        
        # Transcribe
        result = engine.transcribe_raw(audio_data, sample_rate)
        
        # Count temp files after
        files_after = glob.glob(pattern)
        
        # Should have cleaned up
        self.assertEqual(len(files_after), len(files_before),
                       "Temp file not cleaned up after real transcription")
        
        print(f"Real transcription result: '{result}'")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)