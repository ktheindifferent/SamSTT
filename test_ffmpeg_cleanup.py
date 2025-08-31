#!/usr/bin/env python3
"""
Unit tests for FFmpeg process cleanup functionality
Tests various scenarios to ensure no zombie processes are created
"""

import unittest
import subprocess
import time
import threading
import os
import signal
import psutil
import tempfile
from unittest.mock import patch, MagicMock
from io import BytesIO

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stts.base_engine import FFmpegProcessManager, ProcessTimeoutError


class TestFFmpegProcessCleanup(unittest.TestCase):
    """Test FFmpeg process management and cleanup"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_audio = b'RIFF' + b'\x00' * 100  # Minimal WAV header
        self.manager = None
        
    def tearDown(self):
        """Clean up after tests"""
        # Ensure all processes are terminated
        FFmpegProcessManager.terminate_all_processes()
        
    def get_zombie_processes(self):
        """Get count of zombie processes on the system"""
        zombies = []
        for proc in psutil.process_iter(['pid', 'name', 'status']):
            try:
                if proc.info['status'] == psutil.STATUS_ZOMBIE:
                    zombies.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return zombies
    
    def test_normal_process_completion(self):
        """Test that process cleanup works for normal completion"""
        initial_count = FFmpegProcessManager.get_active_process_count()
        
        # Create a simple command that completes quickly
        cmd = ['sleep', '0.1']
        manager = FFmpegProcessManager(timeout=5)
        
        # Track if we're inside the context manager
        inside_context = False
        
        with manager.run_process(cmd, b''):
            inside_context = True
            # Process should be tracked while running
            active = FFmpegProcessManager.get_active_process_count()
            self.assertGreaterEqual(active, initial_count)
        
        # Process should be cleaned up
        time.sleep(0.2)  # Allow cleanup to complete
        self.assertEqual(FFmpegProcessManager.get_active_process_count(), initial_count)
        
        # Check no zombies were created
        zombies = self.get_zombie_processes()
        self.assertEqual(len(zombies), 0, f"Found zombie processes: {zombies}")
    
    def test_process_timeout_cleanup(self):
        """Test that processes are cleaned up on timeout"""
        initial_count = FFmpegProcessManager.get_active_process_count()
        
        # Create a command that will timeout
        cmd = ['sleep', '10']
        manager = FFmpegProcessManager(timeout=0.5)
        
        with self.assertRaises(ProcessTimeoutError):
            with manager.run_process(cmd, b''):
                pass
        
        # Process should be terminated and cleaned up
        time.sleep(0.5)  # Allow cleanup to complete
        self.assertEqual(FFmpegProcessManager.get_active_process_count(), initial_count)
        
        # Check no zombies were created
        zombies = self.get_zombie_processes()
        self.assertEqual(len(zombies), 0, f"Found zombie processes: {zombies}")
    
    def test_process_error_cleanup(self):
        """Test cleanup when process exits with error"""
        initial_count = FFmpegProcessManager.get_active_process_count()
        
        # Create a command that will fail
        cmd = ['false']  # Always exits with code 1
        manager = FFmpegProcessManager(timeout=5)
        
        with self.assertRaises(subprocess.CalledProcessError):
            with manager.run_process(cmd, b''):
                pass
        
        # Process should be cleaned up
        time.sleep(0.2)  # Allow cleanup to complete
        self.assertEqual(FFmpegProcessManager.get_active_process_count(), initial_count)
        
        # Check no zombies were created
        zombies = self.get_zombie_processes()
        self.assertEqual(len(zombies), 0, f"Found zombie processes: {zombies}")
    
    def test_concurrent_processes(self):
        """Test that multiple concurrent processes are tracked and cleaned up"""
        initial_count = FFmpegProcessManager.get_active_process_count()
        
        def run_process(duration, should_timeout=False):
            cmd = ['sleep', str(duration)]
            timeout = 0.5 if should_timeout else duration + 1
            manager = FFmpegProcessManager(timeout=timeout)
            
            try:
                with manager.run_process(cmd, b''):
                    pass
            except (ProcessTimeoutError, subprocess.CalledProcessError):
                pass  # Expected for some test cases
        
        # Start multiple processes
        threads = []
        for i in range(5):
            t = threading.Thread(target=run_process, args=(0.1 + i * 0.1, i % 2 == 0))
            t.start()
            threads.append(t)
        
        # Wait a bit for processes to start
        time.sleep(0.2)
        
        # Should have multiple active processes
        active = FFmpegProcessManager.get_active_process_count()
        self.assertGreater(active, initial_count)
        
        # Wait for all threads to complete
        for t in threads:
            t.join(timeout=3)
        
        # All processes should be cleaned up
        time.sleep(0.5)  # Allow cleanup to complete
        self.assertEqual(FFmpegProcessManager.get_active_process_count(), initial_count)
        
        # Check no zombies were created
        zombies = self.get_zombie_processes()
        self.assertEqual(len(zombies), 0, f"Found zombie processes: {zombies}")
    
    def test_file_descriptor_cleanup(self):
        """Test that file descriptors are properly closed"""
        import resource
        
        # Get initial file descriptor count
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        initial_fds = len(os.listdir('/proc/self/fd'))
        
        # Run multiple processes
        for i in range(10):
            cmd = ['echo', f'test{i}']
            manager = FFmpegProcessManager(timeout=5)
            
            try:
                with manager.run_process(cmd, b'input'):
                    pass
            except Exception:
                pass
        
        # Check file descriptors were released
        time.sleep(0.5)  # Allow cleanup
        final_fds = len(os.listdir('/proc/self/fd'))
        
        # Allow for some variance but should be close to initial
        self.assertLessEqual(final_fds, initial_fds + 5, 
                           f"File descriptor leak: {initial_fds} -> {final_fds}")
    
    def test_signal_handling(self):
        """Test that processes handle signals properly"""
        cmd = ['sleep', '10']
        manager = FFmpegProcessManager(timeout=10)
        
        def send_signal():
            time.sleep(0.5)
            if manager.process:
                os.kill(manager.process.pid, signal.SIGTERM)
        
        # Start process and send signal from another thread
        signal_thread = threading.Thread(target=send_signal)
        signal_thread.start()
        
        with self.assertRaises(subprocess.CalledProcessError):
            with manager.run_process(cmd, b''):
                pass
        
        signal_thread.join()
        
        # Process should be cleaned up
        time.sleep(0.2)
        self.assertEqual(FFmpegProcessManager.get_active_process_count(), 0)
        
        # Check no zombies
        zombies = self.get_zombie_processes()
        self.assertEqual(len(zombies), 0, f"Found zombie processes: {zombies}")
    
    def test_emergency_cleanup(self):
        """Test emergency cleanup of all processes"""
        # Start multiple processes
        managers = []
        for i in range(3):
            cmd = ['sleep', '10']
            manager = FFmpegProcessManager(timeout=20)
            
            # Start process in thread to avoid blocking
            def start_proc(mgr, command):
                try:
                    with mgr.run_process(command, b''):
                        time.sleep(10)
                except Exception:
                    pass
            
            t = threading.Thread(target=start_proc, args=(manager, cmd))
            t.start()
            managers.append((manager, t))
        
        # Wait for processes to start
        time.sleep(0.5)
        
        # Should have active processes
        self.assertGreater(FFmpegProcessManager.get_active_process_count(), 0)
        
        # Emergency cleanup
        FFmpegProcessManager.terminate_all_processes()
        
        # Wait for cleanup
        time.sleep(1)
        
        # All processes should be terminated
        self.assertEqual(FFmpegProcessManager.get_active_process_count(), 0)
        
        # Clean up threads
        for manager, thread in managers:
            thread.join(timeout=1)
    
    def test_process_monitoring_logs(self):
        """Test that process monitoring generates appropriate logs"""
        with self.assertLogs('stts.base_engine', level='DEBUG') as cm:
            cmd = ['echo', 'test']
            manager = FFmpegProcessManager(timeout=5)
            
            with manager.run_process(cmd, b''):
                pass
        
        # Check that monitoring logs were generated
        log_output = '\n'.join(cm.output)
        self.assertIn('Started FFmpeg process', log_output)
        self.assertIn('exited with code', log_output)
        self.assertIn('Active FFmpeg processes remaining', log_output)
    
    def test_stress_test_no_zombies(self):
        """Stress test: Run 100+ concurrent operations and check for zombies"""
        initial_zombies = len(self.get_zombie_processes())
        initial_count = FFmpegProcessManager.get_active_process_count()
        
        def stress_operation(index):
            # Mix of successful, failing, and timeout operations
            if index % 3 == 0:
                cmd = ['sleep', '0.1']  # Success
                timeout = 5
            elif index % 3 == 1:
                cmd = ['false']  # Failure
                timeout = 5
            else:
                cmd = ['sleep', '10']  # Timeout
                timeout = 0.1
            
            manager = FFmpegProcessManager(timeout=timeout)
            
            try:
                with manager.run_process(cmd, b'test'):
                    pass
            except (ProcessTimeoutError, subprocess.CalledProcessError):
                pass  # Expected
        
        # Run stress test
        threads = []
        for i in range(100):
            t = threading.Thread(target=stress_operation, args=(i,))
            t.start()
            threads.append(t)
            
            # Stagger thread starts slightly
            if i % 10 == 0:
                time.sleep(0.01)
        
        # Wait for all operations to complete
        for t in threads:
            t.join(timeout=10)
        
        # Allow final cleanup
        time.sleep(1)
        
        # Check results
        final_count = FFmpegProcessManager.get_active_process_count()
        final_zombies = len(self.get_zombie_processes())
        
        self.assertEqual(final_count, initial_count, 
                        f"Process leak: {initial_count} -> {final_count}")
        self.assertEqual(final_zombies, initial_zombies, 
                        f"Zombie processes created: {initial_zombies} -> {final_zombies}")


class TestFFmpegIntegration(unittest.TestCase):
    """Integration tests with actual FFmpeg commands"""
    
    @unittest.skipUnless(os.system('which ffmpeg > /dev/null 2>&1') == 0, 
                         "FFmpeg not installed")
    def test_real_ffmpeg_normal(self):
        """Test with real FFmpeg command that succeeds"""
        # Create a minimal valid WAV file
        wav_header = b'RIFF' + b'\x24\x00\x00\x00' + b'WAVE'
        wav_header += b'fmt ' + b'\x10\x00\x00\x00'  # fmt chunk size
        wav_header += b'\x01\x00'  # PCM format
        wav_header += b'\x01\x00'  # 1 channel
        wav_header += b'\x80\x3e\x00\x00'  # 16000 Hz sample rate
        wav_header += b'\x00\x7d\x00\x00'  # byte rate
        wav_header += b'\x02\x00'  # block align
        wav_header += b'\x10\x00'  # 16 bits per sample
        wav_header += b'data' + b'\x00\x00\x00\x00'  # data chunk
        
        test_audio = wav_header + b'\x00' * 1000  # Some silence
        
        # Run FFmpeg to convert to 16kHz mono
        cmd = ['ffmpeg', '-f', 'wav', '-i', 'pipe:0', 
               '-f', 'wav', '-acodec', 'pcm_s16le', 
               '-ac', '1', '-ar', '16000', 'pipe:1']
        
        manager = FFmpegProcessManager(timeout=5)
        
        with manager.run_process(cmd, test_audio) as (out, err):
            self.assertIsNotNone(out)
            self.assertGreater(len(out), 0)
        
        # Check cleanup
        self.assertEqual(FFmpegProcessManager.get_active_process_count(), 0)
    
    @unittest.skipUnless(os.system('which ffmpeg > /dev/null 2>&1') == 0, 
                         "FFmpeg not installed")
    def test_real_ffmpeg_invalid_input(self):
        """Test with real FFmpeg command that fails due to invalid input"""
        invalid_audio = b'NOT_A_VALID_AUDIO_FILE'
        
        cmd = ['ffmpeg', '-f', 'wav', '-i', 'pipe:0', 
               '-f', 'wav', '-acodec', 'pcm_s16le', 
               '-ac', '1', '-ar', '16000', 'pipe:1']
        
        manager = FFmpegProcessManager(timeout=5)
        
        with self.assertRaises(subprocess.CalledProcessError):
            with manager.run_process(cmd, invalid_audio):
                pass
        
        # Check cleanup
        time.sleep(0.5)
        self.assertEqual(FFmpegProcessManager.get_active_process_count(), 0)
    
    @unittest.skipUnless(os.system('which ffmpeg > /dev/null 2>&1') == 0, 
                         "FFmpeg not installed")  
    def test_real_ffmpeg_timeout(self):
        """Test FFmpeg timeout with a command that hangs"""
        # Use a large file that will take time to process
        large_audio = b'RIFF' + b'\xFF' * (10 * 1024 * 1024)  # 10MB of data
        
        # Complex filter that will be slow
        cmd = ['ffmpeg', '-f', 'lavfi', '-i', 'anullsrc=duration=1000', 
               '-filter_complex', 'aecho=0.8:0.9:1000:0.3,aecho=0.8:0.9:1000:0.3',
               '-f', 'wav', 'pipe:1']
        
        manager = FFmpegProcessManager(timeout=0.5)  # Very short timeout
        
        with self.assertRaises(ProcessTimeoutError):
            with manager.run_process(cmd, b''):
                pass
        
        # Check cleanup
        time.sleep(1)
        self.assertEqual(FFmpegProcessManager.get_active_process_count(), 0)


if __name__ == '__main__':
    # Set up logging for debugging
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if psutil is installed
    try:
        import psutil
    except ImportError:
        print("WARNING: psutil not installed. Some tests will have limited functionality.")
        print("Install with: pip install psutil")
    
    # Run tests
    unittest.main(verbosity=2)