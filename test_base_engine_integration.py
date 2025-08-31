#!/usr/bin/env python3
"""
Integration test for base_engine with FFmpeg process cleanup
"""

import sys
import os
import time
import logging
import threading

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from stts.base_engine import BaseSTTEngine, FFmpegProcessManager
import numpy as np


class TestEngine(BaseSTTEngine):
    """Test STT engine for integration testing"""
    
    def initialize(self):
        """Initialize test engine"""
        pass
    
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Mock transcription"""
        return f"Test transcription: {len(audio_data)} samples at {sample_rate}Hz"


def create_test_wav():
    """Create a minimal valid WAV file"""
    wav_header = b'RIFF' + b'\x24\x08\x00\x00' + b'WAVE'
    wav_header += b'fmt ' + b'\x10\x00\x00\x00'  # fmt chunk size
    wav_header += b'\x01\x00'  # PCM format
    wav_header += b'\x01\x00'  # 1 channel
    wav_header += b'\x80\x3e\x00\x00'  # 16000 Hz sample rate
    wav_header += b'\x00\x7d\x00\x00'  # byte rate
    wav_header += b'\x02\x00'  # block align
    wav_header += b'\x10\x00'  # 16 bits per sample
    wav_header += b'data' + b'\x00\x08\x00\x00'  # data chunk with size
    
    # Add 1024 samples of silence (2048 bytes for 16-bit audio)
    audio_data = b'\x00' * 2048
    
    return wav_header + audio_data


def test_normal_operation():
    """Test normal audio normalization"""
    print("\n=== Testing Normal Operation ===")
    
    engine = TestEngine()
    test_audio = create_test_wav()
    
    print(f"Initial active processes: {FFmpegProcessManager.get_active_process_count()}")
    
    try:
        # This will use FFmpeg to normalize the audio
        normalized = engine.normalize_audio(test_audio)
        print(f"Normalization successful: {len(normalized)} bytes")
        
        # Test transcription
        result = engine.transcribe(test_audio)
        print(f"Transcription result: {result}")
        
    except Exception as e:
        print(f"Error during normalization: {e}")
        return False
    
    # Check cleanup
    time.sleep(0.5)
    final_count = FFmpegProcessManager.get_active_process_count()
    print(f"Final active processes: {final_count}")
    
    if final_count > 0:
        print("WARNING: Processes not cleaned up!")
        return False
    
    return True


def test_concurrent_normalization():
    """Test multiple concurrent normalizations"""
    print("\n=== Testing Concurrent Normalization ===")
    
    initial_count = FFmpegProcessManager.get_active_process_count()
    print(f"Initial active processes: {initial_count}")
    
    results = []
    errors = []
    
    def normalize_audio(engine, audio, index):
        try:
            normalized = engine.normalize_audio(audio)
            results.append((index, len(normalized)))
        except Exception as e:
            errors.append((index, str(e)))
    
    # Create multiple engines and test audio
    engines = [TestEngine() for _ in range(5)]
    test_audio = create_test_wav()
    
    # Start concurrent normalizations
    threads = []
    for i, engine in enumerate(engines):
        t = threading.Thread(target=normalize_audio, args=(engine, test_audio, i))
        t.start()
        threads.append(t)
    
    # Wait for completion
    for t in threads:
        t.join(timeout=10)
    
    print(f"Successful normalizations: {len(results)}")
    print(f"Failed normalizations: {len(errors)}")
    
    # Check cleanup
    time.sleep(1)
    final_count = FFmpegProcessManager.get_active_process_count()
    print(f"Final active processes: {final_count}")
    
    if final_count > initial_count:
        print(f"WARNING: Process leak detected! {final_count - initial_count} processes not cleaned up")
        return False
    
    return len(results) == 5 and len(errors) == 0


def test_invalid_audio():
    """Test with invalid audio data"""
    print("\n=== Testing Invalid Audio ===")
    
    engine = TestEngine()
    invalid_audio = b'NOT_A_VALID_AUDIO_FILE'
    
    initial_count = FFmpegProcessManager.get_active_process_count()
    print(f"Initial active processes: {initial_count}")
    
    try:
        normalized = engine.normalize_audio(invalid_audio)
        print("ERROR: Should have failed with invalid audio!")
        return False
    except Exception as e:
        print(f"Expected error occurred: {e}")
    
    # Check cleanup even after error
    time.sleep(0.5)
    final_count = FFmpegProcessManager.get_active_process_count()
    print(f"Final active processes: {final_count}")
    
    if final_count > initial_count:
        print("WARNING: Process not cleaned up after error!")
        return False
    
    return True


def test_timeout_handling():
    """Test timeout handling with large file"""
    print("\n=== Testing Timeout Handling ===")
    
    # Temporarily reduce timeout for testing
    import stts.base_engine
    original_timeout = stts.base_engine.FFMPEG_TIMEOUT
    stts.base_engine.FFMPEG_TIMEOUT = 0.1  # Very short timeout
    
    try:
        engine = TestEngine()
        
        # Create a large audio file that will take time to process
        large_audio = create_test_wav() * 1000  # Repeat the audio many times
        
        initial_count = FFmpegProcessManager.get_active_process_count()
        print(f"Initial active processes: {initial_count}")
        print(f"Testing with {len(large_audio)} byte file and {stts.base_engine.FFMPEG_TIMEOUT}s timeout")
        
        try:
            normalized = engine.normalize_audio(large_audio)
            print("Normalization completed (file might have been small enough)")
        except Exception as e:
            if "timed out" in str(e):
                print(f"Expected timeout occurred: {e}")
            else:
                print(f"Unexpected error: {e}")
        
        # Check cleanup after timeout
        time.sleep(1)
        final_count = FFmpegProcessManager.get_active_process_count()
        print(f"Final active processes: {final_count}")
        
        if final_count > initial_count:
            print("WARNING: Process not cleaned up after timeout!")
            return False
        
        return True
        
    finally:
        # Restore original timeout
        stts.base_engine.FFMPEG_TIMEOUT = original_timeout


def test_emergency_cleanup():
    """Test emergency cleanup functionality"""
    print("\n=== Testing Emergency Cleanup ===")
    
    # Start some processes
    engines = [TestEngine() for _ in range(3)]
    test_audio = create_test_wav()
    
    def long_running_normalization(engine, audio):
        try:
            # This will block in the context manager
            normalized = engine.normalize_audio(audio)
        except Exception:
            pass
    
    # Start processes in background threads
    threads = []
    for engine in engines:
        t = threading.Thread(target=long_running_normalization, args=(engine, test_audio))
        t.daemon = True  # Daemon threads so they don't block exit
        t.start()
        threads.append(t)
    
    # Give processes time to start
    time.sleep(0.5)
    
    active_before = FFmpegProcessManager.get_active_process_count()
    print(f"Active processes before cleanup: {active_before}")
    
    # Emergency cleanup
    FFmpegProcessManager.terminate_all_processes()
    print("Emergency cleanup executed")
    
    # Check cleanup
    time.sleep(0.5)
    active_after = FFmpegProcessManager.get_active_process_count()
    print(f"Active processes after cleanup: {active_after}")
    
    return active_after == 0


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("FFmpeg Process Cleanup Integration Tests")
    print("=" * 60)
    
    # Check if FFmpeg is installed
    if os.system('which ffmpeg > /dev/null 2>&1') != 0:
        print("\nWARNING: FFmpeg not installed. Installing...")
        os.system('apt-get update && apt-get install -y ffmpeg')
    
    tests = [
        ("Normal Operation", test_normal_operation),
        ("Concurrent Normalization", test_concurrent_normalization),
        ("Invalid Audio Handling", test_invalid_audio),
        ("Timeout Handling", test_timeout_handling),
        ("Emergency Cleanup", test_emergency_cleanup),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\nTest '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name}: {status}")
    
    # Final check for any lingering processes
    print("\n" + "=" * 60)
    print("Final System Check")
    print("=" * 60)
    
    final_processes = FFmpegProcessManager.get_active_process_count()
    print(f"Active FFmpeg processes: {final_processes}")
    
    # Check for zombies
    try:
        import psutil
        zombies = []
        for proc in psutil.process_iter(['pid', 'name', 'status']):
            try:
                if proc.info['status'] == psutil.STATUS_ZOMBIE:
                    zombies.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if zombies:
            print(f"WARNING: Found {len(zombies)} zombie processes!")
            for z in zombies[:5]:  # Show first 5
                print(f"  - PID {z['pid']}: {z['name']}")
        else:
            print("No zombie processes detected ✓")
    except ImportError:
        print("psutil not available for zombie check")
    
    # Return success if all tests passed
    all_passed = all(success for _, success in results)
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())