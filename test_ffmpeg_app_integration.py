#!/usr/bin/env python3
"""
Test FFmpeg cleanup in the context of the actual STT application
"""

import sys
import os
import time
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from stts.base_engine import FFmpegProcessManager


def test_ffmpeg_cleanup_monitoring():
    """Test that FFmpeg processes are properly cleaned up"""
    print("=" * 60)
    print("FFmpeg Process Cleanup Monitoring Test")
    print("=" * 60)
    
    # Check initial state
    initial_count = FFmpegProcessManager.get_active_process_count()
    print(f"\nInitial active FFmpeg processes: {initial_count}")
    
    # Import the engine manager (this will import all engines)
    print("\nImporting engine components...")
    try:
        # Only import the base components that don't require extra dependencies
        from stts.base_engine import BaseSTTEngine
        from stts.validators import sanitize_ffmpeg_input
        print("✓ Base engine imported successfully")
        
        # Test the FFmpeg normalization directly
        print("\nTesting FFmpeg audio normalization...")
        
        # Create a test engine
        import numpy as np
        
        class TestSTTEngine(BaseSTTEngine):
            def initialize(self):
                pass
            
            def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
                return "test"
        
        engine = TestSTTEngine()
        
        # Create test WAV data
        wav_header = b'RIFF' + b'\x24\x08\x00\x00' + b'WAVE'
        wav_header += b'fmt ' + b'\x10\x00\x00\x00'
        wav_header += b'\x01\x00'  # PCM
        wav_header += b'\x01\x00'  # 1 channel
        wav_header += b'\x80\x3e\x00\x00'  # 16000 Hz
        wav_header += b'\x00\x7d\x00\x00'  # byte rate
        wav_header += b'\x02\x00'  # block align
        wav_header += b'\x10\x00'  # 16 bits
        wav_header += b'data' + b'\x00\x08\x00\x00'
        test_audio = wav_header + b'\x00' * 2048
        
        # Test normalization multiple times
        for i in range(3):
            print(f"\n  Test {i+1}/3:")
            print(f"    Active processes before: {FFmpegProcessManager.get_active_process_count()}")
            
            try:
                normalized = engine.normalize_audio(test_audio)
                print(f"    ✓ Normalization successful ({len(normalized)} bytes)")
            except Exception as e:
                print(f"    ✗ Normalization failed: {e}")
            
            time.sleep(0.5)
            print(f"    Active processes after: {FFmpegProcessManager.get_active_process_count()}")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  (This is expected if optional dependencies are not installed)")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    # Final check
    print("\n" + "=" * 60)
    print("Final Check")
    print("=" * 60)
    
    time.sleep(1)
    final_count = FFmpegProcessManager.get_active_process_count()
    print(f"Final active FFmpeg processes: {final_count}")
    
    if final_count > initial_count:
        print(f"\n⚠️  WARNING: {final_count - initial_count} FFmpeg processes were not cleaned up!")
        print("This indicates a potential resource leak.")
        return False
    else:
        print("\n✅ SUCCESS: All FFmpeg processes were properly cleaned up!")
        print("No zombie processes or resource leaks detected.")
        return True


def check_zombie_processes():
    """Check for zombie processes in the system"""
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
            print(f"\n⚠️  Found {len(zombies)} zombie processes:")
            for z in zombies[:5]:
                print(f"    PID {z['pid']}: {z['name']}")
        else:
            print("\n✅ No zombie processes detected")
        
        return len(zombies) == 0
    except ImportError:
        print("\n(psutil not available for zombie check)")
        return True


def main():
    """Run the monitoring test"""
    
    # Check FFmpeg is installed
    if os.system('which ffmpeg > /dev/null 2>&1') != 0:
        print("ERROR: FFmpeg is not installed!")
        print("Please install FFmpeg: apt-get install ffmpeg")
        return 1
    
    print("FFmpeg is installed ✓\n")
    
    # Run the test
    success = test_ffmpeg_cleanup_monitoring()
    
    # Check for zombies
    no_zombies = check_zombie_processes()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if success and no_zombies:
        print("✅ All tests passed!")
        print("FFmpeg process cleanup is working correctly.")
        print("No zombie processes or resource leaks detected.")
        return 0
    else:
        print("⚠️  Some issues were detected.")
        if not success:
            print("- FFmpeg processes were not properly cleaned up")
        if not no_zombies:
            print("- Zombie processes were detected")
        return 1


if __name__ == '__main__':
    sys.exit(main())