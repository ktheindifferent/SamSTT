#!/usr/bin/env python3
"""Test the actual thread-safe implementation with real engine manager"""

import sys
import os
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest import mock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)-10s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock all dependencies
mock_modules = [
    'numpy', 'scipy', 'scipy.io', 'scipy.io.wavfile',
    'ffmpeg', 'sanic', 'torch', 'torchaudio', 'omegaconf',
    'transformers', 'librosa', 'speechbrain', 'nemo_toolkit',
    'pocketsphinx', 'vosk', 'STT', 'stt', 'whisper'
]

for module_name in mock_modules:
    sys.modules[module_name] = mock.MagicMock()

from stts.engine_manager import STTEngineManager


class ThreadSafeTestEngine:
    """Test engine that tracks initialization attempts"""
    
    _init_lock = threading.Lock()
    _init_counts = {}
    
    def __init__(self, config):
        engine_name = self.__class__.__name__
        
        # Track initialization
        with self._init_lock:
            if engine_name not in self._init_counts:
                self._init_counts[engine_name] = 0
            self._init_counts[engine_name] += 1
            count = self._init_counts[engine_name]
        
        # Simulate some initialization work
        time.sleep(0.05)
        
        self.config = config
        self.name = engine_name
        self.is_available = True
        
        logger.info(f"{engine_name} initialized (attempt #{count}) by {threading.current_thread().name}")
        
        if count > 1:
            logger.error(f"RACE CONDITION: {engine_name} initialized {count} times!")
    
    @classmethod
    def reset_counts(cls):
        cls._init_counts = {}
    
    @classmethod
    def get_count(cls, name):
        return cls._init_counts.get(name, 0)


def test_lazy_initialization_thread_safety():
    """Test thread-safe lazy initialization of engines"""
    
    logger.info("=" * 60)
    logger.info("Testing Thread-Safe Lazy Initialization")
    logger.info("=" * 60)
    
    # Create different test engine classes
    class TestWhisperEngine(ThreadSafeTestEngine):
        pass
    
    class TestVoskEngine(ThreadSafeTestEngine):
        pass
    
    class TestSileroEngine(ThreadSafeTestEngine):
        pass
    
    # Replace engine classes
    STTEngineManager.ENGINES = {
        'whisper': TestWhisperEngine,
        'vosk': TestVoskEngine,
        'silero': TestSileroEngine
    }
    
    # Reset counts
    ThreadSafeTestEngine.reset_counts()
    
    # Create manager WITHOUT initializing any engines at startup
    config = {
        'initialize_all': False,  # Don't initialize anything
        # Don't provide any engine-specific config to prevent initialization
    }
    
    manager = STTEngineManager(default_engine='whisper', config=config)
    
    # Verify thread safety attributes
    assert hasattr(manager, '_engine_locks'), "Missing _engine_locks"
    assert hasattr(manager, '_locks_lock'), "Missing _locks_lock"
    logger.info("✓ Thread safety attributes initialized")
    
    # Clear any engines that might have been initialized
    manager.engines.clear()
    ThreadSafeTestEngine.reset_counts()
    
    # Test: Multiple threads trying to lazily initialize the same engine
    logger.info("\nTest: 20 concurrent threads requesting uninitialized 'whisper' engine")
    logger.info("-" * 40)
    
    def request_engine(engine_name, thread_id):
        thread_name = f"T-{thread_id:02d}"
        threading.current_thread().name = thread_name
        
        try:
            logger.debug(f"{thread_name} requesting {engine_name}")
            engine = manager.get_engine(engine_name)
            logger.debug(f"{thread_name} got {engine_name}")
            return True
        except Exception as e:
            logger.error(f"{thread_name} failed: {e}")
            return False
    
    # Launch threads
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(request_engine, 'whisper', i)
            for i in range(20)
        ]
        
        results = [f.result() for f in as_completed(futures)]
    
    # Check initialization count
    whisper_count = ThreadSafeTestEngine.get_count('TestWhisperEngine')
    logger.info(f"\nWhisper initialization count: {whisper_count}")
    
    if whisper_count == 1:
        logger.info("✓ PASS: Whisper initialized exactly once with 20 concurrent requests")
    else:
        logger.error(f"✗ FAIL: Race condition detected! Whisper initialized {whisper_count} times")
        return False
    
    # Verify lock was created
    if 'whisper' in manager._engine_locks:
        logger.info("✓ Lock created for 'whisper' during lazy initialization")
    else:
        logger.error("✗ No lock found for 'whisper'")
        return False
    
    # Test multiple different engines concurrently
    logger.info("\nTest: Concurrent initialization of multiple engines")
    logger.info("-" * 40)
    
    # Reset for next test
    manager.engines.clear()
    manager._engine_locks.clear()
    ThreadSafeTestEngine.reset_counts()
    
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = []
        # 10 threads each for 3 different engines
        for engine in ['whisper', 'vosk', 'silero']:
            for i in range(10):
                futures.append(
                    executor.submit(request_engine, engine, i + 100)
                )
        
        results = [f.result() for f in as_completed(futures)]
    
    # Check all initialization counts
    logger.info("\nInitialization counts:")
    all_good = True
    for engine_class, engine_name in [
        ('TestWhisperEngine', 'whisper'),
        ('TestVoskEngine', 'vosk'), 
        ('TestSileroEngine', 'silero')
    ]:
        count = ThreadSafeTestEngine.get_count(engine_class)
        if count == 1:
            logger.info(f"  {engine_name}: {count} ✓")
        else:
            logger.error(f"  {engine_name}: {count} ✗ (expected 1)")
            all_good = False
    
    if not all_good:
        logger.error("\n✗ FAIL: Race conditions detected")
        return False
    
    logger.info("\n✓ PASS: All engines initialized exactly once")
    
    # Verify locks were created for all engines
    for engine in ['whisper', 'vosk', 'silero']:
        if engine not in manager._engine_locks:
            logger.error(f"✗ No lock for {engine}")
            return False
    
    logger.info("✓ All engine locks properly created")
    
    logger.info("\n" + "=" * 60)
    logger.info("SUCCESS: Thread-safe lazy initialization working correctly!")
    logger.info("No race conditions detected.")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_lazy_initialization_thread_safety()
    sys.exit(0 if success else 1)