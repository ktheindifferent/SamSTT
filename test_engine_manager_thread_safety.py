#!/usr/bin/env python3
"""Test thread safety of the actual engine_manager implementation"""

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

# Mock all dependencies before importing
mock_modules = [
    'numpy', 'scipy', 'scipy.io', 'scipy.io.wavfile',
    'ffmpeg', 'sanic', 'torch', 'torchaudio', 'omegaconf',
    'transformers', 'librosa', 'speechbrain', 'nemo_toolkit',
    'pocketsphinx', 'vosk', 'STT', 'stt', 'whisper'
]

for module_name in mock_modules:
    sys.modules[module_name] = mock.MagicMock()

# Now import our module
from stts.engine_manager import STTEngineManager


class MockSTTEngine:
    """Mock STT Engine that simulates initialization delay"""
    
    initialization_count = {}
    initialization_lock = threading.Lock()
    
    def __init__(self, config):
        # Track initialization attempts
        engine_name = config.get('name', 'unknown')
        with self.initialization_lock:
            if engine_name not in self.initialization_count:
                self.initialization_count[engine_name] = 0
            self.initialization_count[engine_name] += 1
            count = self.initialization_count[engine_name]
        
        # Simulate initialization delay
        time.sleep(0.05)
        
        self.config = config
        self.name = engine_name
        self.is_available = True
        
        logger.info(f"MockSTTEngine '{engine_name}' initialized (attempt #{count}) by {threading.current_thread().name}")
        
        if count > 1:
            logger.warning(f"WARNING: Duplicate initialization of {engine_name}!")


def test_thread_safe_initialization():
    """Test that the engine manager properly handles concurrent initialization requests"""
    
    logger.info("=" * 60)
    logger.info("Testing Thread-Safe Engine Manager")
    logger.info("=" * 60)
    
    # Replace all engine classes with our mock
    for engine_name in STTEngineManager.ENGINES:
        STTEngineManager.ENGINES[engine_name] = MockSTTEngine
    
    # Reset initialization counter
    MockSTTEngine.initialization_count = {}
    
    # Create manager with no pre-initialization
    config = {
        'initialize_all': False,
        'whisper': {'name': 'whisper'},
        'vosk': {'name': 'vosk'},
        'silero': {'name': 'silero'}
    }
    
    manager = STTEngineManager(default_engine='whisper', config=config)
    
    # Verify thread safety attributes exist
    assert hasattr(manager, '_engine_locks'), "Missing _engine_locks attribute"
    assert hasattr(manager, '_locks_lock'), "Missing _locks_lock attribute"
    assert isinstance(manager._locks_lock, type(threading.Lock())), "_locks_lock should be a Lock"
    
    logger.info("✓ Thread safety attributes properly initialized")
    
    # Test concurrent access to the same engine
    logger.info("\nTest 1: Concurrent access to same engine")
    logger.info("-" * 40)
    
    results = []
    errors = []
    
    def get_engine_worker(engine_name, worker_id):
        thread_name = f"Worker-{worker_id:02d}"
        threading.current_thread().name = thread_name
        
        try:
            logger.debug(f"{thread_name} requesting {engine_name}")
            engine = manager.get_engine(engine_name)
            logger.debug(f"{thread_name} got {engine_name}")
            return f"Success: {thread_name}"
        except Exception as e:
            logger.error(f"{thread_name} failed: {e}")
            return f"Error: {thread_name} - {e}"
    
    # Launch 20 threads all trying to get 'whisper' engine
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(get_engine_worker, 'whisper', i)
            for i in range(20)
        ]
        
        for future in as_completed(futures):
            result = future.result()
            if result.startswith("Success"):
                results.append(result)
            else:
                errors.append(result)
    
    # Check initialization count
    whisper_count = MockSTTEngine.initialization_count.get('whisper', 0)
    logger.info(f"\nWhisper initialization count: {whisper_count}")
    
    if whisper_count == 1:
        logger.info("✓ PASS: Whisper initialized exactly once")
    else:
        logger.error(f"✗ FAIL: Whisper initialized {whisper_count} times (expected 1)")
        return False
    
    # Test 2: Multiple engines with concurrent requests
    logger.info("\nTest 2: Multiple engines with concurrent requests")
    logger.info("-" * 40)
    
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = []
        # 10 threads each for whisper (already initialized), vosk, and silero
        for engine in ['whisper', 'vosk', 'silero']:
            for i in range(10):
                futures.append(
                    executor.submit(get_engine_worker, engine, i + 100)
                )
        
        for future in as_completed(futures):
            result = future.result()
    
    # Check initialization counts
    logger.info("\nFinal initialization counts:")
    all_correct = True
    for engine in ['whisper', 'vosk', 'silero']:
        count = MockSTTEngine.initialization_count.get(engine, 0)
        expected = 1
        status = "✓" if count == expected else "✗"
        logger.info(f"  {engine}: {count} {status}")
        if count != expected:
            all_correct = False
    
    if not all_correct:
        logger.error("\n✗ FAIL: Some engines were initialized multiple times")
        return False
    
    logger.info("\n✓ PASS: All engines initialized exactly once")
    
    # Test 3: Check that locks are created per engine
    logger.info("\nTest 3: Verify per-engine locks")
    logger.info("-" * 40)
    
    # After initialization, we should have locks for each accessed engine
    if 'whisper' in manager._engine_locks:
        logger.info("✓ Lock created for 'whisper'")
    else:
        logger.error("✗ No lock for 'whisper'")
        return False
    
    if 'vosk' in manager._engine_locks:
        logger.info("✓ Lock created for 'vosk'")
    else:
        logger.error("✗ No lock for 'vosk'")
        return False
    
    if 'silero' in manager._engine_locks:
        logger.info("✓ Lock created for 'silero'")
    else:
        logger.error("✗ No lock for 'silero'")
        return False
    
    # Verify all locks are actual Lock objects
    for engine_name, lock in manager._engine_locks.items():
        if not isinstance(lock, type(threading.Lock())):
            logger.error(f"✗ {engine_name} lock is not a threading.Lock")
            return False
    
    logger.info("✓ All engine locks are proper threading.Lock objects")
    
    logger.info("\n" + "=" * 60)
    logger.info("SUCCESS: All thread safety tests PASSED!")
    logger.info("The engine manager is properly thread-safe.")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_thread_safe_initialization()
    sys.exit(0 if success else 1)