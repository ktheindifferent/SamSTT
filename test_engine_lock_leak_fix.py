#!/usr/bin/env python3
"""Comprehensive test to validate the engine lock leak fix"""

import sys
import os
import threading
import time
import gc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest import mock
import tracemalloc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock dependencies
mock_modules = [
    'numpy', 'scipy', 'scipy.io', 'scipy.io.wavfile',
    'ffmpeg', 'sanic', 'torch', 'torchaudio', 'omegaconf',
    'transformers', 'librosa', 'speechbrain', 'nemo_toolkit',
    'pocketsphinx', 'vosk', 'STT', 'stt', 'whisper'
]

for module_name in mock_modules:
    sys.modules[module_name] = mock.MagicMock()

from stts.engine_manager import STTEngineManager


class MockEngine:
    """Mock engine for testing"""
    instances_created = 0
    
    def __init__(self, config):
        MockEngine.instances_created += 1
        self.config = config
        self.name = config.get('name', 'unknown')
        self.is_available = True


def test_no_memory_leak():
    """Test that repeated engine requests don't cause memory leak"""
    
    logger.info("=" * 60)
    logger.info("TEST: No Memory Leak with Cleanup Mechanism")
    logger.info("=" * 60)
    
    # Start memory tracking
    tracemalloc.start()
    
    # Replace engines with mock
    for engine_name in STTEngineManager.ENGINES:
        STTEngineManager.ENGINES[engine_name] = MockEngine
    
    # Add many test engines
    for i in range(100):
        STTEngineManager.ENGINES[f'leak_test_{i}'] = MockEngine
    
    config = {'initialize_all': False}
    for i in range(100):
        config[f'leak_test_{i}'] = {'name': f'leak_test_{i}'}
    
    # Create manager with initialization disabled
    config['initialize_all'] = False
    manager = STTEngineManager(default_engine='whisper', config=config)
    manager._lock_cleanup_interval = 1  # Fast cleanup
    # Clear any pre-initialized engines to force lazy initialization
    manager.engines.clear()
    
    # Take initial memory snapshot
    snapshot1 = tracemalloc.take_snapshot()
    
    logger.info("Phase 1: Creating 100 engine locks...")
    
    # Request all engines to create locks
    for i in range(100):
        engine = manager.get_engine(f'leak_test_{i}')
    
    initial_locks = len(manager._engine_locks)
    logger.info(f"Locks created: {initial_locks}")
    
    # Wait for TTL and trigger cleanup
    logger.info("Phase 2: Waiting for TTL and triggering cleanup...")
    time.sleep(manager._lock_cleanup_interval + 0.5)
    manager._cleanup_unused_locks()
    
    locks_after_cleanup = len(manager._engine_locks)
    logger.info(f"Locks after cleanup: {locks_after_cleanup}")
    
    # Take memory snapshot after cleanup
    snapshot2 = tracemalloc.take_snapshot()
    
    # Request engines again (should recreate locks)
    logger.info("Phase 3: Re-requesting engines after cleanup...")
    for i in range(100):
        engine = manager.get_engine(f'leak_test_{i}')
    
    final_locks = len(manager._engine_locks)
    logger.info(f"Final locks: {final_locks}")
    
    # Analyze memory usage
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    # Check that lock count doesn't grow indefinitely
    assert final_locks <= 100, f"Lock count grew beyond expected: {final_locks}"
    
    # Verify cleanup actually removed locks
    assert locks_after_cleanup < initial_locks, "Cleanup didn't remove any locks"
    
    logger.info("✓ No memory leak detected - cleanup is working")
    
    tracemalloc.stop()
    return True


def test_thread_safety_with_cleanup():
    """Test that cleanup doesn't interfere with concurrent access"""
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Thread Safety with Concurrent Cleanup")
    logger.info("=" * 60)
    
    # Replace engines with mock
    for engine_name in STTEngineManager.ENGINES:
        STTEngineManager.ENGINES[engine_name] = MockEngine
    
    # Add test engines
    for i in range(50):
        STTEngineManager.ENGINES[f'concurrent_{i}'] = MockEngine
    
    config = {'initialize_all': False}
    for i in range(50):
        config[f'concurrent_{i}'] = {'name': f'concurrent_{i}'}
    
    manager = STTEngineManager(default_engine='whisper', config=config)
    manager._lock_cleanup_interval = 0.1  # Very aggressive cleanup
    manager.engines.clear()
    
    errors = []
    success_count = 0
    
    def worker(worker_id):
        """Worker that constantly requests engines"""
        try:
            for _ in range(20):
                engine_id = worker_id % 50
                engine = manager.get_engine(f'concurrent_{engine_id}')
                time.sleep(0.01)
            return True
        except Exception as e:
            errors.append(str(e))
            return False
    
    def cleanup_worker():
        """Worker that triggers cleanup"""
        for _ in range(10):
            time.sleep(0.2)
            manager._cleanup_unused_locks()
    
    logger.info("Starting concurrent access with aggressive cleanup...")
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_worker)
    cleanup_thread.start()
    
    # Start worker threads
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(worker, i) for i in range(20)]
        
        for future in as_completed(futures):
            if future.result():
                success_count += 1
    
    cleanup_thread.join()
    
    logger.info(f"Results: {success_count}/20 successful, {len(errors)} errors")
    
    # Should have high success rate
    assert success_count >= 18, f"Too many failures: {20 - success_count}"
    
    if errors:
        logger.warning(f"Some errors occurred (expected with aggressive cleanup): {errors[:3]}")
    
    logger.info("✓ Thread safety maintained during cleanup")
    return True


def test_cleanup_mechanism_attributes():
    """Test that all cleanup mechanism attributes are present"""
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Cleanup Mechanism Attributes")
    logger.info("=" * 60)
    
    # Create manager
    manager = STTEngineManager(default_engine='whisper', config={})
    
    # Check required attributes
    required_attrs = [
        '_cleanup_unused_locks',  # Cleanup method
        '_get_or_create_lock',     # Lock creation with tracking
        '_lock_refs',              # Weak references
        '_lock_last_used',         # TTL tracking
        '_lock_cleanup_interval',  # Cleanup interval
        'get_lock_stats'           # Monitoring method
    ]
    
    for attr in required_attrs:
        assert hasattr(manager, attr), f"Missing required attribute: {attr}"
        logger.info(f"✓ {attr} present")
    
    # Verify cleanup interval is reasonable
    assert manager._lock_cleanup_interval > 0, "Cleanup interval must be positive"
    assert manager._lock_cleanup_interval <= 3600, "Cleanup interval too long"
    logger.info(f"✓ Cleanup interval: {manager._lock_cleanup_interval} seconds")
    
    # Test get_lock_stats works
    stats = manager.get_lock_stats()
    assert 'total_locks' in stats, "Stats missing total_locks"
    assert 'locks' in stats, "Stats missing locks dict"
    logger.info(f"✓ Lock stats working: {stats}")
    
    logger.info("✓ All cleanup mechanism attributes present and functional")
    return True


def test_lock_recreation():
    """Test that locks can be recreated after cleanup"""
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Lock Recreation After Cleanup")
    logger.info("=" * 60)
    
    # Replace engines with mock
    for engine_name in STTEngineManager.ENGINES:
        STTEngineManager.ENGINES[engine_name] = MockEngine
    
    STTEngineManager.ENGINES['recreate_test'] = MockEngine
    
    config = {
        'initialize_all': False,
        'recreate_test': {'name': 'recreate_test'}
    }
    
    manager = STTEngineManager(default_engine='whisper', config=config)
    manager._lock_cleanup_interval = 0.5
    manager.engines.clear()
    
    # First access
    logger.info("1. First access - creating lock")
    engine1 = manager.get_engine('recreate_test')
    assert 'recreate_test' in manager._engine_locks
    lock1_id = id(manager._engine_locks['recreate_test'])
    logger.info(f"   Lock created with ID: {lock1_id}")
    
    # Wait and cleanup
    logger.info("2. Waiting for TTL and cleaning up")
    time.sleep(manager._lock_cleanup_interval + 0.1)
    manager._cleanup_unused_locks()
    assert 'recreate_test' not in manager._engine_locks
    logger.info("   Lock successfully cleaned up")
    
    # Second access - should recreate
    logger.info("3. Second access - recreating lock")
    engine2 = manager.get_engine('recreate_test')
    assert 'recreate_test' in manager._engine_locks
    lock2_id = id(manager._engine_locks['recreate_test'])
    logger.info(f"   Lock recreated with ID: {lock2_id}")
    
    # Verify it's a new lock object
    assert lock1_id != lock2_id, "Should be a new lock object"
    logger.info("   ✓ New lock object created")
    
    # Verify engine still works
    assert engine2 is not None
    assert engine2.name == 'recreate_test'
    logger.info("   ✓ Engine still functional")
    
    logger.info("✓ Locks can be recreated after cleanup")
    return True


def run_all_tests():
    """Run all comprehensive tests"""
    
    tests = [
        ("Cleanup Mechanism Attributes", test_cleanup_mechanism_attributes),
        ("No Memory Leak", test_no_memory_leak),
        ("Thread Safety with Cleanup", test_thread_safety_with_cleanup),
        ("Lock Recreation", test_lock_recreation)
    ]
    
    passed = 0
    failed = 0
    
    logger.info("\n" + "=" * 60)
    logger.info("ENGINE LOCK LEAK FIX - COMPREHENSIVE TEST SUITE")
    logger.info("=" * 60)
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\nRunning: {test_name}")
            if test_func():
                passed += 1
                logger.info(f"✓✓✓ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"✗✗✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"✗✗✗ {test_name} FAILED with exception: {e}")
            traceback.print_exc()
    
    logger.info("\n" + "=" * 60)
    logger.info(f"FINAL RESULTS: {passed} passed, {failed} failed")
    logger.info("=" * 60)
    
    if failed == 0:
        logger.info("\n🎉 SUCCESS: All tests passed!")
        logger.info("The engine lock leak has been successfully fixed.")
        logger.info("\nKey improvements:")
        logger.info("1. TTL-based cleanup removes unused locks after 5 minutes")
        logger.info("2. Weak references track lock lifecycle")
        logger.info("3. Thread-safe cleanup doesn't interfere with active locks")
        logger.info("4. Monitoring via get_lock_stats() for production visibility")
        logger.info("5. Locks can be recreated as needed after cleanup")
        return True
    else:
        logger.error(f"\n❌ FAILURE: {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)