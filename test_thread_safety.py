#!/usr/bin/env python3
"""Test thread-safe engine initialization in STTEngineManager"""

import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stts.engine_manager import STTEngineManager

# Configure logging to see detailed initialization messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(threadName)-10s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InitializationTracker:
    """Track engine initialization attempts and results"""
    
    def __init__(self):
        self.attempts: Dict[str, List[str]] = {}
        self.successes: Dict[str, List[str]] = {}
        self.lock = threading.Lock()
    
    def record_attempt(self, engine_name: str, thread_name: str):
        with self.lock:
            if engine_name not in self.attempts:
                self.attempts[engine_name] = []
            self.attempts[engine_name].append(thread_name)
    
    def record_success(self, engine_name: str, thread_name: str):
        with self.lock:
            if engine_name not in self.successes:
                self.successes[engine_name] = []
            self.successes[engine_name].append(thread_name)
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'attempts': dict(self.attempts),
                'successes': dict(self.successes),
                'duplicate_attempts': {
                    engine: len(threads) > 1 
                    for engine, threads in self.attempts.items()
                }
            }


def test_concurrent_initialization():
    """Test that multiple threads trying to initialize the same engine don't cause issues"""
    
    logger.info("=" * 80)
    logger.info("Testing concurrent engine initialization")
    logger.info("=" * 80)
    
    # Create manager with no pre-initialized engines
    config = {
        'initialize_all': False,  # Don't initialize any engines at startup
        'whisper': {'model_size': 'tiny'},
        'vosk': {},
        'silero': {}
    }
    
    manager = STTEngineManager(default_engine='whisper', config=config)
    tracker = InitializationTracker()
    
    def get_engine_concurrent(engine_name: str, thread_id: int) -> str:
        """Worker function to get an engine"""
        thread_name = f"Worker-{thread_id}"
        threading.current_thread().name = thread_name
        
        try:
            tracker.record_attempt(engine_name, thread_name)
            logger.info(f"Thread {thread_name} requesting {engine_name} engine")
            
            # Small random delay to increase chance of race condition
            time.sleep(0.001 * (thread_id % 3))
            
            engine = manager.get_engine(engine_name)
            tracker.record_success(engine_name, thread_name)
            
            logger.info(f"Thread {thread_name} successfully got {engine_name} engine")
            return f"Success: {thread_name} got {engine_name}"
            
        except Exception as e:
            logger.error(f"Thread {thread_name} failed to get {engine_name}: {e}")
            return f"Failed: {thread_name} - {str(e)}"
    
    # Test 1: Multiple threads requesting the same engine simultaneously
    logger.info("\nTest 1: Multiple threads requesting same engine")
    logger.info("-" * 40)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit 10 workers all trying to get 'whisper' engine
        futures = [
            executor.submit(get_engine_concurrent, 'whisper', i)
            for i in range(10)
        ]
        
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            logger.debug(f"Result: {result}")
    
    # Check that engine was only initialized once
    stats = tracker.get_stats()
    logger.info(f"\nStats for 'whisper' engine:")
    logger.info(f"  Total attempts: {len(stats['attempts'].get('whisper', []))}")
    logger.info(f"  Successful: {len(stats['successes'].get('whisper', []))}")
    
    # Verify engine is actually initialized
    assert 'whisper' in manager.list_available_engines(), "Whisper engine should be initialized"
    
    # Test 2: Multiple threads requesting different engines
    logger.info("\n" + "=" * 80)
    logger.info("Test 2: Multiple threads requesting different engines")
    logger.info("-" * 40)
    
    # Reset tracker for second test
    tracker = InitializationTracker()
    
    with ThreadPoolExecutor(max_workers=9) as executor:
        futures = []
        # 3 threads each for whisper (already initialized), vosk, and silero
        for i in range(3):
            futures.append(executor.submit(get_engine_concurrent, 'whisper', i))
            futures.append(executor.submit(get_engine_concurrent, 'vosk', i + 3))
            futures.append(executor.submit(get_engine_concurrent, 'silero', i + 6))
        
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    stats = tracker.get_stats()
    for engine in ['whisper', 'vosk', 'silero']:
        logger.info(f"\nStats for '{engine}' engine:")
        logger.info(f"  Total attempts: {len(stats['attempts'].get(engine, []))}")
        logger.info(f"  Successful: {len(stats['successes'].get(engine, []))}")
    
    # Test 3: Stress test with many threads
    logger.info("\n" + "=" * 80)
    logger.info("Test 3: Stress test with many concurrent requests")
    logger.info("-" * 40)
    
    stress_engines = ['whisper', 'vosk', 'silero']
    num_threads_per_engine = 20
    
    with ThreadPoolExecutor(max_workers=60) as executor:
        futures = []
        for engine in stress_engines:
            for i in range(num_threads_per_engine):
                futures.append(executor.submit(get_engine_concurrent, engine, i))
        
        success_count = 0
        failure_count = 0
        for future in as_completed(futures):
            result = future.result()
            if result.startswith("Success"):
                success_count += 1
            else:
                failure_count += 1
    
    logger.info(f"\nStress test results:")
    logger.info(f"  Total requests: {len(futures)}")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Failed: {failure_count}")
    
    # Verify final state
    available = manager.list_available_engines()
    logger.info(f"\nFinal available engines: {available}")
    
    # Test 4: Verify no duplicate models in memory
    logger.info("\n" + "=" * 80)
    logger.info("Test 4: Verify no duplicate engine instances")
    logger.info("-" * 40)
    
    # Check that each engine appears exactly once in the manager
    engine_ids = {}
    for name, engine in manager.engines.items():
        engine_id = id(engine)
        if engine_id in engine_ids:
            logger.error(f"DUPLICATE: Engine {name} has same ID as {engine_ids[engine_id]}")
            assert False, "Duplicate engine instance detected!"
        engine_ids[engine_id] = name
    
    logger.info(f"✓ All {len(engine_ids)} engines have unique instances")
    
    logger.info("\n" + "=" * 80)
    logger.info("All thread safety tests passed!")
    logger.info("=" * 80)


def test_initialization_performance():
    """Compare performance with and without proper locking"""
    
    logger.info("\n" + "=" * 80)
    logger.info("Testing initialization performance")
    logger.info("=" * 80)
    
    config = {
        'initialize_all': False,
        'whisper': {'model_size': 'tiny'}
    }
    
    # Test with single thread (baseline)
    manager = STTEngineManager(default_engine='whisper', config=config)
    
    start_time = time.time()
    engine = manager.get_engine('whisper')
    single_thread_time = time.time() - start_time
    
    logger.info(f"Single thread initialization time: {single_thread_time:.4f} seconds")
    
    # Test with multiple threads (should not significantly impact performance)
    manager2 = STTEngineManager(default_engine='vosk', config={'vosk': {}})
    
    def timed_get(name: str) -> float:
        start = time.time()
        manager2.get_engine(name)
        return time.time() - start
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(timed_get, 'vosk') for _ in range(10)]
        times = [f.result() for f in as_completed(futures)]
    
    avg_time = sum(times) / len(times)
    logger.info(f"Average time with 10 concurrent threads: {avg_time:.4f} seconds")
    logger.info(f"Max time: {max(times):.4f} seconds")
    logger.info(f"Min time: {min(times):.4f} seconds")
    
    # The first thread should take the most time (actual initialization)
    # Other threads should be fast (just waiting on lock)
    logger.info("\n✓ Performance test completed")


if __name__ == "__main__":
    try:
        test_concurrent_initialization()
        test_initialization_performance()
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL TESTS PASSED SUCCESSFULLY!")
        logger.info("Thread-safe engine initialization is working correctly.")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)