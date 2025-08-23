#!/usr/bin/env python3
"""Simplified test for thread-safe engine initialization"""

import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(threadName)-10s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockEngine:
    """Mock STT Engine for testing purposes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', 'mock')
        self.is_available = True
        # Simulate initialization delay
        time.sleep(0.1)
        logger.info(f"MockEngine {self.name} initialized by {threading.current_thread().name}")


class SimplifiedEngineManager:
    """Simplified version of STTEngineManager for testing thread safety"""
    
    ENGINES = {
        'engine1': MockEngine,
        'engine2': MockEngine,
        'engine3': MockEngine,
    }
    
    def __init__(self):
        self.engines: Dict[str, MockEngine] = {}
        self._engine_locks: Dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()
        self.initialization_count = {}  # Track how many times each engine is initialized
        self._count_lock = threading.Lock()
    
    def get_engine(self, name: str) -> MockEngine:
        """Get an engine with thread-safe initialization"""
        
        # Double-checked locking pattern
        if name not in self.engines:
            # Get or create lock for this specific engine
            with self._locks_lock:
                if name not in self._engine_locks:
                    self._engine_locks[name] = threading.Lock()
                engine_lock = self._engine_locks[name]
            
            # Try to initialize with proper locking
            with engine_lock:
                # Double-check after acquiring lock
                if name not in self.engines:
                    logger.info(f"Thread {threading.current_thread().name}: Initializing {name}")
                    
                    # Track initialization count
                    with self._count_lock:
                        if name not in self.initialization_count:
                            self.initialization_count[name] = 0
                        self.initialization_count[name] += 1
                    
                    # Create the engine
                    if name in self.ENGINES:
                        engine = self.ENGINES[name]({'name': name})
                        self.engines[name] = engine
                        logger.info(f"Thread {threading.current_thread().name}: Successfully initialized {name}")
                    else:
                        raise ValueError(f"Unknown engine: {name}")
                else:
                    logger.debug(f"Thread {threading.current_thread().name}: {name} already initialized")
        
        return self.engines[name]


def test_thread_safety():
    """Test that multiple threads don't cause duplicate initialization"""
    
    logger.info("=" * 60)
    logger.info("Testing Thread-Safe Engine Initialization")
    logger.info("=" * 60)
    
    manager = SimplifiedEngineManager()
    results = {'success': 0, 'errors': 0}
    results_lock = threading.Lock()
    
    def get_engine_worker(engine_name: str, worker_id: int):
        """Worker function to get an engine"""
        thread_name = f"Worker-{worker_id:02d}"
        threading.current_thread().name = thread_name
        
        try:
            logger.debug(f"{thread_name} requesting {engine_name}")
            engine = manager.get_engine(engine_name)
            logger.debug(f"{thread_name} got {engine_name}")
            
            with results_lock:
                results['success'] += 1
            return True
            
        except Exception as e:
            logger.error(f"{thread_name} failed: {e}")
            with results_lock:
                results['errors'] += 1
            return False
    
    # Test 1: Many threads requesting the same engine
    logger.info("\nTest 1: 20 threads requesting the same engine")
    logger.info("-" * 40)
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(get_engine_worker, 'engine1', i)
            for i in range(20)
        ]
        
        for future in as_completed(futures):
            future.result()
    
    # Check that engine was initialized exactly once
    init_count = manager.initialization_count.get('engine1', 0)
    logger.info(f"\nEngine1 initialization count: {init_count}")
    
    if init_count == 1:
        logger.info("✓ PASS: Engine initialized exactly once despite 20 concurrent requests")
    else:
        logger.error(f"✗ FAIL: Engine initialized {init_count} times (expected 1)")
        return False
    
    # Test 2: Multiple engines with concurrent requests
    logger.info("\nTest 2: Multiple engines with concurrent requests")
    logger.info("-" * 40)
    
    manager2 = SimplifiedEngineManager()
    
    def get_engine_worker2(engine_name: str, worker_id: int):
        """Worker function for manager2"""
        thread_name = f"Worker-{worker_id:02d}"
        threading.current_thread().name = thread_name
        
        try:
            logger.debug(f"{thread_name} requesting {engine_name}")
            engine = manager2.get_engine(engine_name)  # Use manager2 here
            logger.debug(f"{thread_name} got {engine_name}")
            
            with results_lock:
                results['success'] += 1
            return True
            
        except Exception as e:
            logger.error(f"{thread_name} failed: {e}")
            with results_lock:
                results['errors'] += 1
            return False
    
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = []
        # 10 threads for each of 3 engines
        for engine_num in range(1, 4):
            for worker_id in range(10):
                futures.append(
                    executor.submit(get_engine_worker2, f'engine{engine_num}', worker_id + engine_num*10)
                )
        
        for future in as_completed(futures):
            future.result()
    
    # Check initialization counts
    logger.info("\nInitialization counts:")
    all_correct = True
    for engine_name in ['engine1', 'engine2', 'engine3']:
        count = manager2.initialization_count.get(engine_name, 0)
        logger.info(f"  {engine_name}: {count}")
        if count != 1:
            all_correct = False
            logger.error(f"    ✗ Expected 1, got {count}")
        else:
            logger.info(f"    ✓ Correct")
    
    if all_correct:
        logger.info("\n✓ PASS: All engines initialized exactly once")
    else:
        logger.error("\n✗ FAIL: Some engines initialized multiple times")
        return False
    
    # Test 3: Verify engine instances are unique
    logger.info("\nTest 3: Verify unique engine instances")
    logger.info("-" * 40)
    
    engine_ids = set()
    for name, engine in manager2.engines.items():
        engine_id = id(engine)
        if engine_id in engine_ids:
            logger.error(f"✗ FAIL: Duplicate engine instance found for {name}")
            return False
        engine_ids.add(engine_id)
    
    logger.info(f"✓ PASS: All {len(engine_ids)} engine instances are unique")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: All thread safety tests PASSED!")
    logger.info("The double-checked locking pattern is working correctly.")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_thread_safety()
    sys.exit(0 if success else 1)