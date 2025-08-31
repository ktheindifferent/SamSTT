#!/usr/bin/env python3
"""
Test suite for verifying race condition fixes in STT Engine Manager initialization.

This test suite implements comprehensive stress tests to ensure thread-safe
engine initialization without race conditions or duplicate instances.
"""

import threading
import time
import unittest
import tempfile
import json
import os
import sys
import gc
import tracemalloc
from typing import Dict, List, Set, Any
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stts.engine_manager import STTEngineManager
from stts.base_engine import BaseSTTEngine


class MockEngine(BaseSTTEngine):
    """Mock STT engine for testing thread-safe initialization"""
    
    # Class-level tracking of all instances created
    instances_created = 0
    instances_lock = threading.Lock()
    instance_ids: Set[int] = set()
    initialization_times: List[float] = []
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize mock engine with tracking"""
        super().__init__(config)
        
        # Track instance creation atomically
        with MockEngine.instances_lock:
            MockEngine.instances_created += 1
            instance_id = id(self)
            MockEngine.instance_ids.add(instance_id)
            MockEngine.initialization_times.append(time.time())
        
        # Simulate initialization work
        time.sleep(0.01)  # Simulate some initialization work
        
        self.name = "mock_engine"
        self.is_available = True
        self.instance_id = instance_id
        
    @classmethod
    def reset_tracking(cls):
        """Reset class-level tracking variables"""
        with cls.instances_lock:
            cls.instances_created = 0
            cls.instance_ids.clear()
            cls.initialization_times.clear()
    
    def initialize(self):
        """Initialize the mock engine"""
        pass
    
    def transcribe_raw(self, audio: bytes) -> str:
        """Mock transcription"""
        return "mock transcription"


class TestEngineInitRaceCondition(unittest.TestCase):
    """Test suite for engine initialization race conditions"""
    
    def setUp(self):
        """Set up test environment"""
        MockEngine.reset_tracking()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        MockEngine.reset_tracking()
        gc.collect()  # Force garbage collection to clean up any lingering references
    
    def test_concurrent_initialization_no_duplicates(self):
        """Test that concurrent threads don't create duplicate engine instances"""
        print("\n=== Test: Concurrent Initialization - No Duplicates ===")
        
        # Patch the ENGINES registry to use our mock
        with patch.object(STTEngineManager, 'ENGINES', {'mock': MockEngine}):
            manager = STTEngineManager(default_engine='mock')
            
            # Number of concurrent threads
            num_threads = 100
            results = []
            barrier = threading.Barrier(num_threads)
            
            def get_engine(thread_id):
                """Worker function to get engine with barrier synchronization"""
                # Wait for all threads to be ready (maximize race condition probability)
                barrier.wait()
                
                try:
                    engine = manager.get_engine('mock')
                    return (thread_id, id(engine), engine.instance_id)
                except Exception as e:
                    return (thread_id, None, str(e))
            
            # Launch threads
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(get_engine, i) for i in range(num_threads)]
                
                for future in as_completed(futures):
                    results.append(future.result())
            
            # Verify results
            engine_ids = set()
            instance_ids = set()
            errors = []
            
            for thread_id, engine_id, instance_id in results:
                if engine_id is None:
                    errors.append((thread_id, instance_id))
                else:
                    engine_ids.add(engine_id)
                    instance_ids.add(instance_id)
            
            print(f"Threads executed: {num_threads}")
            print(f"Unique engine IDs returned: {len(engine_ids)}")
            print(f"Unique instance IDs: {len(instance_ids)}")
            print(f"MockEngine instances created: {MockEngine.instances_created}")
            print(f"Errors: {len(errors)}")
            
            # Assertions
            self.assertEqual(len(errors), 0, f"Some threads failed: {errors}")
            self.assertEqual(len(engine_ids), 1, "Multiple engine objects returned (should be singleton)")
            self.assertEqual(len(instance_ids), 1, "Multiple instance IDs found")
            self.assertEqual(MockEngine.instances_created, 1, "Multiple MockEngine instances created")
    
    def test_multiple_engines_concurrent_initialization(self):
        """Test concurrent initialization of multiple different engines"""
        print("\n=== Test: Multiple Engines Concurrent Initialization ===")
        
        # Create multiple mock engine classes
        class MockEngineA(MockEngine):
            def __init__(self, config=None):
                super().__init__(config)
                self.name = "mock_a"
        
        class MockEngineB(MockEngine):
            def __init__(self, config=None):
                super().__init__(config)
                self.name = "mock_b"
        
        class MockEngineC(MockEngine):
            def __init__(self, config=None):
                super().__init__(config)
                self.name = "mock_c"
        
        # Reset tracking for all
        for cls in [MockEngineA, MockEngineB, MockEngineC]:
            cls.reset_tracking()
        
        engines = {
            'mock_a': MockEngineA,
            'mock_b': MockEngineB,
            'mock_c': MockEngineC
        }
        
        with patch.object(STTEngineManager, 'ENGINES', engines):
            manager = STTEngineManager(default_engine='mock_a')
            
            num_threads_per_engine = 50
            results = []
            barrier = threading.Barrier(num_threads_per_engine * 3)
            
            def get_engine(thread_id, engine_name):
                """Worker function to get specific engine"""
                barrier.wait()
                
                try:
                    engine = manager.get_engine(engine_name)
                    return (thread_id, engine_name, id(engine))
                except Exception as e:
                    return (thread_id, engine_name, str(e))
            
            # Launch threads for all engines
            with ThreadPoolExecutor(max_workers=num_threads_per_engine * 3) as executor:
                futures = []
                for engine_name in engines.keys():
                    for i in range(num_threads_per_engine):
                        futures.append(executor.submit(get_engine, i, engine_name))
                
                for future in as_completed(futures):
                    results.append(future.result())
            
            # Verify results per engine
            engine_instances = {'mock_a': set(), 'mock_b': set(), 'mock_c': set()}
            
            for thread_id, engine_name, engine_id in results:
                if isinstance(engine_id, int):
                    engine_instances[engine_name].add(engine_id)
            
            print(f"Total threads: {num_threads_per_engine * 3}")
            for engine_name, instances in engine_instances.items():
                print(f"Engine {engine_name}: {len(instances)} unique instances")
                self.assertEqual(len(instances), 1, f"Multiple instances for {engine_name}")
    
    def test_stress_test_with_barriers(self):
        """Stress test using threading barriers to maximize race conditions"""
        print("\n=== Test: Stress Test with Barriers ===")
        
        with patch.object(STTEngineManager, 'ENGINES', {'mock': MockEngine}):
            manager = STTEngineManager(default_engine='mock')
            
            num_rounds = 10
            threads_per_round = 20
            
            for round_num in range(num_rounds):
                MockEngine.reset_tracking()
                
                # Clear existing engines to force re-initialization
                manager.engines.clear()
                
                barrier = threading.Barrier(threads_per_round)
                results = []
                
                def stress_worker(worker_id):
                    """Worker that uses barrier to synchronize"""
                    barrier.wait()  # All threads start exactly together
                    
                    engine = manager.get_engine('mock')
                    return id(engine)
                
                with ThreadPoolExecutor(max_workers=threads_per_round) as executor:
                    futures = [executor.submit(stress_worker, i) for i in range(threads_per_round)]
                    
                    for future in as_completed(futures):
                        results.append(future.result())
                
                unique_engines = set(results)
                
                print(f"Round {round_num + 1}: {len(unique_engines)} unique engines, "
                      f"{MockEngine.instances_created} instances created")
                
                self.assertEqual(len(unique_engines), 1, f"Round {round_num}: Multiple engines created")
                self.assertEqual(MockEngine.instances_created, 1, f"Round {round_num}: Multiple instances")
    
    def test_memory_leak_prevention(self):
        """Test that repeated engine requests don't cause memory leaks"""
        print("\n=== Test: Memory Leak Prevention ===")
        
        tracemalloc.start()
        
        with patch.object(STTEngineManager, 'ENGINES', {'mock': MockEngine}):
            manager = STTEngineManager(default_engine='mock')
            
            # Take initial snapshot
            snapshot1 = tracemalloc.take_snapshot()
            
            # Perform many engine requests
            num_requests = 1000
            
            def make_requests():
                for _ in range(num_requests):
                    engine = manager.get_engine('mock')
                    # Use the engine to ensure it's not optimized away
                    _ = engine.name
            
            # Run requests in multiple threads
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_requests) for _ in range(10)]
                
                for future in as_completed(futures):
                    future.result()
            
            # Take final snapshot
            snapshot2 = tracemalloc.take_snapshot()
            
            # Check that we still have only one engine instance
            self.assertEqual(len(manager.engines), 1)
            self.assertEqual(MockEngine.instances_created, 1)
            
            # Analyze memory difference
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            
            # Find any significant memory growth (> 1MB)
            significant_growth = []
            for stat in top_stats:
                if stat.size_diff > 1024 * 1024:  # 1MB
                    significant_growth.append(stat)
            
            print(f"Total requests: {num_requests * 10}")
            print(f"Engine instances: {MockEngine.instances_created}")
            print(f"Significant memory growth items: {len(significant_growth)}")
            
            # Should not have significant memory growth
            self.assertEqual(len(significant_growth), 0, 
                           f"Memory leak detected: {significant_growth[:3]}")
        
        tracemalloc.stop()
    
    def test_lock_cleanup_mechanism(self):
        """Test that the lock cleanup mechanism works correctly"""
        print("\n=== Test: Lock Cleanup Mechanism ===")
        
        with patch.object(STTEngineManager, 'ENGINES', {'mock': MockEngine}):
            manager = STTEngineManager(default_engine='mock')
            
            # Set a short cleanup interval for testing
            manager._lock_cleanup_interval = 0.1  # 100ms for testing
            
            # Create locks for multiple engines
            engine_names = [f'mock_{i}' for i in range(15)]
            
            # Patch ENGINES to have all these mock engines
            mock_engines = {name: MockEngine for name in engine_names}
            with patch.object(STTEngineManager, 'ENGINES', mock_engines):
                # Request each engine to create locks
                for name in engine_names[:10]:
                    lock = manager._get_or_create_lock(name)
                    self.assertIsNotNone(lock)
                
                initial_lock_count = len(manager._engine_locks)
                print(f"Initial lock count: {initial_lock_count}")
                
                # Wait for cleanup interval
                time.sleep(0.2)
                
                # Trigger cleanup by requesting more locks
                for name in engine_names[10:]:
                    manager._get_or_create_lock(name)
                
                # Force cleanup
                manager._cleanup_unused_locks()
                
                final_lock_count = len(manager._engine_locks)
                print(f"Final lock count after cleanup: {final_lock_count}")
                
                # Old locks should be cleaned up
                self.assertLess(final_lock_count, initial_lock_count, 
                              "Lock cleanup didn't remove old locks")
    
    def test_performance_benchmark(self):
        """Benchmark performance to ensure no significant degradation"""
        print("\n=== Test: Performance Benchmark ===")
        
        with patch.object(STTEngineManager, 'ENGINES', {'mock': MockEngine}):
            manager = STTEngineManager(default_engine='mock')
            
            # Warm up - initialize engine
            manager.get_engine('mock')
            
            # Benchmark sequential access (should be fast with fixed implementation)
            num_iterations = 10000
            
            start_time = time.time()
            for _ in range(num_iterations):
                engine = manager.get_engine('mock')
                _ = engine.name  # Use the engine
            sequential_time = time.time() - start_time
            
            # Benchmark concurrent access
            num_threads = 10
            iterations_per_thread = 1000
            
            def concurrent_worker():
                for _ in range(iterations_per_thread):
                    engine = manager.get_engine('mock')
                    _ = engine.name
            
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(concurrent_worker) for _ in range(num_threads)]
                
                for future in as_completed(futures):
                    future.result()
            concurrent_time = time.time() - start_time
            
            print(f"Sequential access ({num_iterations} iterations): {sequential_time:.4f}s")
            print(f"Avg time per call: {(sequential_time / num_iterations) * 1000:.4f}ms")
            print(f"Concurrent access ({num_threads} threads, {iterations_per_thread} each): {concurrent_time:.4f}s")
            print(f"Avg time per call: {(concurrent_time / (num_threads * iterations_per_thread)) * 1000:.4f}ms")
            
            # Performance should be reasonable
            avg_sequential_time = sequential_time / num_iterations
            self.assertLess(avg_sequential_time, 0.001, "Sequential access too slow (>1ms per call)")
            
            avg_concurrent_time = concurrent_time / (num_threads * iterations_per_thread)
            self.assertLess(avg_concurrent_time, 0.01, "Concurrent access too slow (>10ms per call)")
    
    def test_exception_handling_under_concurrency(self):
        """Test that exceptions during initialization are handled correctly"""
        print("\n=== Test: Exception Handling Under Concurrency ===")
        
        class FailingEngine(MockEngine):
            """Engine that fails during initialization"""
            fail_count = 0
            fail_lock = threading.Lock()
            
            def __init__(self, config=None):
                with FailingEngine.fail_lock:
                    FailingEngine.fail_count += 1
                    if FailingEngine.fail_count <= 5:
                        raise RuntimeError(f"Initialization failed (attempt {FailingEngine.fail_count})")
                
                super().__init__(config)
                self.is_available = False  # Mark as unavailable
        
        with patch.object(STTEngineManager, 'ENGINES', {'failing': FailingEngine}):
            manager = STTEngineManager(default_engine='mock')
            
            num_threads = 20
            results = []
            barrier = threading.Barrier(num_threads)
            
            def try_get_engine(thread_id):
                barrier.wait()
                
                try:
                    engine = manager.get_engine('failing')
                    return (thread_id, 'success', id(engine))
                except ValueError as e:
                    return (thread_id, 'error', str(e))
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(try_get_engine, i) for i in range(num_threads)]
                
                for future in as_completed(futures):
                    results.append(future.result())
            
            # Count successes and failures
            successes = sum(1 for _, status, _ in results if status == 'success')
            failures = sum(1 for _, status, _ in results if status == 'error')
            
            print(f"Successes: {successes}, Failures: {failures}")
            print(f"FailingEngine initialization attempts: {FailingEngine.fail_count}")
            
            # All threads should get consistent results
            self.assertEqual(failures, num_threads, "All threads should fail since engine is unavailable")
            # Should not have excessive initialization attempts
            self.assertLessEqual(FailingEngine.fail_count, 10, "Too many initialization attempts")


def run_stress_test():
    """Run the stress test suite with detailed output"""
    print("=" * 70)
    print("STT Engine Manager - Race Condition Test Suite")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEngineInitRaceCondition)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed! Race condition fix verified.")
    else:
        print("\n✗ Some tests failed. Review the race condition fix.")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_stress_test()
    sys.exit(0 if success else 1)