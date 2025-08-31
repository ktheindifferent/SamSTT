#!/usr/bin/env python3
"""
Simplified test for race condition fix - only tests the core logic
"""

import threading
import time
import sys
import os
from typing import Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MockBaseEngine:
    """Mock base engine"""
    def __init__(self, config=None):
        self.config = config or {}
        self.name = "mock"
        self.is_available = True


class MockEngineTracker(MockBaseEngine):
    """Mock engine that tracks initialization"""
    instances_created = 0
    instances_lock = threading.Lock()
    instance_ids: Set[int] = set()
    initialization_times = []
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Track instance creation atomically
        with MockEngineTracker.instances_lock:
            MockEngineTracker.instances_created += 1
            instance_id = id(self)
            MockEngineTracker.instance_ids.add(instance_id)
            MockEngineTracker.initialization_times.append(time.time())
            print(f"[Thread {threading.current_thread().name}] Created instance #{MockEngineTracker.instances_created} (id: {instance_id})")
        
        # Simulate initialization work
        time.sleep(0.01)
        self.instance_id = instance_id
    
    @classmethod
    def reset_tracking(cls):
        with cls.instances_lock:
            cls.instances_created = 0
            cls.instance_ids.clear()
            cls.initialization_times.clear()


def test_race_condition():
    """Test the race condition fix"""
    print("=" * 70)
    print("Testing Race Condition Fix in Engine Manager")
    print("=" * 70)
    
    # Import only the manager to test
    from stts.engine_manager import STTEngineManager
    
    # Reset tracking
    MockEngineTracker.reset_tracking()
    
    # Patch the ENGINES registry
    original_engines = STTEngineManager.ENGINES.copy()
    STTEngineManager.ENGINES = {'mock': MockEngineTracker}
    
    try:
        manager = STTEngineManager(default_engine='mock')
        
        # Clear any pre-initialized engines
        manager.engines.clear()
        
        print("\n--- Test 1: Concurrent Initialization (100 threads) ---")
        
        num_threads = 100
        results = []
        barrier = threading.Barrier(num_threads)
        
        def get_engine(thread_id):
            """Worker function to get engine"""
            thread = threading.current_thread()
            thread.name = f"Worker-{thread_id:03d}"
            
            # Wait for all threads to be ready
            barrier.wait()
            
            try:
                engine = manager.get_engine('mock')
                return (thread_id, id(engine), engine.instance_id)
            except Exception as e:
                return (thread_id, None, str(e))
        
        # Launch threads
        print(f"Launching {num_threads} concurrent threads...")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(get_engine, i) for i in range(num_threads)]
            
            for future in as_completed(futures):
                results.append(future.result())
        
        # Analyze results
        engine_ids = set()
        instance_ids = set()
        errors = []
        
        for thread_id, engine_id, instance_id in results:
            if engine_id is None:
                errors.append((thread_id, instance_id))
            else:
                engine_ids.add(engine_id)
                if isinstance(instance_id, int):
                    instance_ids.add(instance_id)
        
        print(f"\nResults:")
        print(f"  Threads executed: {num_threads}")
        print(f"  Unique engine IDs returned: {len(engine_ids)}")
        print(f"  Unique instance IDs: {len(instance_ids)}")
        print(f"  MockEngine instances created: {MockEngineTracker.instances_created}")
        print(f"  Errors: {len(errors)}")
        
        # Check for race condition
        if MockEngineTracker.instances_created == 1:
            print("\n✓ PASS: No race condition detected - only 1 instance created")
        else:
            print(f"\n✗ FAIL: Race condition detected - {MockEngineTracker.instances_created} instances created!")
            return False
        
        if len(engine_ids) == 1:
            print("✓ PASS: All threads received the same engine object")
        else:
            print(f"✗ FAIL: Multiple engine objects returned: {len(engine_ids)}")
            return False
        
        # Test 2: Stress test with multiple rounds
        print("\n--- Test 2: Stress Test (10 rounds x 50 threads) ---")
        
        total_instances_created = 0
        
        for round_num in range(10):
            MockEngineTracker.reset_tracking()
            manager.engines.clear()  # Force re-initialization
            
            barrier = threading.Barrier(50)
            round_results = []
            
            def stress_worker(worker_id):
                barrier.wait()
                engine = manager.get_engine('mock')
                return id(engine)
            
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = [executor.submit(stress_worker, i) for i in range(50)]
                
                for future in as_completed(futures):
                    round_results.append(future.result())
            
            unique_engines = set(round_results)
            instances = MockEngineTracker.instances_created
            total_instances_created += instances
            
            print(f"  Round {round_num + 1}: {len(unique_engines)} unique engines, "
                  f"{instances} instances created", end="")
            
            if instances == 1:
                print(" ✓")
            else:
                print(f" ✗ RACE CONDITION!")
                return False
        
        print(f"\n✓ PASS: All {10} rounds completed successfully")
        print(f"  Total instances that should have been created: 10")
        print(f"  Total instances actually created: {total_instances_created}")
        
        if total_instances_created == 10:
            print("✓ PASS: Correct number of instances across all rounds")
        else:
            print(f"✗ FAIL: Expected 10 instances, got {total_instances_created}")
            return False
        
        # Test 3: Performance test
        print("\n--- Test 3: Performance Benchmark ---")
        
        # Initialize engine first
        manager.get_engine('mock')
        
        # Sequential access
        num_iterations = 10000
        start_time = time.time()
        for _ in range(num_iterations):
            engine = manager.get_engine('mock')
        sequential_time = time.time() - start_time
        
        avg_time_ms = (sequential_time / num_iterations) * 1000
        print(f"  Sequential access ({num_iterations} calls): {sequential_time:.4f}s")
        print(f"  Average time per call: {avg_time_ms:.4f}ms")
        
        if avg_time_ms < 1.0:
            print("✓ PASS: Performance is good (<1ms per call)")
        else:
            print(f"✗ WARNING: Performance may be degraded ({avg_time_ms:.4f}ms per call)")
        
        return True
        
    finally:
        # Restore original engines
        STTEngineManager.ENGINES = original_engines


def main():
    """Main test runner"""
    success = test_race_condition()
    
    print("\n" + "=" * 70)
    if success:
        print("✓✓✓ ALL TESTS PASSED - Race condition fix verified!")
        print("=" * 70)
        return 0
    else:
        print("✗✗✗ TESTS FAILED - Race condition still present!")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())