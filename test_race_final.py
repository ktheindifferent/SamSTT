#!/usr/bin/env python3
"""
Final test for race condition fix - properly accounts for initialization
"""

import threading
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MockEngineFinal:
    """Mock engine with proper tracking"""
    instances_created = 0
    instances_lock = threading.Lock()
    creation_log = []
    
    def __init__(self, config=None):
        self.config = config or {}
        self.name = "mock"
        self.is_available = True
        
        # Track instance creation
        with MockEngineFinal.instances_lock:
            MockEngineFinal.instances_created += 1
            instance_num = MockEngineFinal.instances_created
            instance_id = id(self)
            thread_name = threading.current_thread().name
            
            log_entry = {
                'instance_num': instance_num,
                'instance_id': instance_id,
                'thread': thread_name,
            }
            MockEngineFinal.creation_log.append(log_entry)
        
        # Simulate initialization work
        time.sleep(0.005)
        self.instance_id = instance_id
        self.instance_num = instance_num
    
    @classmethod
    def reset_tracking(cls):
        with cls.instances_lock:
            cls.instances_created = 0
            cls.creation_log.clear()


def run_test_scenario(name, num_threads):
    """Run a single test scenario"""
    from stts.engine_manager import STTEngineManager
    
    print(f"\n{'='*60}")
    print(f"TEST: {name} ({num_threads} threads)")
    print(f"{'='*60}")
    
    MockEngineFinal.reset_tracking()
    
    # Patch the ENGINES registry
    original_engines = STTEngineManager.ENGINES.copy()
    STTEngineManager.ENGINES = {'test_engine': MockEngineFinal}
    
    try:
        # Create manager without auto-initialization
        manager = STTEngineManager(default_engine='nonexistent', config={})
        
        # Ensure engines dict is empty
        manager.engines.clear()
        
        # Now test concurrent initialization
        barrier = threading.Barrier(num_threads)
        results = []
        
        def get_engine_worker(thread_id):
            thread = threading.current_thread()
            thread.name = f"W{thread_id:03d}"
            
            # Synchronize all threads
            barrier.wait()
            
            try:
                engine = manager.get_engine('test_engine')
                return (thread_id, id(engine), engine.instance_num)
            except Exception as e:
                return (thread_id, None, str(e))
        
        # Launch threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(get_engine_worker, i) for i in range(num_threads)]
            
            for future in as_completed(futures):
                results.append(future.result())
        
        # Analyze results
        unique_engine_ids = set(r[1] for r in results if r[1] is not None)
        unique_instance_nums = set(r[2] for r in results if isinstance(r[2], int))
        errors = [r for r in results if r[1] is None]
        
        print(f"Results:")
        print(f"  • Instances created: {MockEngineFinal.instances_created}")
        print(f"  • Unique engine IDs returned: {len(unique_engine_ids)}")
        print(f"  • Unique instance numbers: {unique_instance_nums}")
        print(f"  • Errors: {len(errors)}")
        
        if MockEngineFinal.instances_created == 1:
            print(f"  ✓ PASS: No race condition - only 1 instance created")
            return True
        else:
            print(f"  ✗ FAIL: Race condition detected - {MockEngineFinal.instances_created} instances created")
            print(f"  Creating threads: {[e['thread'] for e in MockEngineFinal.creation_log]}")
            return False
            
    finally:
        STTEngineManager.ENGINES = original_engines


def test_performance():
    """Test performance of the fixed implementation"""
    from stts.engine_manager import STTEngineManager
    
    print(f"\n{'='*60}")
    print(f"PERFORMANCE TEST")
    print(f"{'='*60}")
    
    original_engines = STTEngineManager.ENGINES.copy()
    STTEngineManager.ENGINES = {'perf_test': MockEngineFinal}
    
    try:
        manager = STTEngineManager(default_engine='perf_test', config={})
        
        # Warm up
        manager.get_engine('perf_test')
        
        # Test sequential access
        iterations = 10000
        start = time.time()
        for _ in range(iterations):
            engine = manager.get_engine('perf_test')
        elapsed = time.time() - start
        
        avg_ms = (elapsed / iterations) * 1000
        print(f"Sequential access:")
        print(f"  • {iterations} iterations in {elapsed:.3f}s")
        print(f"  • Average: {avg_ms:.4f}ms per call")
        
        if avg_ms < 0.1:  # Should be much faster than 0.1ms
            print(f"  ✓ PASS: Excellent performance")
            return True
        elif avg_ms < 1.0:
            print(f"  ✓ PASS: Good performance")
            return True
        else:
            print(f"  ✗ FAIL: Poor performance")
            return False
            
    finally:
        STTEngineManager.ENGINES = original_engines


def main():
    """Main test runner"""
    print("=" * 70)
    print("RACE CONDITION FIX VERIFICATION TEST SUITE")
    print("=" * 70)
    
    all_passed = True
    
    # Test different concurrency levels
    test_scenarios = [
        ("Low concurrency", 10),
        ("Medium concurrency", 50),
        ("High concurrency", 100),
        ("Extreme concurrency", 200),
    ]
    
    for name, threads in test_scenarios:
        passed = run_test_scenario(name, threads)
        all_passed = all_passed and passed
    
    # Test performance
    perf_passed = test_performance()
    all_passed = all_passed and perf_passed
    
    # Stress test: Multiple rounds
    print(f"\n{'='*60}")
    print(f"STRESS TEST: 20 rounds x 50 threads")
    print(f"{'='*60}")
    
    from stts.engine_manager import STTEngineManager
    
    original_engines = STTEngineManager.ENGINES.copy()
    STTEngineManager.ENGINES = {'stress_test': MockEngineFinal}
    
    try:
        failures = 0
        for round_num in range(20):
            MockEngineFinal.reset_tracking()
            
            manager = STTEngineManager(default_engine='nonexistent', config={})
            manager.engines.clear()
            
            barrier = threading.Barrier(50)
            
            def stress_worker(tid):
                barrier.wait()
                return manager.get_engine('stress_test')
            
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = [executor.submit(stress_worker, i) for i in range(50)]
                for f in as_completed(futures):
                    f.result()
            
            if MockEngineFinal.instances_created != 1:
                failures += 1
                print(f"  Round {round_num+1}: ✗ FAIL ({MockEngineFinal.instances_created} instances)")
            else:
                print(f"  Round {round_num+1}: ✓ PASS", end=" ")
                if (round_num + 1) % 5 == 0:
                    print()  # New line every 5 rounds
        
        if failures == 0:
            print(f"\n  ✓ All 20 rounds passed!")
        else:
            print(f"\n  ✗ {failures} rounds failed")
            all_passed = False
            
    finally:
        STTEngineManager.ENGINES = original_engines
    
    # Final summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED - RACE CONDITION FIX VERIFIED!")
        print("The double-checked locking pattern is now properly implemented.")
        print("No duplicate engine instances are created under concurrent access.")
    else:
        print("✗✗✗ SOME TESTS FAILED")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())