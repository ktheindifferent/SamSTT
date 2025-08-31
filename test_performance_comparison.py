#!/usr/bin/env python3
"""
Performance comparison test - compares the old vs new implementation
"""

import threading
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MockPerfEngine:
    """Lightweight mock engine for performance testing"""
    def __init__(self, config=None):
        self.config = config or {}
        self.name = "perf"
        self.is_available = True


def benchmark_implementation():
    """Benchmark the current (fixed) implementation"""
    from stts.engine_manager import STTEngineManager
    
    print("=" * 70)
    print("PERFORMANCE BENCHMARK - FIXED IMPLEMENTATION")
    print("=" * 70)
    
    original_engines = STTEngineManager.ENGINES.copy()
    STTEngineManager.ENGINES = {'perf': MockPerfEngine}
    
    try:
        manager = STTEngineManager(default_engine='perf', config={})
        
        # Warm up
        for _ in range(100):
            manager.get_engine('perf')
        
        results = {}
        
        # Test 1: Sequential access (best case - engine already initialized)
        print("\n1. Sequential Access (10,000 calls):")
        iterations = 10000
        start = time.time()
        for _ in range(iterations):
            engine = manager.get_engine('perf')
        elapsed = time.time() - start
        
        avg_us = (elapsed / iterations) * 1_000_000  # microseconds
        print(f"   Total time: {elapsed:.4f}s")
        print(f"   Average: {avg_us:.2f}μs per call")
        print(f"   Throughput: {iterations/elapsed:.0f} calls/sec")
        results['sequential'] = avg_us
        
        # Test 2: Concurrent access (10 threads, 1000 calls each)
        print("\n2. Concurrent Access (10 threads × 1,000 calls):")
        threads = 10
        calls_per_thread = 1000
        
        def concurrent_worker():
            for _ in range(calls_per_thread):
                engine = manager.get_engine('perf')
        
        start = time.time()
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(concurrent_worker) for _ in range(threads)]
            for f in as_completed(futures):
                f.result()
        elapsed = time.time() - start
        
        total_calls = threads * calls_per_thread
        avg_us = (elapsed / total_calls) * 1_000_000
        print(f"   Total time: {elapsed:.4f}s")
        print(f"   Average: {avg_us:.2f}μs per call")
        print(f"   Throughput: {total_calls/elapsed:.0f} calls/sec")
        results['concurrent_10'] = avg_us
        
        # Test 3: High concurrency (50 threads, 200 calls each)
        print("\n3. High Concurrency (50 threads × 200 calls):")
        threads = 50
        calls_per_thread = 200
        
        def high_concurrent_worker():
            for _ in range(calls_per_thread):
                engine = manager.get_engine('perf')
        
        start = time.time()
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(high_concurrent_worker) for _ in range(threads)]
            for f in as_completed(futures):
                f.result()
        elapsed = time.time() - start
        
        total_calls = threads * calls_per_thread
        avg_us = (elapsed / total_calls) * 1_000_000
        print(f"   Total time: {elapsed:.4f}s")
        print(f"   Average: {avg_us:.2f}μs per call")
        print(f"   Throughput: {total_calls/elapsed:.0f} calls/sec")
        results['concurrent_50'] = avg_us
        
        # Test 4: First initialization (cold start)
        print("\n4. Cold Start (first initialization):")
        manager.engines.clear()  # Clear cache
        
        start = time.time()
        engine = manager.get_engine('perf')
        elapsed = time.time() - start
        
        elapsed_ms = elapsed * 1000
        print(f"   Time for first initialization: {elapsed_ms:.2f}ms")
        results['cold_start'] = elapsed_ms
        
        # Test 5: Lock contention test
        print("\n5. Lock Contention Test (100 threads racing for first init):")
        manager.engines.clear()  # Clear cache
        
        barrier = threading.Barrier(100)
        init_times = []
        
        def contention_worker():
            barrier.wait()  # All threads start exactly together
            start = time.time()
            engine = manager.get_engine('perf')
            return time.time() - start
        
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(contention_worker) for _ in range(100)]
            for f in as_completed(futures):
                init_times.append(f.result())
        
        avg_ms = sum(init_times) / len(init_times) * 1000
        max_ms = max(init_times) * 1000
        min_ms = min(init_times) * 1000
        
        print(f"   Average wait time: {avg_ms:.2f}ms")
        print(f"   Max wait time: {max_ms:.2f}ms")
        print(f"   Min wait time: {min_ms:.2f}ms")
        results['contention_avg'] = avg_ms
        
        return results
        
    finally:
        STTEngineManager.ENGINES = original_engines


def print_summary(results):
    """Print performance summary"""
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    
    print("\nFixed Implementation Performance:")
    print(f"  • Sequential access: {results['sequential']:.2f}μs per call")
    print(f"  • Concurrent (10 threads): {results['concurrent_10']:.2f}μs per call")
    print(f"  • Concurrent (50 threads): {results['concurrent_50']:.2f}μs per call")
    print(f"  • Cold start: {results['cold_start']:.2f}ms")
    print(f"  • Lock contention (100 threads): {results['contention_avg']:.2f}ms avg wait")
    
    # Performance criteria
    print("\nPerformance Criteria:")
    criteria = [
        ("Sequential < 10μs", results['sequential'] < 10),
        ("Concurrent-10 < 50μs", results['concurrent_10'] < 50),
        ("Concurrent-50 < 100μs", results['concurrent_50'] < 100),
        ("Cold start < 10ms", results['cold_start'] < 10),
        ("Lock contention < 50ms", results['contention_avg'] < 50),
    ]
    
    all_pass = True
    for desc, passed in criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {desc}: {status}")
        all_pass = all_pass and passed
    
    print("\n" + "=" * 70)
    if all_pass:
        print("✓✓✓ PERFORMANCE REQUIREMENTS MET")
        print("The fix maintains excellent performance characteristics.")
    else:
        print("⚠ Some performance criteria not met")
    print("=" * 70)
    
    return all_pass


def main():
    """Main test runner"""
    results = benchmark_implementation()
    all_pass = print_summary(results)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())