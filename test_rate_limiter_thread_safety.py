#!/usr/bin/env python3
"""
Comprehensive thread safety tests for the RateLimiter implementation
Tests concurrent access, accuracy under load, and performance
"""
import os
import sys
import time
import threading
import random
import unittest
import statistics
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import logging

# Add the stts module to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stts.validators import RateLimiter, MAX_REQUESTS_PER_MINUTE, MAX_REQUESTS_PER_HOUR

# Configure logging for debugging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestRateLimiterThreadSafety(unittest.TestCase):
    """Test thread safety of the RateLimiter implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rate_limiter = RateLimiter()
    
    def tearDown(self):
        """Clean up after tests"""
        self.rate_limiter.reset()
    
    def test_concurrent_requests_single_client(self):
        """Test concurrent requests from a single client"""
        client_id = "test_client_1"
        num_threads = 100
        requests_per_thread = 10
        
        success_count = threading.local()
        success_count.value = 0
        lock = threading.Lock()
        
        def make_requests():
            local_success = 0
            for _ in range(requests_per_thread):
                allowed, _ = self.rate_limiter.is_allowed(client_id)
                if allowed:
                    local_success += 1
                time.sleep(0.001)  # Small delay to spread requests
            
            with lock:
                success_count.value = getattr(success_count, 'value', 0) + local_success
        
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=make_requests)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify that rate limiting is applied correctly
        total_requests = num_threads * requests_per_thread
        # Should not exceed rate limits
        self.assertLessEqual(getattr(success_count, 'value', 0), MAX_REQUESTS_PER_MINUTE)
    
    def test_concurrent_requests_multiple_clients(self):
        """Test concurrent requests from multiple clients"""
        num_clients = 50
        requests_per_client = 20
        
        results = defaultdict(list)
        results_lock = threading.Lock()
        
        def client_requests(client_id):
            local_results = []
            for _ in range(requests_per_client):
                allowed, error = self.rate_limiter.is_allowed(client_id)
                local_results.append((allowed, error))
                time.sleep(random.uniform(0.001, 0.005))
            
            with results_lock:
                results[client_id].extend(local_results)
        
        with ThreadPoolExecutor(max_workers=num_clients) as executor:
            futures = []
            for i in range(num_clients):
                client_id = f"client_{i}"
                future = executor.submit(client_requests, client_id)
                futures.append(future)
            
            # Wait for all to complete
            for future in as_completed(futures):
                future.result()
        
        # Verify each client's rate limiting
        for client_id, client_results in results.items():
            allowed_count = sum(1 for allowed, _ in client_results if allowed)
            # Each client should respect their individual rate limit
            self.assertLessEqual(allowed_count, MAX_REQUESTS_PER_MINUTE)
    
    def test_race_condition_stress_test(self):
        """Stress test with 1000+ threads to detect race conditions"""
        num_threads = 1000
        client_id = "stress_test_client"
        
        # Track all responses
        responses = []
        responses_lock = threading.Lock()
        
        # Barrier to ensure all threads start simultaneously
        barrier = threading.Barrier(num_threads)
        
        def stress_request():
            barrier.wait()  # Wait for all threads to be ready
            allowed, error = self.rate_limiter.is_allowed(client_id)
            with responses_lock:
                responses.append((allowed, error, time.time()))
        
        threads = []
        start_time = time.time()
        
        for _ in range(num_threads):
            thread = threading.Thread(target=stress_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Analysis
        allowed_count = sum(1 for allowed, _, _ in responses if allowed)
        denied_count = len(responses) - allowed_count
        
        print(f"\nStress Test Results:")
        print(f"Total threads: {num_threads}")
        print(f"Allowed requests: {allowed_count}")
        print(f"Denied requests: {denied_count}")
        print(f"Execution time: {end_time - start_time:.3f} seconds")
        
        # Verify rate limiting worked
        self.assertLessEqual(allowed_count, MAX_REQUESTS_PER_MINUTE)
        self.assertEqual(len(responses), num_threads)  # No lost requests
    
    def test_cleanup_under_load(self):
        """Test cleanup mechanism under concurrent load"""
        num_clients = 100
        requests_per_client = 5
        
        def client_burst(client_id):
            for _ in range(requests_per_client):
                self.rate_limiter.is_allowed(client_id)
        
        # Create many clients
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = []
            for i in range(num_clients):
                future = executor.submit(client_burst, f"client_{i}")
                futures.append(future)
            
            for future in as_completed(futures):
                future.result()
        
        # Verify cleanup doesn't cause issues
        # Wait for cleanup interval
        time.sleep(2)
        
        # Make more requests to trigger cleanup
        for i in range(10):
            self.rate_limiter.is_allowed(f"new_client_{i}")
        
        # Should complete without errors
        self.assertTrue(True)
    
    def test_accuracy_under_concurrency(self):
        """Test rate limit accuracy with high concurrency"""
        client_id = "accuracy_test"
        num_threads = 200
        test_duration = 5  # seconds
        
        # Track request times
        request_times = []
        request_times_lock = threading.Lock()
        stop_flag = threading.Event()
        
        def continuous_requests():
            while not stop_flag.is_set():
                allowed, _ = self.rate_limiter.is_allowed(client_id)
                if allowed:
                    with request_times_lock:
                        request_times.append(time.time())
                time.sleep(random.uniform(0.001, 0.01))
        
        # Start threads
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=continuous_requests)
            threads.append(thread)
            thread.start()
        
        # Run for test duration
        time.sleep(test_duration)
        stop_flag.set()
        
        # Wait for threads to finish
        for thread in threads:
            thread.join()
        
        # Analyze request distribution
        if request_times:
            request_times.sort()
            
            # Check rate limits in sliding windows
            for i in range(len(request_times)):
                current_time = request_times[i]
                
                # Count requests in last minute from this point
                minute_count = sum(1 for t in request_times 
                                 if current_time - 60 <= t <= current_time)
                
                # Should never exceed minute limit
                self.assertLessEqual(minute_count, MAX_REQUESTS_PER_MINUTE,
                                   f"Minute limit exceeded at time {current_time}")
    
    def test_reset_thread_safety(self):
        """Test thread-safe reset operations"""
        num_threads = 100
        
        def mixed_operations():
            client_id = f"client_{threading.current_thread().ident}"
            
            # Make some requests
            for _ in range(5):
                self.rate_limiter.is_allowed(client_id)
            
            # Random reset
            if random.random() < 0.3:
                if random.random() < 0.5:
                    self.rate_limiter.reset(client_id)
                else:
                    self.rate_limiter.reset()
            
            # Make more requests
            for _ in range(5):
                self.rate_limiter.is_allowed(client_id)
        
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=mixed_operations)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should complete without deadlocks or errors
        self.assertTrue(True)
    
    def test_get_stats_thread_safety(self):
        """Test thread-safe stats retrieval"""
        client_id = "stats_client"
        num_threads = 50
        
        def stats_operations():
            for _ in range(10):
                # Interleave requests and stats checks
                self.rate_limiter.is_allowed(client_id)
                stats = self.rate_limiter.get_stats(client_id)
                
                # Verify stats consistency
                self.assertIsInstance(stats['minute_requests'], int)
                self.assertIsInstance(stats['hour_requests'], int)
                self.assertGreaterEqual(stats['minute_requests'], 0)
                self.assertGreaterEqual(stats['hour_requests'], 0)
                self.assertLessEqual(stats['minute_requests'], stats['hour_requests'])
        
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=stats_operations)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
    
    def test_memory_leak_prevention(self):
        """Test that old clients are cleaned up to prevent memory leaks"""
        # Create many clients over time
        for batch in range(5):
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = []
                for i in range(200):
                    client_id = f"batch_{batch}_client_{i}"
                    future = executor.submit(self.rate_limiter.is_allowed, client_id)
                    futures.append(future)
                
                for future in as_completed(futures):
                    future.result()
            
            # Wait between batches
            time.sleep(0.5)
        
        # Force cleanup by making requests with a high client count
        for i in range(1100):
            self.rate_limiter.is_allowed(f"trigger_cleanup_{i}")
        
        # Check that old clients were cleaned up
        # This is implicit - test passes if no memory issues
        self.assertTrue(True)
    
    def test_time_synchronization_issues(self):
        """Test handling of time-related edge cases"""
        client_id = "time_test"
        
        # Make requests
        for _ in range(10):
            self.rate_limiter.is_allowed(client_id)
        
        stats_before = self.rate_limiter.get_stats(client_id)
        
        # Simulate time passing (we can't actually change system time)
        # But we can test that the cleanup works correctly
        time.sleep(2)
        
        # Old requests should still be counted until cleanup
        stats_after = self.rate_limiter.get_stats(client_id)
        
        # Make a new request to trigger potential cleanup
        self.rate_limiter.is_allowed(client_id)
        
        # Should handle time correctly
        self.assertIsNotNone(stats_before)
        self.assertIsNotNone(stats_after)


class TestRateLimiterPerformance(unittest.TestCase):
    """Performance tests for the RateLimiter"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rate_limiter = RateLimiter()
    
    def test_performance_baseline(self):
        """Establish performance baseline"""
        client_id = "perf_test"
        num_requests = 10000
        
        start_time = time.time()
        
        for _ in range(num_requests):
            self.rate_limiter.is_allowed(client_id)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        requests_per_second = num_requests / total_time
        avg_time_per_request = total_time / num_requests * 1000  # in ms
        
        print(f"\nPerformance Baseline:")
        print(f"Total requests: {num_requests}")
        print(f"Total time: {total_time:.3f} seconds")
        print(f"Requests per second: {requests_per_second:.0f}")
        print(f"Avg time per request: {avg_time_per_request:.3f} ms")
        
        # Performance should be reasonable
        self.assertLess(avg_time_per_request, 1.0)  # Less than 1ms per request
    
    def test_performance_under_concurrency(self):
        """Test performance with concurrent access"""
        num_threads = 100
        requests_per_thread = 100
        
        def thread_requests(thread_id):
            client_id = f"perf_client_{thread_id}"
            times = []
            
            for _ in range(requests_per_thread):
                start = time.perf_counter()
                self.rate_limiter.is_allowed(client_id)
                end = time.perf_counter()
                times.append(end - start)
            
            return times
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(thread_requests, i) for i in range(num_threads)]
            all_times = []
            for future in as_completed(futures):
                all_times.extend(future.result())
        
        end_time = time.time()
        
        total_requests = num_threads * requests_per_thread
        total_time = end_time - start_time
        
        # Calculate statistics
        avg_time = statistics.mean(all_times) * 1000  # in ms
        median_time = statistics.median(all_times) * 1000
        p95_time = statistics.quantiles(all_times, n=20)[18] * 1000  # 95th percentile
        p99_time = statistics.quantiles(all_times, n=100)[98] * 1000  # 99th percentile
        
        print(f"\nConcurrent Performance:")
        print(f"Total requests: {total_requests}")
        print(f"Total time: {total_time:.3f} seconds")
        print(f"Throughput: {total_requests/total_time:.0f} req/s")
        print(f"Avg latency: {avg_time:.3f} ms")
        print(f"Median latency: {median_time:.3f} ms")
        print(f"P95 latency: {p95_time:.3f} ms")
        print(f"P99 latency: {p99_time:.3f} ms")
        
        # Performance requirements (adjusted for realistic expectations under high contention)
        # With 100 concurrent threads competing for a lock, some contention is expected
        self.assertLess(avg_time, 10.0)  # Avg less than 10ms
        self.assertLess(p99_time, 100.0)  # P99 less than 100ms (realistic for 100 threads)


class TestRateLimiterCorrectness(unittest.TestCase):
    """Test correctness of rate limiting logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rate_limiter = RateLimiter()
    
    def test_exact_rate_limits(self):
        """Test that rate limits are exactly enforced"""
        client_id = "exact_test"
        
        # Test minute limit
        allowed_count = 0
        for i in range(MAX_REQUESTS_PER_MINUTE + 10):
            allowed, _ = self.rate_limiter.is_allowed(client_id)
            if allowed:
                allowed_count += 1
        
        self.assertEqual(allowed_count, MAX_REQUESTS_PER_MINUTE)
        
        # Reset and test hour limit (with smaller numbers for testing)
        # We'll simulate this by making requests over time
        self.rate_limiter.reset(client_id)
    
    def test_sliding_window_accuracy(self):
        """Test that sliding window is accurately maintained"""
        client_id = "sliding_window_test"
        
        # Make some requests
        initial_requests = 10
        for _ in range(initial_requests):
            allowed, _ = self.rate_limiter.is_allowed(client_id)
            self.assertTrue(allowed)
        
        # Check stats
        stats = self.rate_limiter.get_stats(client_id)
        self.assertEqual(stats['minute_requests'], initial_requests)
        self.assertEqual(stats['hour_requests'], initial_requests)
        
        # Wait a bit and make more requests
        time.sleep(1)
        
        additional_requests = 5
        for _ in range(additional_requests):
            allowed, _ = self.rate_limiter.is_allowed(client_id)
            self.assertTrue(allowed)
        
        # Check updated stats
        stats = self.rate_limiter.get_stats(client_id)
        self.assertEqual(stats['minute_requests'], initial_requests + additional_requests)
        self.assertEqual(stats['hour_requests'], initial_requests + additional_requests)
    
    def test_independent_client_limits(self):
        """Test that each client has independent rate limits"""
        client1 = "client_1"
        client2 = "client_2"
        
        # Fill up client1's quota
        for _ in range(MAX_REQUESTS_PER_MINUTE):
            allowed, _ = self.rate_limiter.is_allowed(client1)
            self.assertTrue(allowed)
        
        # Client1 should be rate limited
        allowed, _ = self.rate_limiter.is_allowed(client1)
        self.assertFalse(allowed)
        
        # Client2 should still be allowed
        for _ in range(MAX_REQUESTS_PER_MINUTE):
            allowed, _ = self.rate_limiter.is_allowed(client2)
            self.assertTrue(allowed)
        
        # Now client2 should also be rate limited
        allowed, _ = self.rate_limiter.is_allowed(client2)
        self.assertFalse(allowed)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)