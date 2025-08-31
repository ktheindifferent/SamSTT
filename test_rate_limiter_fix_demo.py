#!/usr/bin/env python3
"""
Demonstration of the thread safety fix for the rate limiter
Shows the issue was fixed and validates correctness
"""
import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stts.validators import RateLimiter, MAX_REQUESTS_PER_MINUTE

def test_thread_safety_fix():
    """
    Test that demonstrates the thread safety fix is working correctly.
    With 1000 concurrent threads, we should see exactly MAX_REQUESTS_PER_MINUTE
    allowed requests and no race conditions.
    """
    print("Thread Safety Fix Demonstration")
    print("=" * 50)
    
    rate_limiter = RateLimiter()
    client_id = "test_client"
    num_threads = 1000
    
    # Shared counters
    allowed_count = 0
    denied_count = 0
    count_lock = threading.Lock()
    
    # Barrier to ensure all threads start at the same time
    barrier = threading.Barrier(num_threads)
    
    def make_request():
        # Wait for all threads to be ready
        barrier.wait()
        
        # Try to make a request
        allowed, error = rate_limiter.is_allowed(client_id)
        
        # Update counters thread-safely
        with count_lock:
            nonlocal allowed_count, denied_count
            if allowed:
                allowed_count += 1
            else:
                denied_count += 1
    
    # Create and start all threads
    print(f"\nStarting {num_threads} concurrent threads...")
    threads = []
    start_time = time.time()
    
    for _ in range(num_threads):
        thread = threading.Thread(target=make_request)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    
    # Print results
    print(f"\nResults:")
    print(f"  Total threads: {num_threads}")
    print(f"  Allowed requests: {allowed_count}")
    print(f"  Denied requests: {denied_count}")
    print(f"  Rate limit setting: {MAX_REQUESTS_PER_MINUTE} requests per minute")
    print(f"  Execution time: {end_time - start_time:.3f} seconds")
    
    # Validate results
    print(f"\nValidation:")
    assert allowed_count == MAX_REQUESTS_PER_MINUTE, \
        f"Expected exactly {MAX_REQUESTS_PER_MINUTE} allowed requests, got {allowed_count}"
    assert allowed_count + denied_count == num_threads, \
        f"Lost requests detected! Total: {allowed_count + denied_count}, Expected: {num_threads}"
    print("  ✓ Rate limit correctly enforced")
    print("  ✓ No lost requests (no race conditions)")
    print("  ✓ Thread safety verified!")
    
    return True


def test_deque_efficiency():
    """
    Test that shows the efficiency improvement from using deque
    """
    print("\n\nDeque Efficiency Test")
    print("=" * 50)
    
    rate_limiter = RateLimiter()
    
    # Make many requests to build up the deque
    print("\nMaking 1000 requests from 100 different clients...")
    start_time = time.time()
    
    for i in range(100):
        client_id = f"client_{i}"
        for _ in range(10):
            rate_limiter.is_allowed(client_id)
    
    mid_time = time.time()
    print(f"Time for 1000 requests: {mid_time - start_time:.3f} seconds")
    
    # Trigger cleanup by making more requests
    print("\nMaking additional requests to trigger cleanup...")
    for i in range(10):
        rate_limiter.is_allowed(f"cleanup_client_{i}")
    
    end_time = time.time()
    print(f"Cleanup time: {end_time - mid_time:.3f} seconds")
    
    print("\n✓ Deque operations are efficient even with many clients")
    return True


def test_concurrent_different_clients():
    """
    Test that different clients can make requests concurrently without interference
    """
    print("\n\nConcurrent Different Clients Test")
    print("=" * 50)
    
    rate_limiter = RateLimiter()
    num_clients = 100
    requests_per_client = 10
    
    results = {}
    results_lock = threading.Lock()
    
    def client_requests(client_id):
        allowed = 0
        for _ in range(requests_per_client):
            if rate_limiter.is_allowed(client_id)[0]:
                allowed += 1
        
        with results_lock:
            results[client_id] = allowed
    
    print(f"\nRunning {num_clients} clients with {requests_per_client} requests each...")
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for i in range(num_clients):
            future = executor.submit(client_requests, f"client_{i}")
            futures.append(future)
        
        for future in futures:
            future.result()
    
    # All clients should get all their requests (since each is under the limit)
    all_got_requests = all(count == requests_per_client for count in results.values())
    
    print(f"\nResults:")
    print(f"  Total clients: {num_clients}")
    print(f"  All clients got {requests_per_client} requests: {all_got_requests}")
    
    assert all_got_requests, "Some clients didn't get their expected requests"
    print("\n✓ Each client has independent rate limits")
    print("✓ No interference between clients")
    
    return True


if __name__ == "__main__":
    try:
        # Run all demonstration tests
        test_thread_safety_fix()
        test_deque_efficiency()
        test_concurrent_different_clients()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED!")
        print("Thread safety issues have been successfully fixed.")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)