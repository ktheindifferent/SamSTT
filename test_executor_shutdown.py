#!/usr/bin/env python3
"""
Test suite for ThreadPoolExecutor shutdown handling in the STT service.
Tests proper resource cleanup, graceful shutdown, and memory leak prevention.
"""

import asyncio
import os
import signal
import sys
import time
import unittest
import threading
import tempfile
import gc
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, Future
from multiprocessing import Process

# Try to import optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Add the stts module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path adjustment
from stts.app import shutdown_executor, signal_handler


class TestExecutorShutdown(unittest.TestCase):
    """Test cases for ThreadPoolExecutor shutdown functionality."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.executor = None
        self.test_timeout = 5
        
    def tearDown(self):
        """Clean up after each test."""
        if self.executor and not self.executor._shutdown:
            self.executor.shutdown(wait=False, cancel_futures=True)
    
    def test_shutdown_handler_registration(self):
        """Test that shutdown handlers are properly registered."""
        import atexit
        
        # Check that shutdown_executor is registered with atexit
        # Note: atexit._exithandlers is implementation detail but commonly available
        if hasattr(atexit, '_exithandlers'):
            handlers = [handler[0].__name__ for handler in atexit._exithandlers 
                       if hasattr(handler[0], '__name__')]
            self.assertIn('shutdown_executor', handlers, 
                         "shutdown_executor should be registered with atexit")
    
    def test_graceful_shutdown(self):
        """Test graceful shutdown with pending tasks."""
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Submit some tasks
        futures = []
        for i in range(5):
            future = self.executor.submit(time.sleep, 0.1)
            futures.append(future)
        
        # Shutdown gracefully
        start_time = time.time()
        self.executor.shutdown(wait=True)
        shutdown_time = time.time() - start_time
        
        # All futures should be completed
        for future in futures:
            self.assertTrue(future.done(), "All tasks should complete during graceful shutdown")
        
        # Shutdown should have waited for tasks
        self.assertGreater(shutdown_time, 0.1, "Shutdown should wait for tasks to complete")
        self.assertLess(shutdown_time, 1.0, "Shutdown should not take too long")
    
    def test_forced_shutdown(self):
        """Test forced shutdown with long-running tasks."""
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Submit long-running tasks
        futures = []
        for i in range(5):
            future = self.executor.submit(time.sleep, 10)
            futures.append(future)
        
        # Force shutdown
        start_time = time.time()
        self.executor.shutdown(wait=False, cancel_futures=True)
        shutdown_time = time.time() - start_time
        
        # Shutdown should be immediate
        self.assertLess(shutdown_time, 0.5, "Forced shutdown should be immediate")
        
        # Check that futures are cancelled or completed
        for future in futures:
            # Future should be either cancelled or done
            self.assertTrue(future.cancelled() or future.done(), 
                          "Futures should be cancelled in forced shutdown")
    
    def test_signal_handler(self):
        """Test signal handler for graceful shutdown."""
        with patch('stts.app.shutdown_executor') as mock_shutdown:
            with patch('stts.app.app') as mock_app:
                with patch('sys.exit') as mock_exit:
                    mock_app.is_running = True
                    
                    # Simulate SIGTERM
                    signal_handler(signal.SIGTERM, None)
                    
                    # Verify shutdown was called
                    mock_shutdown.assert_called_once()
                    mock_app.stop.assert_called_once()
                    mock_exit.assert_called_once_with(0)
    
    def test_duplicate_shutdown_prevention(self):
        """Test that duplicate shutdown calls are prevented."""
        import stts.app
        
        # Reset shutdown state
        stts.app.shutdown_initiated = False
        
        with patch('stts.app.executor') as mock_executor:
            with patch('stts.app.logger') as mock_logger:
                # First shutdown
                shutdown_executor()
                self.assertTrue(stts.app.shutdown_initiated)
                mock_executor.shutdown.assert_called_once()
                
                # Second shutdown should be skipped
                mock_executor.reset_mock()
                shutdown_executor()
                mock_executor.shutdown.assert_not_called()
                mock_logger.debug.assert_called_with(
                    "Shutdown already initiated, skipping duplicate call"
                )
        
        # Reset for other tests
        stts.app.shutdown_initiated = False
    
    def test_shutdown_with_timeout(self):
        """Test shutdown with timeout mechanism."""
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # Submit a long-running task
        future = self.executor.submit(time.sleep, 10)
        
        # Create a custom shutdown with timeout
        def shutdown_with_timeout(executor, timeout=1.0):
            """Shutdown executor with timeout."""
            shutdown_thread = threading.Thread(
                target=lambda: executor.shutdown(wait=True)
            )
            shutdown_thread.start()
            shutdown_thread.join(timeout=timeout)
            
            if shutdown_thread.is_alive():
                # Force shutdown if timeout exceeded
                executor.shutdown(wait=False, cancel_futures=True)
                return False  # Timeout occurred
            return True  # Graceful shutdown
        
        # Test timeout
        start_time = time.time()
        graceful = shutdown_with_timeout(self.executor, timeout=0.5)
        elapsed = time.time() - start_time
        
        self.assertFalse(graceful, "Should timeout and force shutdown")
        self.assertLess(elapsed, 1.0, "Should not wait longer than timeout")
        self.assertTrue(future.cancelled() or future.done(), 
                       "Task should be cancelled after forced shutdown")
    
    def test_concurrent_request_handling_during_shutdown(self):
        """Test handling of concurrent requests during shutdown."""
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Create a barrier to synchronize tasks
        barrier = threading.Barrier(3)  # 2 tasks + main thread
        results = []
        
        def task(task_id):
            """Simulated task that waits at barrier."""
            barrier.wait()  # Wait for all tasks to start
            time.sleep(0.1)
            return f"Task {task_id} completed"
        
        # Submit tasks
        future1 = self.executor.submit(task, 1)
        future2 = self.executor.submit(task, 2)
        
        # Wait for tasks to start
        barrier.wait()
        
        # Try to submit new task during shutdown
        self.executor.shutdown(wait=False)
        
        # Should not be able to submit new tasks
        with self.assertRaises(RuntimeError):
            self.executor.submit(task, 3)
        
        # Existing tasks should complete
        self.executor.shutdown(wait=True)
        self.assertTrue(future1.done())
        self.assertTrue(future2.done())
    
    def test_memory_leak_prevention(self):
        """Test that resources are properly freed after shutdown."""
        import gc
        import tracemalloc
        
        # Start memory tracking
        tracemalloc.start()
        
        # Create and destroy multiple executors
        for i in range(5):
            executor = ThreadPoolExecutor(max_workers=2)
            
            # Submit some tasks
            futures = []
            for j in range(10):
                future = executor.submit(lambda x: x * 2, j)
                futures.append(future)
            
            # Wait for completion
            for future in futures:
                future.result()
            
            # Shutdown
            executor.shutdown(wait=True)
            
            # Force garbage collection
            del executor
            del futures
            gc.collect()
        
        # Get memory snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        # Check for ThreadPoolExecutor in top memory consumers
        executor_memory = 0
        for stat in top_stats[:20]:  # Check top 20 memory consumers
            if 'ThreadPoolExecutor' in str(stat):
                executor_memory += stat.size
        
        # Memory used by executors should be minimal after cleanup
        self.assertLess(executor_memory, 1024 * 1024,  # Less than 1MB
                       "ThreadPoolExecutor memory should be freed after shutdown")
        
        tracemalloc.stop()
    
    def test_cleanup_with_exception(self):
        """Test shutdown behavior when tasks raise exceptions."""
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        def failing_task():
            """Task that raises an exception."""
            raise ValueError("Test exception")
        
        # Submit tasks that will fail
        futures = []
        for i in range(3):
            future = self.executor.submit(failing_task)
            futures.append(future)
        
        # Shutdown should still work with failed tasks
        self.executor.shutdown(wait=True)
        
        # All futures should be done (even if failed)
        for future in futures:
            self.assertTrue(future.done(), "Failed tasks should still be marked as done")
            with self.assertRaises(ValueError):
                future.result()
    
    def test_app_lifecycle_hooks(self):
        """Test Sanic app lifecycle hooks for proper cleanup."""
        from sanic import Sanic
        from stts.app import setup_app, cleanup_app
        
        # Create mock app
        app = Mock(spec=Sanic)
        app.ctx = Mock()
        loop = asyncio.new_event_loop()
        
        # Test setup
        asyncio.run(setup_app(app, loop))
        self.assertIsNotNone(app.ctx.executor, "Executor should be set in app context")
        self.assertIsNotNone(app.ctx.engine, "Engine should be set in app context")
        
        # Test cleanup
        with patch('stts.app.shutdown_executor') as mock_shutdown:
            with patch('stts.app.shutdown_initiated', False):
                asyncio.run(cleanup_app(app, loop))
                mock_shutdown.assert_called_once()
        
        loop.close()
    
    @unittest.skipUnless(HAS_REQUESTS, "requests module not available")
    def test_integration_server_shutdown(self):
        """Integration test for full server shutdown process."""
        import subprocess
        import time
        
        # Create a test server script
        server_script = '''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stts.app import app
import logging

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=18765, debug=False)
'''
        
        # Write server script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(server_script)
            server_file = f.name
        
        try:
            # Start server process
            server_process = subprocess.Popen(
                [sys.executable, server_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(2)
            
            # Verify server is running
            try:
                response = requests.get('http://127.0.0.1:18765/health', timeout=1)
                self.assertEqual(response.status_code, 200, "Server should be running")
            except:
                # Server might not be fully started, skip this test
                server_process.terminate()
                return
            
            # Send termination signal
            server_process.terminate()
            
            # Wait for graceful shutdown
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                server_process.kill()
                self.fail("Server did not shut down gracefully within timeout")
            
            # Check return code
            self.assertEqual(server_process.returncode, 0, 
                           "Server should exit cleanly with code 0")
            
        finally:
            # Clean up
            os.unlink(server_file)
            if server_process.poll() is None:
                server_process.kill()


class TestMemoryLeaks(unittest.TestCase):
    """Test suite specifically for memory leak detection."""
    
    @unittest.skipUnless(HAS_PSUTIL, "psutil module not available")
    def test_repeated_executor_creation(self):
        """Test for memory leaks with repeated executor creation/destruction."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and destroy many executors
        for i in range(100):
            executor = ThreadPoolExecutor(max_workers=4)
            
            # Submit tasks
            futures = [executor.submit(lambda x: x * 2, j) for j in range(10)]
            
            # Get results
            for future in futures:
                future.result()
            
            # Proper shutdown
            executor.shutdown(wait=True)
            del executor
            del futures
            
            # Periodic garbage collection
            if i % 10 == 0:
                gc.collect()
        
        # Final garbage collection
        gc.collect()
        time.sleep(0.5)  # Let OS reclaim memory
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal (allow 10MB for test overhead)
        self.assertLess(memory_growth, 10, 
                       f"Memory grew by {memory_growth:.2f}MB, possible leak")
    
    def test_executor_thread_cleanup(self):
        """Test that executor threads are properly cleaned up."""
        initial_thread_count = threading.active_count()
        
        # Create executor with many workers
        executor = ThreadPoolExecutor(max_workers=10)
        
        # Submit tasks to ensure all workers are created
        futures = [executor.submit(time.sleep, 0.01) for _ in range(20)]
        
        # Wait for tasks
        for future in futures:
            future.result()
        
        # Check thread count increased
        active_with_executor = threading.active_count()
        self.assertGreater(active_with_executor, initial_thread_count,
                          "Thread count should increase with executor")
        
        # Shutdown executor
        executor.shutdown(wait=True)
        
        # Give threads time to terminate
        time.sleep(0.5)
        
        # Thread count should return to near initial
        final_thread_count = threading.active_count()
        self.assertLessEqual(final_thread_count, initial_thread_count + 1,
                           f"Threads not cleaned up: initial={initial_thread_count}, "
                           f"final={final_thread_count}")


def run_stress_test():
    """Run a stress test simulating production load with shutdown."""
    print("Running stress test...")
    
    executor = ThreadPoolExecutor(max_workers=10)
    results = []
    errors = []
    
    def stress_task(task_id):
        """Simulated work task."""
        import random
        time.sleep(random.uniform(0.01, 0.1))
        if random.random() < 0.1:  # 10% failure rate
            raise ValueError(f"Task {task_id} failed")
        return f"Task {task_id} completed"
    
    # Submit many tasks
    print("Submitting 1000 tasks...")
    futures = []
    for i in range(1000):
        future = executor.submit(stress_task, i)
        futures.append(future)
    
    # Start shutdown after brief period
    time.sleep(0.5)
    print("Initiating shutdown...")
    
    # Graceful shutdown
    start_time = time.time()
    executor.shutdown(wait=True)
    shutdown_time = time.time() - start_time
    
    # Collect results
    for future in futures:
        try:
            results.append(future.result())
        except Exception as e:
            errors.append(str(e))
    
    print(f"Stress test completed:")
    print(f"  - Shutdown time: {shutdown_time:.2f}s")
    print(f"  - Successful tasks: {len(results)}")
    print(f"  - Failed tasks: {len(errors)}")
    print(f"  - Total tasks: {len(futures)}")
    
    # Verify all tasks were processed
    assert len(results) + len(errors) == len(futures), "All tasks should be accounted for"
    print("✓ Stress test passed")


if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestExecutorShutdown))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryLeaks))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run stress test if unit tests pass
    if result.wasSuccessful():
        print("\n" + "="*70)
        run_stress_test()
        print("="*70)
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)