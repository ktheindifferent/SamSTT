#!/usr/bin/env python3
"""
Simple test for ThreadPoolExecutor shutdown behavior
Tests the core shutdown functionality without external dependencies
"""

import os
import sys
import time
import signal
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
from unittest.mock import Mock, patch
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, '/root/repo')

def test_shutdown_handler():
    """Test the shutdown handler implementation"""
    logger.info("\n=== Testing Shutdown Handler Implementation ===")
    
    # Import after adding to path
    from stts.app import (
        executor, shutdown_event, active_tasks,
        handle_signal, SHUTDOWN_TIMEOUT
    )
    
    # Test 1: Verify executor is properly initialized
    logger.info("Test 1: Executor initialization")
    assert executor is not None, "Executor not initialized"
    assert isinstance(executor, ThreadPoolExecutor), "Wrong executor type"
    assert not executor._shutdown, "Executor should not be shutdown initially"
    logger.info("✓ Executor properly initialized")
    
    # Test 2: Verify shutdown event
    logger.info("\nTest 2: Shutdown event")
    shutdown_event.clear()  # Reset
    assert not shutdown_event.is_set(), "Shutdown event should not be set initially"
    
    # Simulate signal
    handle_signal(signal.SIGTERM, None)
    assert shutdown_event.is_set(), "Shutdown event should be set after signal"
    logger.info("✓ Shutdown event works correctly")
    
    # Test 3: Active tasks tracking
    logger.info("\nTest 3: Active tasks tracking")
    active_tasks.clear()  # Reset
    
    # Create a mock future
    mock_future = Mock(spec=Future)
    mock_future.done.return_value = False
    
    active_tasks.add(mock_future)
    assert len(active_tasks) == 1, "Task not added to active_tasks"
    
    active_tasks.discard(mock_future)
    assert len(active_tasks) == 0, "Task not removed from active_tasks"
    logger.info("✓ Active tasks tracking works")
    
    # Test 4: Shutdown timeout configuration
    logger.info("\nTest 4: Shutdown timeout")
    assert SHUTDOWN_TIMEOUT > 0, "Shutdown timeout must be positive"
    logger.info(f"✓ Shutdown timeout configured: {SHUTDOWN_TIMEOUT} seconds")
    
    return True


def test_lifecycle_functions():
    """Test Sanic lifecycle functions"""
    logger.info("\n=== Testing Lifecycle Functions ===")
    
    from stts.app import app
    import inspect
    
    # Check for lifecycle handlers
    lifecycle_handlers = {
        'before_server_start': [],
        'before_server_stop': [],
        'after_server_stop': []
    }
    
    # Find registered listeners
    for event_type in lifecycle_handlers.keys():
        if hasattr(app, 'listeners') and event_type in app.listeners:
            handlers = app.listeners[event_type]
            lifecycle_handlers[event_type] = [h.__name__ for h in handlers]
    
    logger.info("Registered lifecycle handlers:")
    for event_type, handlers in lifecycle_handlers.items():
        logger.info(f"  {event_type}: {handlers}")
    
    # Verify critical handlers are present
    assert 'setup_executor' in lifecycle_handlers['before_server_start'], \
        "setup_executor not registered"
    assert 'shutdown_executor' in lifecycle_handlers['before_server_stop'], \
        "shutdown_executor not registered"
    assert 'cleanup_resources' in lifecycle_handlers['after_server_stop'], \
        "cleanup_resources not registered"
    
    logger.info("✓ All lifecycle handlers registered correctly")
    
    return True


def test_health_endpoint_modifications():
    """Test that health endpoint includes thread pool info"""
    logger.info("\n=== Testing Health Endpoint Modifications ===")
    
    from stts.app import health
    import inspect
    
    # Check health function source
    source = inspect.getsource(health)
    
    # Verify thread pool monitoring is included
    assert 'thread_pool' in source, "Health endpoint doesn't include thread pool info"
    assert 'active_threads' in source, "Missing active_threads monitoring"
    assert 'active_tasks' in source, "Missing active_tasks monitoring"
    assert 'shutting_down' in source, "Missing shutdown status"
    
    logger.info("✓ Health endpoint includes thread pool monitoring")
    
    return True


def test_task_tracking_in_endpoints():
    """Test that STT endpoints track active tasks"""
    logger.info("\n=== Testing Task Tracking in Endpoints ===")
    
    from stts.app import stt, stt_with_engine
    import inspect
    
    # Check both STT endpoints
    for endpoint_name, endpoint_func in [('stt', stt), ('stt_with_engine', stt_with_engine)]:
        logger.info(f"Checking {endpoint_name} endpoint...")
        source = inspect.getsource(endpoint_func)
        
        # Verify task tracking
        assert 'active_tasks.add' in source, f"{endpoint_name} doesn't add tasks"
        assert 'active_tasks.discard' in source, f"{endpoint_name} doesn't remove tasks"
        assert 'finally:' in source, f"{endpoint_name} doesn't use finally block"
        
        logger.info(f"  ✓ {endpoint_name} tracks active tasks correctly")
    
    return True


def test_executor_shutdown_with_mock():
    """Test executor shutdown behavior with mocking"""
    logger.info("\n=== Testing Executor Shutdown with Mock ===")
    
    from stts.app import shutdown_executor, executor, active_tasks, shutdown_event
    import asyncio
    
    # Create mock app and loop
    mock_app = Mock()
    mock_loop = Mock()
    
    # Reset state
    shutdown_event.clear()
    active_tasks.clear()
    
    # Add some mock active tasks
    for i in range(3):
        mock_task = Mock()
        mock_task.done.return_value = False
        mock_task.cancel = Mock()
        active_tasks.add(mock_task)
    
    logger.info(f"Created {len(active_tasks)} mock active tasks")
    
    # Run shutdown (with very short timeout for testing)
    with patch('stts.app.SHUTDOWN_TIMEOUT', 0.1):
        # Run the async shutdown function
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(shutdown_executor(mock_app, mock_loop))
        except Exception as e:
            logger.warning(f"Shutdown raised exception (expected): {e}")
        finally:
            loop.close()
    
    # Verify shutdown was attempted
    assert shutdown_event.is_set(), "Shutdown event should be set"
    
    # Verify tasks were cancelled (due to timeout)
    for task in list(active_tasks):
        if hasattr(task, 'cancel'):
            task.cancel.assert_called()
    
    logger.info("✓ Executor shutdown handles active tasks correctly")
    
    return True


def test_thread_safety():
    """Test thread safety of shutdown mechanism"""
    logger.info("\n=== Testing Thread Safety ===")
    
    from stts.app import active_tasks, shutdown_event
    import threading
    
    # Reset state
    active_tasks.clear()
    errors = []
    
    def add_remove_tasks():
        """Worker that adds and removes tasks"""
        try:
            for i in range(100):
                task = Mock()
                active_tasks.add(task)
                time.sleep(0.001)
                active_tasks.discard(task)
        except Exception as e:
            errors.append(e)
    
    # Run multiple threads concurrently
    threads = []
    for i in range(5):
        t = threading.Thread(target=add_remove_tasks)
        threads.append(t)
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    # Check for errors
    if errors:
        logger.error(f"Thread safety errors: {errors}")
        return False
    
    logger.info("✓ Thread-safe operations verified")
    
    return True


def run_all_tests():
    """Run all tests"""
    tests = [
        ("Shutdown Handler", test_shutdown_handler),
        ("Lifecycle Functions", test_lifecycle_functions),
        ("Health Endpoint", test_health_endpoint_modifications),
        ("Task Tracking", test_task_tracking_in_endpoints),
        ("Executor Shutdown Mock", test_executor_shutdown_with_mock),
        ("Thread Safety", test_thread_safety)
    ]
    
    passed = 0
    failed = 0
    
    logger.info("\n" + "="*60)
    logger.info("ThreadPoolExecutor Shutdown Tests")
    logger.info("="*60)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name}: PASSED\n")
            else:
                failed += 1
                logger.error(f"✗ {test_name}: FAILED\n")
        except Exception as e:
            failed += 1
            logger.error(f"✗ {test_name}: FAILED - {e}\n")
            import traceback
            traceback.print_exc()
    
    logger.info("="*60)
    logger.info(f"Results: {passed}/{len(tests)} passed")
    if failed == 0:
        logger.info("SUCCESS: All tests passed!")
    else:
        logger.error(f"FAILURE: {failed} tests failed")
    logger.info("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)