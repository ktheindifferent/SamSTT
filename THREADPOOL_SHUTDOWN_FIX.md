# ThreadPoolExecutor Shutdown Handler Implementation

## Problem Addressed

The global ThreadPoolExecutor in `stts/app.py` was created without a proper shutdown handler, causing thread leaks when the application restarts or shuts down. This led to resource exhaustion over time in production deployments.

## Solution Implemented

### 1. Lifecycle Management
- Integrated with Sanic's lifecycle events:
  - `@app.before_server_start`: Initialize and log configuration
  - `@app.before_server_stop`: Graceful shutdown with task waiting
  - `@app.after_server_stop`: Final cleanup and verification

### 2. Active Task Tracking
- Added `active_tasks` set to track all in-flight requests
- Each STT request adds itself to the set when starting
- Removes itself in a `finally` block ensuring cleanup even on errors
- Shutdown handler waits for active tasks to complete

### 3. Graceful Shutdown Process
```python
1. Set shutdown_event to signal shutdown state
2. Wait for active tasks to complete (up to SHUTDOWN_TIMEOUT)
3. Cancel remaining tasks if timeout exceeded
4. Call executor.shutdown(wait=True, timeout=SHUTDOWN_TIMEOUT)
5. Log shutdown status and any remaining threads
```

### 4. Signal Handling
- Handles SIGTERM and SIGINT for graceful shutdown
- Sets shutdown event when signals received
- Allows clean termination from Docker, Kubernetes, etc.

### 5. Health Monitoring
Enhanced `/health` endpoint with thread pool metrics:
```json
{
  "thread_pool": {
    "active_threads": 2,
    "max_threads": 4,
    "active_tasks": 1,
    "is_shutdown": false,
    "shutting_down": false
  }
}
```

## Configuration

### Environment Variables
- `MAX_ENGINE_WORKERS`: Number of worker threads (default: 2)
- `EXECUTOR_SHUTDOWN_TIMEOUT`: Shutdown timeout in seconds (default: 30)

### Recommended Settings
```bash
# Production
export MAX_ENGINE_WORKERS=4
export EXECUTOR_SHUTDOWN_TIMEOUT=30

# Development/Testing
export MAX_ENGINE_WORKERS=2
export EXECUTOR_SHUTDOWN_TIMEOUT=10
```

## Testing

### Static Analysis Test
```bash
python3 test_shutdown_static.py
```
Verifies all implementation requirements without running the server.

### Integration Tests
```bash
python3 test_threadpool_shutdown.py  # Requires 'requests' module
python3 test_shutdown_simple.py      # Unit tests with mocking
```

### Manual Testing
1. Start the service
2. Make concurrent STT requests
3. Send SIGTERM while requests are active
4. Verify graceful shutdown in logs

## Benefits

1. **No Thread Leaks**: All threads properly cleaned up on shutdown
2. **Graceful Handling**: Active requests complete before shutdown
3. **Timeout Protection**: Prevents hanging on shutdown
4. **Monitoring**: Health endpoint shows thread pool status
5. **Signal Safety**: Responds to standard Unix signals
6. **Container Ready**: Works with Docker/Kubernetes lifecycle

## Deployment Considerations

### Docker
```dockerfile
# Ensure proper signal handling
STOPSIGNAL SIGTERM

# Set appropriate timeout
HEALTHCHECK --timeout=5s CMD curl -f http://localhost:8000/health
```

### Kubernetes
```yaml
spec:
  containers:
  - name: stt-service
    env:
    - name: EXECUTOR_SHUTDOWN_TIMEOUT
      value: "30"
    lifecycle:
      preStop:
        exec:
          command: ["/bin/sh", "-c", "sleep 5"]
    terminationGracePeriodSeconds: 40
```

### Docker Compose
```yaml
services:
  stt:
    stop_grace_period: 35s
    environment:
      - EXECUTOR_SHUTDOWN_TIMEOUT=30
```

## Monitoring

### Key Metrics to Track
1. `thread_pool.active_threads`: Should not exceed max_threads
2. `thread_pool.active_tasks`: Should return to 0 between requests
3. `thread_pool.shutting_down`: Should only be true during shutdown
4. Shutdown duration in logs: Should be < SHUTDOWN_TIMEOUT

### Log Messages
```
INFO: Initializing ThreadPoolExecutor with 4 workers
INFO: Initiating graceful shutdown of ThreadPoolExecutor...
INFO: Waiting for 2 active tasks to complete...
INFO: Shutting down ThreadPoolExecutor...
INFO: ThreadPoolExecutor shutdown completed successfully
INFO: All worker threads terminated successfully
```

### Warning Signs
```
WARNING: Timeout: 3 tasks still active after 30s
WARNING: Forced ThreadPoolExecutor shutdown
WARNING: 2 threads still alive after shutdown
```

## Rollback Plan

If issues arise, the changes can be reverted by:
1. Checking out previous commit
2. Removing lifecycle handlers
3. Removing signal handlers
4. Simplifying health endpoint

However, this would reintroduce the thread leak issue.

## Future Enhancements

1. **Metrics Export**: Prometheus metrics for thread pool
2. **Dynamic Scaling**: Adjust worker count based on load
3. **Request Priority**: Priority queue for important requests
4. **Circuit Breaker**: Prevent overload during high traffic
5. **Distributed Tracing**: Track requests across thread boundaries