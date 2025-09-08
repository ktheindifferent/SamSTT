from os import getenv
import logging
import asyncio
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from sanic import Sanic
from sanic.response import json
from sanic.exceptions import InvalidUsage, RequestTimeout

from .engine import SpeechToTextEngine
from .validators import (
    SecurityMiddleware, 
    validate_audio_file, 
    REQUEST_TIMEOUT,
    MAX_FILE_SIZE
)


# Configure logging
logging.basicConfig(level=getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

MAX_ENGINE_WORKERS = int(getenv('MAX_ENGINE_WORKERS', 2))
SHUTDOWN_TIMEOUT = int(getenv('EXECUTOR_SHUTDOWN_TIMEOUT', 30))

# Initialize the unified STT engine
engine = SpeechToTextEngine()

# Thread pool executor with tracking for graceful shutdown
executor = ThreadPoolExecutor(max_workers=MAX_ENGINE_WORKERS, thread_name_prefix='stt-worker')
shutdown_event = Event()
active_tasks = set()

app = Sanic("stt-service")

# Configure request size limit
app.config.REQUEST_MAX_SIZE = MAX_FILE_SIZE
app.config.REQUEST_TIMEOUT = REQUEST_TIMEOUT
app.config.RESPONSE_TIMEOUT = REQUEST_TIMEOUT


@app.route('/api/v1/stt', methods=['POST'])
async def stt(request):
    """Legacy endpoint for STT - uses default engine"""
    speech = request.files.get('speech')
    if not speech:
        raise InvalidUsage("Missing \"speech\" payload.")
    
    # Validate the audio file
    client_id = SecurityMiddleware.get_client_id(request)
    is_valid, error_msg, metadata = validate_audio_file(
        speech.body,
        filename=speech.name,
        content_type=request.headers.get('Content-Type'),
        client_id=client_id
    )
    
    if not is_valid:
        logger.warning(f"File validation failed for {client_id}: {error_msg}")
        raise InvalidUsage(error_msg)
    
    # Check if specific engine is requested via query parameter or header
    engine_name = request.args.get('engine') or request.headers.get('X-STT-Engine')
    
    from time import perf_counter
    inference_start = perf_counter()
    
    try:
        # Add timeout for transcription with task tracking
        future = app.loop.run_in_executor(
            executor, 
            lambda: engine.transcribe(speech.body, engine=engine_name)
        )
        active_tasks.add(future)
        try:
            result = await asyncio.wait_for(future, timeout=REQUEST_TIMEOUT)
        finally:
            active_tasks.discard(future)
        inference_end = perf_counter() - inference_start
        
        return json({
            'text': result['text'],
            'time': inference_end,
            'engine': result.get('engine'),
            'fallback': result.get('fallback', False)
        })
    except asyncio.TimeoutError:
        logger.error(f"STT request timed out after {REQUEST_TIMEOUT} seconds")
        raise RequestTimeout(f"Request timed out after {REQUEST_TIMEOUT} seconds")
    except Exception as e:
        logger.error(f"STT failed: {e}")
        raise InvalidUsage(f"STT processing failed: {str(e)}")


@app.route('/api/v2/stt/<engine_name>', methods=['POST'])
async def stt_with_engine(request, engine_name):
    """New endpoint for STT with explicit engine selection"""
    speech = request.files.get('speech')
    if not speech:
        raise InvalidUsage("Missing \"speech\" payload.")
    
    # Validate the audio file
    client_id = SecurityMiddleware.get_client_id(request)
    is_valid, error_msg, metadata = validate_audio_file(
        speech.body,
        filename=speech.name,
        content_type=request.headers.get('Content-Type'),
        client_id=client_id
    )
    
    if not is_valid:
        logger.warning(f"File validation failed for {client_id}: {error_msg}")
        raise InvalidUsage(error_msg)
    
    from time import perf_counter
    inference_start = perf_counter()
    
    try:
        # Add timeout for transcription with task tracking
        future = app.loop.run_in_executor(
            executor,
            lambda: engine.transcribe(speech.body, engine=engine_name)
        )
        active_tasks.add(future)
        try:
            result = await asyncio.wait_for(future, timeout=REQUEST_TIMEOUT)
        finally:
            active_tasks.discard(future)
        inference_end = perf_counter() - inference_start
        
        return json({
            'text': result['text'],
            'time': inference_end,
            'engine': result.get('engine'),
            'fallback': result.get('fallback', False)
        })
    except ValueError as e:
        raise InvalidUsage(f"Invalid engine: {str(e)}")
    except asyncio.TimeoutError:
        logger.error(f"STT request timed out after {REQUEST_TIMEOUT} seconds")
        raise RequestTimeout(f"Request timed out after {REQUEST_TIMEOUT} seconds")
    except Exception as e:
        logger.error(f"STT failed with {engine_name}: {e}")
        raise InvalidUsage(f"STT processing failed: {str(e)}")


@app.route('/api/v1/engines', methods=['GET'])
async def list_engines(request):
    """List available STT engines"""
    try:
        available = engine.list_engines()
        all_engines = engine.manager.list_all_engines()
        engine_info = engine.get_engine_info()
        
        return json({
            'available': available,
            'all': all_engines,
            'default': engine.manager.default_engine_name,
            'details': engine_info
        })
    except Exception as e:
        logger.error(f"Failed to list engines: {e}")
        raise InvalidUsage(f"Failed to list engines: {str(e)}")


@app.route('/api/v1/engines/<engine_name>', methods=['GET'])
async def get_engine_info(request, engine_name):
    """Get information about a specific engine"""
    try:
        info = engine.get_engine_info(engine_name)
        return json(info)
    except ValueError as e:
        raise InvalidUsage(str(e))
    except Exception as e:
        logger.error(f"Failed to get engine info: {e}")
        raise InvalidUsage(f"Failed to get engine info: {str(e)}")


@app.route('/health', methods=['GET'])
async def health(request):
    """Health check endpoint with thread pool monitoring"""
    available_engines = engine.list_engines()
    
    # Get thread pool status
    if hasattr(executor, '_threads'):
        active_threads = len([t for t in executor._threads if t and t.is_alive()])
        max_threads = executor._max_workers
    else:
        active_threads = 0
        max_threads = MAX_ENGINE_WORKERS
    
    # Check if we're shutting down
    is_shutting_down = shutdown_event.is_set()
    
    return json({
        'status': 'shutting_down' if is_shutting_down else ('healthy' if available_engines else 'degraded'),
        'engines_available': len(available_engines),
        'engines': available_engines,
        'thread_pool': {
            'active_threads': active_threads,
            'max_threads': max_threads,
            'active_tasks': len(active_tasks),
            'is_shutdown': executor._shutdown,
            'shutting_down': is_shutting_down
        }
    })


@app.middleware('request')
async def security_middleware(request):
    """Middleware for security checks on all requests"""
    # Skip validation for health and info endpoints
    if request.path in ['/health', '/api/v1/engines'] or request.method == 'GET':
        return
    
    # Additional security headers
    request.ctx.client_id = SecurityMiddleware.get_client_id(request)


@app.middleware('response')
async def security_response_middleware(request, response):
    """Add security headers to responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src 'self'"


@app.listener('before_server_start')
async def setup_executor(app, loop):
    """Initialize resources on server start"""
    logger.info(f"Initializing ThreadPoolExecutor with {MAX_ENGINE_WORKERS} workers")
    logger.info(f"Shutdown timeout configured: {SHUTDOWN_TIMEOUT} seconds")
    
    # Log available engines at startup
    available = engine.list_engines()
    logger.info(f"Starting STT service with {len(available)} available engines: {available}")
    logger.info(f"Default engine: {engine.manager.default_engine_name}")
    logger.info(f"Security settings: Max file size: {MAX_FILE_SIZE/1024/1024:.1f}MB, Request timeout: {REQUEST_TIMEOUT}s")


@app.listener('before_server_stop')
async def shutdown_executor(app, loop):
    """Gracefully shutdown ThreadPoolExecutor before server stops"""
    logger.info("Initiating graceful shutdown of ThreadPoolExecutor...")
    shutdown_event.set()
    
    # Wait for active tasks to complete (with timeout)
    if active_tasks:
        logger.info(f"Waiting for {len(active_tasks)} active tasks to complete...")
        start_time = time.time()
        
        while active_tasks and (time.time() - start_time) < SHUTDOWN_TIMEOUT:
            logger.debug(f"Active tasks remaining: {len(active_tasks)}")
            await asyncio.sleep(0.5)
        
        if active_tasks:
            logger.warning(f"Timeout: {len(active_tasks)} tasks still active after {SHUTDOWN_TIMEOUT}s")
            # Cancel remaining tasks
            for task in list(active_tasks):
                if not task.done():
                    task.cancel()
            active_tasks.clear()
    
    # Shutdown the executor
    logger.info("Shutting down ThreadPoolExecutor...")
    try:
        executor.shutdown(wait=True, timeout=SHUTDOWN_TIMEOUT)
        logger.info("ThreadPoolExecutor shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during executor shutdown: {e}")
        # Force shutdown if graceful fails
        executor.shutdown(wait=False)
        logger.warning("Forced ThreadPoolExecutor shutdown")


@app.after_server_stop
async def cleanup_resources(app, loop):
    """Final cleanup after server stops"""
    logger.info("Server stopped, cleanup completed")
    
    # Log final statistics if available
    if hasattr(executor, '_threads'):
        alive_threads = [t for t in executor._threads if t and t.is_alive()]
        if alive_threads:
            logger.warning(f"Warning: {len(alive_threads)} threads still alive after shutdown")
        else:
            logger.info("All worker threads terminated successfully")


def handle_signal(signum, frame):
    """Handle shutdown signals gracefully"""
    sig_name = signal.Signals(signum).name
    logger.info(f"Received signal {sig_name}, initiating graceful shutdown...")
    shutdown_event.set()


# Register signal handlers for graceful shutdown
signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)