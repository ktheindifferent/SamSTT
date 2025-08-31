from os import getenv
import logging
import asyncio
import atexit
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
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
SHUTDOWN_TIMEOUT = float(getenv('SHUTDOWN_TIMEOUT', 10.0))

# Initialize the unified STT engine
engine = SpeechToTextEngine()
executor = ThreadPoolExecutor(max_workers=MAX_ENGINE_WORKERS)

app = Sanic("stt-service")

# Track shutdown state
shutdown_initiated = False


def shutdown_executor(wait_timeout=SHUTDOWN_TIMEOUT):
    """Gracefully shutdown the ThreadPoolExecutor."""
    global shutdown_initiated
    if shutdown_initiated:
        logger.debug("Shutdown already initiated, skipping duplicate call")
        return
    
    shutdown_initiated = True
    logger.info("Initiating ThreadPoolExecutor shutdown...")
    
    try:
        # Shutdown the executor, waiting for pending tasks
        executor.shutdown(wait=True, cancel_futures=False)
        logger.info("ThreadPoolExecutor shutdown completed gracefully")
    except Exception as e:
        logger.error(f"Error during executor shutdown: {e}")
        # Force shutdown if graceful shutdown fails
        try:
            executor.shutdown(wait=False, cancel_futures=True)
            logger.warning("ThreadPoolExecutor forced shutdown completed")
        except Exception as force_error:
            logger.error(f"Force shutdown also failed: {force_error}")


def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_executor()
    if app.is_running:
        app.stop()
    sys.exit(0)


# Register shutdown handlers
atexit.register(shutdown_executor)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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
        # Add timeout for transcription
        result = await asyncio.wait_for(
            app.loop.run_in_executor(
                executor, 
                lambda: engine.transcribe(speech.body, engine=engine_name)
            ),
            timeout=REQUEST_TIMEOUT
        )
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
        # Add timeout for transcription
        result = await asyncio.wait_for(
            app.loop.run_in_executor(
                executor,
                lambda: engine.transcribe(speech.body, engine=engine_name)
            ),
            timeout=REQUEST_TIMEOUT
        )
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
    """Health check endpoint"""
    available_engines = engine.list_engines()
    return json({
        'status': 'healthy' if available_engines else 'degraded',
        'engines_available': len(available_engines),
        'engines': available_engines
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
async def setup_app(app, loop):
    """Initialize application resources before server starts."""
    logger.info("Setting up application resources...")
    app.ctx.executor = executor
    app.ctx.engine = engine
    logger.info(f"ThreadPoolExecutor initialized with {MAX_ENGINE_WORKERS} workers")


@app.listener('after_server_stop')
async def cleanup_app(app, loop):
    """Clean up application resources after server stops."""
    logger.info("Cleaning up application resources...")
    
    # Wait for any pending tasks in the event loop
    pending = asyncio.all_tasks(loop)
    if pending:
        logger.info(f"Waiting for {len(pending)} pending tasks to complete...")
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=SHUTDOWN_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for pending tasks after {SHUTDOWN_TIMEOUT}s")
            for task in pending:
                task.cancel()
    
    # Shutdown the executor if not already done
    if not shutdown_initiated:
        shutdown_executor()
    
    # Clean up engine resources
    if hasattr(app.ctx, 'engine'):
        try:
            # Cleanup any engine resources
            if hasattr(app.ctx.engine, 'cleanup'):
                app.ctx.engine.cleanup()
            logger.info("Engine resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up engine resources: {e}")
    
    logger.info("Application cleanup completed")
    

if __name__ == '__main__':
    # Log available engines at startup
    available = engine.list_engines()
    logger.info(f"Starting STT service with {len(available)} available engines: {available}")
    logger.info(f"Default engine: {engine.manager.default_engine_name}")
    logger.info(f"Security settings: Max file size: {MAX_FILE_SIZE/1024/1024:.1f}MB, Request timeout: {REQUEST_TIMEOUT}s")
    
    app.run(host='0.0.0.0', port=8000)