from os import getenv
import logging
from concurrent.futures import ThreadPoolExecutor
from sanic import Sanic
from sanic.response import json
from sanic.exceptions import InvalidUsage

from .engine import SpeechToTextEngine


# Configure logging
logging.basicConfig(level=getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

MAX_ENGINE_WORKERS = int(getenv('MAX_ENGINE_WORKERS', 2))

# Initialize the unified STT engine
engine = SpeechToTextEngine()
executor = ThreadPoolExecutor(max_workers=MAX_ENGINE_WORKERS)

app = Sanic("stt-service")


@app.route('/api/v1/stt', methods=['POST'])
async def stt(request):
    """Legacy endpoint for STT - uses default engine"""
    speech = request.files.get('speech')
    if not speech:
        raise InvalidUsage("Missing \"speech\" payload.")
    
    # Check if specific engine is requested via query parameter or header
    engine_name = request.args.get('engine') or request.headers.get('X-STT-Engine')
    
    from time import perf_counter
    inference_start = perf_counter()
    
    try:
        result = await app.loop.run_in_executor(
            executor, 
            lambda: engine.transcribe(speech.body, engine=engine_name)
        )
        inference_end = perf_counter() - inference_start
        
        return json({
            'text': result['text'],
            'time': inference_end,
            'engine': result.get('engine'),
            'fallback': result.get('fallback', False)
        })
    except Exception as e:
        logger.error(f"STT failed: {e}")
        raise InvalidUsage(f"STT processing failed: {str(e)}")


@app.route('/api/v2/stt/<engine_name>', methods=['POST'])
async def stt_with_engine(request, engine_name):
    """New endpoint for STT with explicit engine selection"""
    speech = request.files.get('speech')
    if not speech:
        raise InvalidUsage("Missing \"speech\" payload.")
    
    from time import perf_counter
    inference_start = perf_counter()
    
    try:
        result = await app.loop.run_in_executor(
            executor,
            lambda: engine.transcribe(speech.body, engine=engine_name)
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


if __name__ == '__main__':
    # Log available engines at startup
    available = engine.list_engines()
    logger.info(f"Starting STT service with {len(available)} available engines: {available}")
    logger.info(f"Default engine: {engine.manager.default_engine_name}")
    
    app.run(host='0.0.0.0', port=8000)