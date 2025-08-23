from os import getenv
import logging
from concurrent.futures import ThreadPoolExecutor
from sanic import Sanic
from sanic.response import json
from sanic.exceptions import InvalidUsage

from .engine import SpeechToTextEngine
from .exceptions import (
    STTException,
    EngineNotAvailableError,
    ModelNotFoundError,
    AudioProcessingError,
    InvalidAudioError,
    TranscriptionError,
    ConfigurationError,
    InsufficientResourcesError,
    EngineTimeoutError
)


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
        
        response_data = {
            'text': result['text'],
            'time': inference_end,
            'engine': result.get('engine'),
            'fallback': result.get('fallback', False)
        }
        
        # Include original error info if fallback was used
        if result.get('original_error'):
            response_data['original_error'] = result['original_error']
        
        return json(response_data)
    except InvalidAudioError as e:
        logger.error(f"Invalid audio data: {e.message}")
        raise InvalidUsage(f"Invalid audio data: {e.message}", status_code=400)
    except AudioProcessingError as e:
        logger.error(f"Audio processing failed: {e.message}")
        raise InvalidUsage(f"Audio processing failed: {e.message}", status_code=400)
    except EngineNotAvailableError as e:
        logger.error(f"Engine not available: {e.message}")
        raise InvalidUsage(f"Engine not available: {e.message}", status_code=503)
    except ModelNotFoundError as e:
        logger.error(f"Model not found: {e.message}")
        raise InvalidUsage(f"Model not found: {e.message}", status_code=503)
    except InsufficientResourcesError as e:
        logger.error(f"Insufficient resources: {e.message}")
        raise InvalidUsage(f"Insufficient resources: {e.message}", status_code=503)
    except EngineTimeoutError as e:
        logger.error(f"Transcription timeout: {e.message}")
        raise InvalidUsage(f"Transcription timeout: {e.message}", status_code=504)
    except TranscriptionError as e:
        logger.error(f"Transcription failed: {e.message}")
        raise InvalidUsage(f"Transcription failed: {e.message}", status_code=500)
    except STTException as e:
        logger.error(f"STT error: {e.message}")
        raise InvalidUsage(f"STT error: {e.message}", status_code=500)
    except Exception as e:
        logger.error(f"Unexpected STT error: {e}")
        raise InvalidUsage(f"STT processing failed: {str(e)}", status_code=500)


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
        
        response_data = {
            'text': result['text'],
            'time': inference_end,
            'engine': result.get('engine'),
            'fallback': result.get('fallback', False)
        }
        
        # Include original error info if fallback was used
        if result.get('original_error'):
            response_data['original_error'] = result['original_error']
        
        return json(response_data)
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e.message}")
        raise InvalidUsage(f"Configuration error: {e.message}", status_code=400)
    except InvalidAudioError as e:
        logger.error(f"Invalid audio data: {e.message}")
        raise InvalidUsage(f"Invalid audio data: {e.message}", status_code=400)
    except AudioProcessingError as e:
        logger.error(f"Audio processing failed: {e.message}")
        raise InvalidUsage(f"Audio processing failed: {e.message}", status_code=400)
    except EngineNotAvailableError as e:
        logger.error(f"Engine {engine_name} not available: {e.message}")
        raise InvalidUsage(f"Engine {engine_name} not available: {e.message}", status_code=404)
    except ModelNotFoundError as e:
        logger.error(f"Model not found for {engine_name}: {e.message}")
        raise InvalidUsage(f"Model not found: {e.message}", status_code=503)
    except InsufficientResourcesError as e:
        logger.error(f"Insufficient resources for {engine_name}: {e.message}")
        raise InvalidUsage(f"Insufficient resources: {e.message}", status_code=503)
    except EngineTimeoutError as e:
        logger.error(f"Transcription timeout with {engine_name}: {e.message}")
        raise InvalidUsage(f"Transcription timeout: {e.message}", status_code=504)
    except TranscriptionError as e:
        logger.error(f"Transcription failed with {engine_name}: {e.message}")
        raise InvalidUsage(f"Transcription failed: {e.message}", status_code=500)
    except STTException as e:
        logger.error(f"STT error with {engine_name}: {e.message}")
        raise InvalidUsage(f"STT error: {e.message}", status_code=500)
    except Exception as e:
        logger.error(f"Unexpected error with {engine_name}: {e}")
        raise InvalidUsage(f"STT processing failed: {str(e)}", status_code=500)


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
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e.message}")
        raise InvalidUsage(e.message, status_code=404)
    except STTException as e:
        logger.error(f"Error getting engine info: {e.message}")
        raise InvalidUsage(e.message, status_code=500)
    except Exception as e:
        logger.error(f"Failed to get engine info: {e}")
        raise InvalidUsage(f"Failed to get engine info: {str(e)}", status_code=500)


@app.route('/health', methods=['GET'])
async def health(request):
    """Health check endpoint"""
    try:
        available_engines = engine.list_engines()
        status = 'healthy' if available_engines else 'degraded'
        
        # Get detailed engine status if verbose flag is set
        verbose = request.args.get('verbose', '').lower() == 'true'
        response_data = {
            'status': status,
            'engines_available': len(available_engines),
            'engines': available_engines
        }
        
        if verbose:
            try:
                response_data['engine_details'] = engine.get_engine_info()
            except Exception as e:
                response_data['engine_details_error'] = str(e)
        
        return json(response_data)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return json({
            'status': 'unhealthy',
            'error': str(e)
        }, status=503)


if __name__ == '__main__':
    # Log available engines at startup
    available = engine.list_engines()
    logger.info(f"Starting STT service with {len(available)} available engines: {available}")
    logger.info(f"Default engine: {engine.manager.default_engine_name}")
    
    app.run(host='0.0.0.0', port=8000)