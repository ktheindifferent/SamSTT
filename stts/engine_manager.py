from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from .base_engine import BaseSTTEngine
from .engines.deepspeech import DeepSpeechEngine
from .engines.whisper import WhisperEngine
from .engines.coqui import CoquiEngine
from .engines.vosk import VoskEngine
from .engines.silero import SileroEngine
from .engines.wav2vec2 import Wav2Vec2Engine
from .engines.speechbrain import SpeechBrainEngine
from .engines.nemo import NeMoEngine
from .engines.pocketsphinx import PocketSphinxEngine
from .exceptions import (
    STTException,
    EngineNotAvailableError,
    EngineInitializationError,
    ModelNotFoundError,
    TranscriptionError,
    AudioProcessingError,
    ConfigurationError
)


logger = logging.getLogger(__name__)


class STTEngineManager:
    """Manager for multiple STT engine backends"""
    
    # Registry of available engines
    ENGINES = {
        'deepspeech': DeepSpeechEngine,
        'whisper': WhisperEngine,
        'coqui': CoquiEngine,
        'vosk': VoskEngine,
        'silero': SileroEngine,
        'wav2vec2': Wav2Vec2Engine,
        'speechbrain': SpeechBrainEngine,
        'nemo': NeMoEngine,
        'pocketsphinx': PocketSphinxEngine
    }
    
    def __init__(self, default_engine: str = 'whisper', config: Optional[Dict[str, Any]] = None):
        """Initialize the STT Engine Manager
        
        Args:
            default_engine: Name of the default engine to use
            config: Configuration dict with engine-specific settings
        """
        self.config = config or {}
        self.engines: Dict[str, BaseSTTEngine] = {}
        self.default_engine_name = default_engine
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all configured engines"""
        # Try to initialize the default engine first
        if self.default_engine_name in self.ENGINES:
            engine_config = self.config.get(self.default_engine_name, {})
            try:
                engine = self.ENGINES[self.default_engine_name](engine_config)
                if engine.is_available:
                    self.engines[self.default_engine_name] = engine
                    logger.info(f"Initialized {self.default_engine_name} as default engine")
                else:
                    logger.warning(f"Default engine {self.default_engine_name} is not available")
            except EngineNotAvailableError as e:
                logger.warning(f"Default engine {self.default_engine_name} not available: {e.message}")
            except ModelNotFoundError as e:
                logger.error(f"Model not found for default engine {self.default_engine_name}: {e.message}")
            except EngineInitializationError as e:
                logger.error(f"Failed to initialize default engine {self.default_engine_name}: {e.message}")
            except Exception as e:
                logger.error(f"Unexpected error initializing default engine {self.default_engine_name}: {e}")
        
        # Try to initialize other configured engines
        for engine_name, engine_class in self.ENGINES.items():
            if engine_name in self.engines:
                continue  # Already initialized
            
            if engine_name in self.config or self.config.get('initialize_all', False):
                engine_config = self.config.get(engine_name, {})
                try:
                    engine = engine_class(engine_config)
                    if engine.is_available:
                        self.engines[engine_name] = engine
                        logger.info(f"Initialized {engine_name} engine")
                except EngineNotAvailableError as e:
                    logger.debug(f"{engine_name} not available: {e.message}")
                except ModelNotFoundError as e:
                    logger.debug(f"Model not found for {engine_name}: {e.message}")
                except (EngineInitializationError, ConfigurationError) as e:
                    logger.debug(f"Could not initialize {engine_name}: {e.message}")
                except Exception as e:
                    logger.debug(f"Unexpected error with {engine_name}: {e}")
    
    def add_engine(self, name: str, engine: BaseSTTEngine):
        """Add a custom engine instance
        
        Args:
            name: Name for the engine
            engine: Engine instance
        """
        self.engines[name] = engine
        logger.info(f"Added custom engine: {name}")
    
    def get_engine(self, name: Optional[str] = None) -> BaseSTTEngine:
        """Get a specific engine or the default engine
        
        Args:
            name: Name of the engine to get. If None, returns default engine
            
        Returns:
            The requested engine
            
        Raises:
            EngineNotAvailableError: If the requested engine is not available
            ConfigurationError: If the engine name is unknown
        """
        if name is None:
            name = self.default_engine_name
        
        if name not in self.engines:
            # Try to initialize on-demand
            if name in self.ENGINES:
                engine_config = self.config.get(name, {})
                try:
                    engine = self.ENGINES[name](engine_config)
                    if engine.is_available:
                        self.engines[name] = engine
                        logger.info(f"Initialized {name} engine on-demand")
                    else:
                        raise EngineNotAvailableError(
                            f"Engine {name} is not available on this system",
                            engine=name
                        )
                except STTException:
                    # Re-raise our custom exceptions as-is
                    raise
                except Exception as e:
                    raise EngineInitializationError(
                        f"Failed to initialize engine {name}: {str(e)}",
                        engine=name,
                        original_error=e
                    )
            else:
                raise ConfigurationError(
                    f"Unknown engine: {name}. Available engines: {', '.join(self.ENGINES.keys())}",
                    engine=name
                )
        
        return self.engines[name]
    
    def list_available_engines(self) -> List[str]:
        """List all available and initialized engines"""
        return list(self.engines.keys())
    
    def list_all_engines(self) -> List[str]:
        """List all registered engines (including non-initialized)"""
        return list(self.ENGINES.keys())
    
    def transcribe(self, audio: bytes, engine: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio using specified or default engine
        
        Args:
            audio: Audio bytes to transcribe
            engine: Name of engine to use (optional)
            
        Returns:
            Dict with transcription result and metadata
        """
        primary_engine = None
        primary_error = None
        
        try:
            primary_engine = self.get_engine(engine)
            text = primary_engine.transcribe(audio)
            return {
                'text': text,
                'engine': primary_engine.name,
                'success': True
            }
        except (AudioProcessingError, TranscriptionError) as e:
            # These are errors we might be able to recover from with fallback
            primary_error = e
            logger.error(f"Transcription failed with {primary_engine.name if primary_engine else engine}: {e.message}")
        except (EngineNotAvailableError, ModelNotFoundError, ConfigurationError) as e:
            # These errors are not recoverable for this engine
            logger.error(f"Engine error: {e.message}")
            primary_error = e
            # Try fallback immediately
        except Exception as e:
            # Unexpected error
            primary_error = e
            logger.error(f"Unexpected transcription error with {primary_engine.name if primary_engine else engine}: {e}")
        
        # Try fallback engines if available
        fallback_errors = []
        for fallback_name, fallback_engine in self.engines.items():
            if fallback_name != (engine or self.default_engine_name):
                try:
                    logger.info(f"Attempting fallback to {fallback_name}")
                    text = fallback_engine.transcribe(audio)
                    logger.info(f"Fallback to {fallback_name} succeeded")
                    return {
                        'text': text,
                        'engine': fallback_engine.name,
                        'success': True,
                        'fallback': True,
                        'original_error': str(primary_error) if primary_error else None
                    }
                except AudioProcessingError as e:
                    # Audio processing errors likely affect all engines
                    logger.debug(f"Fallback {fallback_name} had audio processing error: {e.message}")
                    fallback_errors.append(f"{fallback_name}: {e.message}")
                    # Don't try more engines for audio processing errors
                    break
                except Exception as e:
                    error_msg = e.message if isinstance(e, STTException) else str(e)
                    logger.debug(f"Fallback {fallback_name} failed: {error_msg}")
                    fallback_errors.append(f"{fallback_name}: {error_msg}")
        
        # All engines failed - raise the most informative error
        if isinstance(primary_error, AudioProcessingError):
            # If it's an audio processing error, it's likely the audio itself is bad
            raise primary_error
        elif isinstance(primary_error, STTException):
            # Raise the original custom exception with additional context
            raise TranscriptionError(
                f"All STT engines failed. Primary error: {primary_error.message}. " +
                (f"Fallback errors: {'; '.join(fallback_errors)}" if fallback_errors else "No fallback engines available."),
                engine="multiple",
                original_error=primary_error
            )
        else:
            # Raise a generic transcription error
            raise TranscriptionError(
                f"All STT engines failed. Last error: {str(primary_error)}. " +
                (f"Fallback errors: {'; '.join(fallback_errors)}" if fallback_errors else "No fallback engines available."),
                engine="multiple",
                original_error=primary_error
            )
    
    def get_engine_info(self, engine_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about an engine
        
        Args:
            engine_name: Name of engine to get info for
            
        Returns:
            Dict with engine information
            
        Raises:
            ConfigurationError: If the engine name is unknown
        """
        if engine_name:
            if engine_name in self.engines:
                engine = self.engines[engine_name]
                return {
                    'name': engine.name,
                    'available': engine.is_available,
                    'initialized': True,
                    'config': engine.config
                }
            elif engine_name in self.ENGINES:
                return {
                    'name': engine_name,
                    'available': False,
                    'initialized': False,
                    'config': self.config.get(engine_name, {})
                }
            else:
                raise ConfigurationError(
                    f"Unknown engine: {engine_name}. Available engines: {', '.join(self.ENGINES.keys())}",
                    engine=engine_name
                )
        else:
            # Return info for all engines
            info = {}
            for name in self.ENGINES:
                if name in self.engines:
                    engine = self.engines[name]
                    info[name] = {
                        'available': engine.is_available,
                        'initialized': True,
                        'config': engine.config
                    }
                else:
                    info[name] = {
                        'available': False,
                        'initialized': False,
                        'config': self.config.get(name, {})
                    }
            return info