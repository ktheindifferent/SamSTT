import wave
from io import BytesIO
from pathlib import Path
import os
from typing import Optional, Dict, Any
import logging

import ffmpeg
import numpy as np
from .engine_manager import STTEngineManager
from .config_manager import get_config_manager

logger = logging.getLogger(__name__)


class SpeechToTextEngine:
    """Legacy wrapper for backward compatibility with unified STT engine system"""
    
    def __init__(self, engine_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None, config_file: Optional[str] = None):
        """Initialize the STT engine
        
        Args:
            engine_name: Name of the engine to use (deepspeech, whisper, coqui, vosk, silero, wav2vec2)
            config: Configuration for the engines
            config_file: Path to JSON configuration file
        """
        # Use ConfigManager for proper file handle management
        config_manager = get_config_manager()
        
        # Load configuration with proper resource management
        loaded_config, engine_name = config_manager.load_config(
            config=config,
            config_file=config_file,
            default_engine=engine_name
        )
        
        # If no config was loaded, build default from environment
        if not loaded_config:
            loaded_config = config_manager.build_default_config()
        
        # Initialize the engine manager
        self.manager = STTEngineManager(default_engine=engine_name, config=loaded_config)
        
        # For backward compatibility, check if old model exists
        self._setup_legacy_support()
    
    
    def _setup_legacy_support(self):
        """Setup support for legacy code expecting 'model' attribute"""
        try:
            # Try to get the default engine
            engine = self.manager.get_engine()
            if hasattr(engine, 'model'):
                self.model = engine.model
        except ValueError as e:
            # Engine not available or unknown engine
            logger.debug(f"Could not setup legacy model attribute: {e}")
        except AttributeError as e:
            # Engine doesn't have expected attributes
            logger.debug(f"Engine doesn't have model attribute for legacy support: {e}")
        except Exception as e:
            # Catch any other unexpected exceptions during legacy setup
            logger.debug(f"Unexpected error during legacy support setup: {type(e).__name__}: {e}")
    
    def normalize_audio(self, audio):
        """Normalize audio to 16kHz mono WAV format (legacy method)"""
        out, err = ffmpeg.input('pipe:0') \
            .output('pipe:1', f='WAV', acodec='pcm_s16le', ac=1, ar='16k', loglevel='error', hide_banner=None) \
            .run(input=audio, capture_stdout=True, capture_stderr=True)
        if err:
            raise Exception(err)
        return out
    
    def run(self, audio, engine: Optional[str] = None):
        """Run STT on audio (legacy method)
        
        Args:
            audio: Audio bytes to transcribe
            engine: Optional engine name to use
            
        Returns:
            Transcribed text
        """
        result = self.manager.transcribe(audio, engine=engine)
        return result['text']
    
    def transcribe(self, audio, engine: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio with metadata
        
        Args:
            audio: Audio bytes to transcribe
            engine: Optional engine name to use
            
        Returns:
            Dict with transcription result and metadata
        """
        return self.manager.transcribe(audio, engine=engine)
    
    def list_engines(self):
        """List available STT engines"""
        return self.manager.list_available_engines()
    
    def get_engine_info(self, engine_name: Optional[str] = None):
        """Get information about engines"""
        return self.manager.get_engine_info(engine_name)