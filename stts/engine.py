import wave
from io import BytesIO
from pathlib import Path
import os
from typing import Optional, Dict, Any
import logging

import ffmpeg
import numpy as np
from .engine_manager import STTEngineManager

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
        # Get engine preference from environment if not specified
        if engine_name is None:
            engine_name = os.getenv('STT_ENGINE', 'deepspeech')
        
        # Load configuration
        if config is None:
            if config_file and Path(config_file).exists():
                import json
                with open(config_file, 'r') as f:
                    config = json.load(f)
                # Override default engine if specified in config
                if 'default_engine' in config:
                    engine_name = config.pop('default_engine')
            else:
                # Check for default config file locations
                default_config_paths = [
                    Path('/app/config.json'),
                    Path('./config.json'),
                    Path(os.getenv('STT_CONFIG_FILE', ''))
                ]
                for config_path in default_config_paths:
                    if config_path and config_path.exists():
                        import json
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        if 'default_engine' in config:
                            engine_name = config.pop('default_engine')
                        break
                else:
                    config = self._build_default_config()
        
        # Initialize the engine manager
        self.manager = STTEngineManager(default_engine=engine_name, config=config)
        
        # For backward compatibility, check if old model exists
        self._setup_legacy_support()
    
    def _build_default_config(self) -> Dict[str, Any]:
        """Build default configuration from environment and file paths"""
        config = {}
        
        # DeepSpeech configuration
        deepspeech_model_paths = [
            Path(__file__).parents[1].joinpath('model.tflite'),
            Path(__file__).parents[1].joinpath('model.pbmm'),
            Path('/app/model.pbmm'),
            Path('/app/model.tflite')
        ]
        for path in deepspeech_model_paths:
            if path.exists():
                config['deepspeech'] = {'model_path': str(path.absolute())}
                break
        
        # Whisper configuration
        config['whisper'] = {
            'model_size': os.getenv('WHISPER_MODEL_SIZE', 'base'),
            'device': os.getenv('WHISPER_DEVICE', 'cpu'),
            'language': os.getenv('WHISPER_LANGUAGE', None)
        }
        
        # Coqui configuration
        coqui_model_paths = [
            Path(__file__).parents[1].joinpath('coqui_model.tflite'),
            Path(__file__).parents[1].joinpath('coqui_model.pbmm'),
            Path('/app/coqui_model.tflite'),
            Path('/app/coqui_model.pbmm')
        ]
        for path in coqui_model_paths:
            if path.exists():
                config['coqui'] = {'model_path': str(path.absolute())}
                break
        
        # Vosk configuration
        vosk_model_path = os.getenv('VOSK_MODEL_PATH')
        if vosk_model_path:
            config['vosk'] = {'model_path': vosk_model_path}
        
        # Silero configuration
        config['silero'] = {
            'language': os.getenv('SILERO_LANGUAGE', 'en'),
            'device': os.getenv('SILERO_DEVICE', 'cpu')
        }
        
        # Wav2Vec2 configuration
        config['wav2vec2'] = {
            'model_name': os.getenv('WAV2VEC2_MODEL', 'facebook/wav2vec2-base-960h'),
            'device': os.getenv('WAV2VEC2_DEVICE', 'cpu')
        }
        
        return config
    
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