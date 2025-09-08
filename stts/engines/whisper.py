from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from ..base_engine import BaseSTTEngine


class WhisperEngine(BaseSTTEngine):
    """Whisper.cpp Engine - Fast C++ implementation of OpenAI's Whisper"""
    
    def initialize(self):
        """Initialize Whisper.cpp model"""
        try:
            from pywhispercpp.model import Model
            import os
            
            # Get model configuration
            model_name = self.config.get('model_size', os.environ.get('WHISPER_MODEL_SIZE', 'base'))
            
            # Map model sizes to whisper.cpp model names
            model_map = {
                'tiny': 'tiny',
                'tiny.en': 'tiny.en',
                'base': 'base',
                'base.en': 'base.en', 
                'small': 'small',
                'small.en': 'small.en',
                'medium': 'medium',
                'medium.en': 'medium.en',
                'large-v1': 'large-v1',
                'large-v2': 'large-v2',
                'large-v3': 'large-v3',
                'large': 'large-v3'  # Default to latest
            }
            
            model_name = model_map.get(model_name, model_name)
            self.model_name = model_name  # Store for config display
            
            # Check for available models in model directory
            model_dir = os.environ.get('PYWHISPERCPP_MODEL_DIR', '/app/models/whisper')
            available_models = []
            if Path(model_dir).exists():
                for model_file in Path(model_dir).glob('ggml-*.bin'):
                    model_type = model_file.name.replace('ggml-', '').replace('.bin', '')
                    available_models.append(model_type)
            
            # Always set available_models, even if empty, so it shows in config
            self.available_models = available_models
            
            # Log what models we found
            import logging
            logger = logging.getLogger(__name__)
            if available_models:
                logger.info(f"Found Whisper models: {available_models}")
            else:
                logger.warning(f"No Whisper models found in {model_dir}")
            
            # Check for custom model path
            model_path = self.config.get('model_path')
            
            try:
                if model_path and Path(model_path).exists():
                    # Use custom model file
                    self.model = Model(model_path)
                else:
                    # Try to use specific model file if available
                    specific_model_path = Path(model_dir) / f'ggml-{model_name}.bin'
                    if specific_model_path.exists():
                        self.model = Model(str(specific_model_path))
                    else:
                        # Fall back to download or default
                        self.model = Model(model_name, 
                                         n_threads=self.config.get('n_threads', 4),
                                         models_dir=model_dir)
                
                # Set language if specified
                self.language = self.config.get('language', None)
                
            except Exception as init_error:
                # If initialization fails, mark as unavailable
                self.model = None
                raise Exception(f"Whisper.cpp model initialization failed: {init_error}")
            
        except ImportError:
            raise ImportError("Whisper.cpp (pywhispercpp) not installed. Install with: pip install pywhispercpp")
        except Exception as e:
            raise Exception(f"Failed to initialize Whisper.cpp: {e}")
    
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using Whisper.cpp"""
        if not hasattr(self, 'model') or self.model is None:
            raise Exception("Whisper.cpp model not initialized")
            
        try:
            # Whisper.cpp expects float32 audio in range [-1, 1]
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Ensure audio is in correct range
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # Transcribe with options
            # Whisper.cpp expects language as a string, not None
            lang = self.language if self.language else 'en'
            
            # Basic transcribe without optional parameters that might cause issues
            segments = self.model.transcribe(
                audio_data,
                language=lang
            )
            
            # Combine all segments into single transcript
            transcript = ' '.join([seg.text for seg in segments])
            return transcript.strip()
            
        except Exception as e:
            raise Exception(f"Whisper.cpp transcription failed: {e}")
    
    def _check_availability(self) -> bool:
        """Check if Whisper.cpp is available"""
        try:
            import pywhispercpp
            return True
        except ImportError:
            return False