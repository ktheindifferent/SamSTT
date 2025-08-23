from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import os
from ..base_engine import BaseSTTEngine
from ..exceptions import (
    EngineNotAvailableError,
    EngineInitializationError,
    ModelNotFoundError,
    TranscriptionError,
    AudioProcessingError,
    UnsupportedAudioFormatError
)


class PocketSphinxEngine(BaseSTTEngine):
    """PocketSphinx STT Engine - Lightweight speech recognition from CMU"""
    
    def initialize(self):
        """Initialize PocketSphinx decoder"""
        try:
            from pocketsphinx import Pocketsphinx, get_model_path, get_data_path
            
            # Get configuration
            model_path = self.config.get('model_path', get_model_path())
            data_path = self.config.get('data_path', get_data_path())
            
            # Configure decoder parameters
            config = {
                'hmm': self.config.get('hmm', os.path.join(model_path, 'en-us')),
                'lm': self.config.get('lm', os.path.join(model_path, 'en-us.lm.bin')),
                'dict': self.config.get('dict', os.path.join(model_path, 'cmudict-en-us.dict'))
            }
            
            # Add optional parameters
            if 'keyphrase' in self.config:
                config['keyphrase'] = self.config['keyphrase']
                config['kws_threshold'] = self.config.get('kws_threshold', 1e-20)
            
            self.ps = Pocketsphinx(**config)
            self.sample_rate = 16000  # PocketSphinx expects 16kHz
            
        except ImportError as e:
            raise EngineNotAvailableError(
                "PocketSphinx not installed. Install with: pip install pocketsphinx",
                engine="pocketsphinx",
                original_error=e
            )
        except Exception as e:
            # Check if it's a model file issue
            if any(path_key in str(e).lower() for path_key in ['hmm', 'lm', 'dict', 'model', 'file not found']):
                raise ModelNotFoundError(
                    f"PocketSphinx model files not found. Check model paths: {e}",
                    engine="pocketsphinx",
                    original_error=e
                )
            raise EngineInitializationError(
                f"Failed to initialize PocketSphinx engine: {e}",
                engine="pocketsphinx",
                original_error=e
            )
    
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using PocketSphinx"""
        try:
            import os
            
            if sample_rate != 16000:
                # PocketSphinx requires 16kHz audio
                try:
                    import scipy.signal
                    # Resample to 16kHz
                    num_samples = int(len(audio_data) * 16000 / sample_rate)
                    audio_data = scipy.signal.resample(audio_data, num_samples)
                except Exception as e:
                    raise AudioProcessingError(
                        f"Failed to resample audio from {sample_rate}Hz to 16000Hz: {e}",
                        engine="pocketsphinx",
                        original_error=e
                    )
            
            # Convert to int16 if needed
            try:
                if audio_data.dtype != np.int16:
                    if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                        audio_data = (audio_data * 32767).astype(np.int16)
                    else:
                        audio_data = audio_data.astype(np.int16)
            except Exception as e:
                raise AudioProcessingError(
                    f"Failed to convert audio data to int16 format: {e}",
                    engine="pocketsphinx",
                    original_error=e
                )
            
            # Decode audio
            try:
                self.ps.start_utt()
                self.ps.process_raw(audio_data.tobytes(), False, True)
                self.ps.end_utt()
                
                # Get hypothesis
                hypothesis = self.ps.hypothesis()
                if hypothesis:
                    return hypothesis
                
                # Try to get partial result if no final hypothesis
                return self.ps.partial_hypothesis() or ""
                
            except Exception as e:
                raise TranscriptionError(
                    f"PocketSphinx decoding failed: {e}",
                    engine="pocketsphinx",
                    original_error=e
                )
                
        except (AudioProcessingError, TranscriptionError):
            raise  # Re-raise our custom exceptions
        except Exception as e:
            raise TranscriptionError(
                f"Unexpected error during PocketSphinx transcription: {e}",
                engine="pocketsphinx",
                original_error=e
            )
    
    def _check_availability(self) -> bool:
        """Check if PocketSphinx is available"""
        try:
            import pocketsphinx
            return True
        except ImportError:
            return False