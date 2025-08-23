from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from ..base_engine import BaseSTTEngine
from ..exceptions import (
    EngineNotAvailableError,
    EngineInitializationError,
    TranscriptionError,
    AudioProcessingError,
    InsufficientResourcesError
)


class WhisperEngine(BaseSTTEngine):
    """OpenAI Whisper STT Engine (offline)"""
    
    def initialize(self):
        """Initialize Whisper model"""
        try:
            import whisper
        except ImportError as e:
            raise EngineNotAvailableError(
                "Whisper package not installed. Install with: pip install openai-whisper",
                engine="whisper",
                original_error=e
            )
        
        try:
            model_size = self.config.get('model_size', 'base')
            self.device = self.config.get('device', 'cpu')
            download_root = self.config.get('download_root', None)
            
            # Load model
            self.model = whisper.load_model(
                model_size, 
                device=self.device,
                download_root=download_root
            )
            
            # Set default options
            self.transcribe_options = {
                'language': self.config.get('language', None),
                'task': self.config.get('task', 'transcribe'),
                'fp16': self.config.get('fp16', False),
                'verbose': self.config.get('verbose', False)
            }
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                raise InsufficientResourcesError(
                    f"Insufficient memory to load Whisper model '{model_size}'. Try a smaller model or increase available memory.",
                    engine="whisper",
                    original_error=e
                )
            raise EngineInitializationError(
                f"Failed to initialize Whisper model '{model_size}': {str(e)}",
                engine="whisper",
                original_error=e
            )
        except Exception as e:
            raise EngineInitializationError(
                f"Failed to initialize Whisper: {str(e)}",
                engine="whisper",
                original_error=e
            )
    
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using Whisper"""
        try:
            # Convert to float32 and normalize
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Whisper expects audio at 16kHz
            if sample_rate != 16000:
                try:
                    import librosa
                    audio_float = librosa.resample(audio_float, orig_sr=sample_rate, target_sr=16000)
                except ImportError as e:
                    raise AudioProcessingError(
                        "librosa package required for audio resampling. Install with: pip install librosa",
                        engine="whisper",
                        original_error=e
                    )
                except Exception as e:
                    raise AudioProcessingError(
                        f"Failed to resample audio from {sample_rate}Hz to 16000Hz: {str(e)}",
                        engine="whisper",
                        original_error=e
                    )
            
            # Transcribe
            result = self.model.transcribe(audio_float, **self.transcribe_options)
            return result['text'].strip()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                raise InsufficientResourcesError(
                    "Insufficient memory during Whisper transcription. Try reducing audio length or using a smaller model.",
                    engine="whisper",
                    original_error=e
                )
            raise TranscriptionError(
                f"Whisper transcription failed: {str(e)}",
                engine="whisper",
                original_error=e
            )
        except Exception as e:
            if not isinstance(e, (AudioProcessingError, InsufficientResourcesError)):
                raise TranscriptionError(
                    f"Whisper transcription failed: {str(e)}",
                    engine="whisper",
                    original_error=e
                )
            raise
    
    def _check_availability(self) -> bool:
        """Check if Whisper is available"""
        try:
            import whisper
            return True
        except ImportError:
            return False