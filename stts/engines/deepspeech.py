from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from ..base_engine import BaseSTTEngine
from ..exceptions import (
    EngineNotAvailableError,
    EngineInitializationError,
    ModelNotFoundError,
    TranscriptionError,
    UnsupportedAudioFormatError
)


class DeepSpeechEngine(BaseSTTEngine):
    """Mozilla DeepSpeech STT Engine"""
    
    def initialize(self):
        """Initialize DeepSpeech model"""
        try:
            from stt import Model
        except ImportError as e:
            raise EngineNotAvailableError(
                "DeepSpeech (stt) package not installed. Install with: pip install stt",
                engine="deepspeech",
                original_error=e
            )
        
        try:
            model_path = self.config.get('model_path')
            if not model_path:
                # Try default paths
                default_paths = [
                    Path(__file__).parents[2] / 'model.tflite',
                    Path(__file__).parents[2] / 'model.pbmm',
                    Path('/app/model.pbmm'),
                    Path('/app/model.tflite')
                ]
                for path in default_paths:
                    if path.exists():
                        model_path = str(path.absolute())
                        break
                
                if not model_path:
                    raise ModelNotFoundError(
                        "DeepSpeech model not found in default locations. Expected locations: " + 
                        ", ".join(str(p) for p in default_paths),
                        engine="deepspeech",
                        model_path=str(default_paths[0])
                    )
            
            # Verify model file exists
            if not Path(model_path).exists():
                raise ModelNotFoundError(
                    f"DeepSpeech model file not found: {model_path}",
                    engine="deepspeech",
                    model_path=model_path
                )
            
            self.model = Model(model_path=model_path)
            
            # Load scorer if available
            scorer_path = self.config.get('scorer_path')
            if scorer_path and Path(scorer_path).exists():
                self.model.enableExternalScorer(scorer_path)
        except (ModelNotFoundError, EngineNotAvailableError):
            raise
        except Exception as e:
            raise EngineInitializationError(
                f"Failed to initialize DeepSpeech: {str(e)}",
                engine="deepspeech",
                original_error=e
            )
    
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using DeepSpeech"""
        if sample_rate != 16000:
            raise UnsupportedAudioFormatError(
                f"DeepSpeech requires 16kHz audio, got {sample_rate}Hz",
                engine="deepspeech"
            )
        
        try:
            result = self.model.stt(audio_buffer=audio_data)
            return result
        except Exception as e:
            raise TranscriptionError(
                f"DeepSpeech transcription failed: {str(e)}",
                engine="deepspeech",
                original_error=e
            )
    
    def _check_availability(self) -> bool:
        """Check if DeepSpeech is available"""
        try:
            import stt
            # Check if model exists
            model_path = self.config.get('model_path')
            if model_path:
                return Path(model_path).exists()
            # Check default locations
            default_paths = [
                Path(__file__).parents[2] / 'model.tflite',
                Path(__file__).parents[2] / 'model.pbmm',
                Path('/app/model.pbmm'),
                Path('/app/model.tflite')
            ]
            return any(path.exists() for path in default_paths)
        except ImportError:
            return False