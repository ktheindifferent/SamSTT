import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from ..base_engine import BaseSTTEngine
from ..exceptions import (
    EngineNotAvailableError,
    EngineInitializationError,
    ModelNotFoundError,
    TranscriptionError,
    AudioProcessingError
)


class VoskEngine(BaseSTTEngine):
    """Vosk STT Engine - lightweight and supports many languages"""
    
    def initialize(self):
        """Initialize Vosk model"""
        try:
            import vosk
        except ImportError as e:
            raise EngineNotAvailableError(
                "Vosk package not installed. Install with: pip install vosk",
                engine="vosk",
                original_error=e
            )
        
        try:
            model_path = self.config.get('model_path')
            if not model_path:
                # Try default paths for Vosk models
                default_paths = [
                    Path(__file__).parents[2] / 'vosk_model',
                    Path('/app/vosk_model'),
                    Path.home() / '.vosk' / 'model'
                ]
                for path in default_paths:
                    if path.exists() and path.is_dir():
                        model_path = str(path.absolute())
                        break
                
                if not model_path:
                    raise ModelNotFoundError(
                        "Vosk model not found in default locations. Expected locations: " +
                        ", ".join(str(p) for p in default_paths),
                        engine="vosk",
                        model_path=str(default_paths[0])
                    )
            
            # Verify model directory exists
            if not Path(model_path).exists() or not Path(model_path).is_dir():
                raise ModelNotFoundError(
                    f"Vosk model directory not found or invalid: {model_path}",
                    engine="vosk",
                    model_path=model_path
                )
            
            self.model = vosk.Model(model_path)
            self.sample_rate = self.config.get('sample_rate', 16000)
            
        except (ModelNotFoundError, EngineNotAvailableError):
            raise
        except Exception as e:
            raise EngineInitializationError(
                f"Failed to initialize Vosk: {str(e)}",
                engine="vosk",
                original_error=e
            )
    
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using Vosk"""
        try:
            import vosk
            
            # Create recognizer for this transcription
            rec = vosk.KaldiRecognizer(self.model, sample_rate)
            rec.SetWords(True)
            
            # Convert audio to bytes if needed
            if audio_data.dtype != np.int16:
                try:
                    audio_data = (audio_data * 32767).astype(np.int16)
                except Exception as e:
                    raise AudioProcessingError(
                        f"Failed to convert audio data to int16: {str(e)}",
                        engine="vosk",
                        original_error=e
                    )
            
            audio_bytes = audio_data.tobytes()
            
            # Process audio
            rec.AcceptWaveform(audio_bytes)
            result = json.loads(rec.FinalResult())
            
            return result.get('text', '')
        except json.JSONDecodeError as e:
            raise TranscriptionError(
                f"Failed to parse Vosk result: {str(e)}",
                engine="vosk",
                original_error=e
            )
        except Exception as e:
            if not isinstance(e, AudioProcessingError):
                raise TranscriptionError(
                    f"Vosk transcription failed: {str(e)}",
                    engine="vosk",
                    original_error=e
                )
            raise
    
    def _check_availability(self) -> bool:
        """Check if Vosk is available"""
        try:
            import vosk
            # Check if model exists
            model_path = self.config.get('model_path')
            if model_path:
                return Path(model_path).exists() and Path(model_path).is_dir()
            # Check default locations
            default_paths = [
                Path(__file__).parents[2] / 'vosk_model',
                Path('/app/vosk_model'),
                Path.home() / '.vosk' / 'model'
            ]
            return any(path.exists() and path.is_dir() for path in default_paths)
        except ImportError:
            return False