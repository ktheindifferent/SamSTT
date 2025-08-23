from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from ..base_engine import BaseSTTEngine
from ..exceptions import (
    EngineNotAvailableError,
    EngineInitializationError,
    ModelNotFoundError,
    TranscriptionError,
    UnsupportedAudioFormatError,
    ConfigurationError
)


class CoquiEngine(BaseSTTEngine):
    """Coqui STT Engine (formerly Mozilla DeepSpeech)"""
    
    def initialize(self):
        """Initialize Coqui STT model"""
        try:
            from STT import Model
        except ImportError as e:
            raise EngineNotAvailableError(
                "Coqui STT package not installed. Install with: pip install STT",
                engine="coqui",
                original_error=e
            )
        
        try:
            model_path = self.config.get('model_path')
            if not model_path:
                # Try default paths for Coqui models
                default_paths = [
                    Path(__file__).parents[2] / 'coqui_model.tflite',
                    Path(__file__).parents[2] / 'coqui_model.pbmm',
                    Path('/app/coqui_model.tflite'),
                    Path('/app/coqui_model.pbmm')
                ]
                for path in default_paths:
                    if path.exists():
                        model_path = str(path.absolute())
                        break
                
                if not model_path:
                    raise ModelNotFoundError(
                        "Coqui STT model not found in default locations. Expected locations: " +
                        ", ".join(str(p) for p in default_paths),
                        engine="coqui",
                        model_path=str(default_paths[0])
                    )
            
            # Verify model file exists
            if not Path(model_path).exists():
                raise ModelNotFoundError(
                    f"Coqui STT model file not found: {model_path}",
                    engine="coqui",
                    model_path=model_path
                )
            
            self.model = Model(model_path)
            
            # Set beam width if specified
            beam_width = self.config.get('beam_width')
            if beam_width:
                try:
                    self.model.setBeamWidth(beam_width)
                except Exception as e:
                    raise ConfigurationError(
                        f"Invalid beam width parameter: {beam_width}",
                        engine="coqui",
                        original_error=e
                    )
            
            # Load scorer if available
            scorer_path = self.config.get('scorer_path')
            if scorer_path:
                if not Path(scorer_path).exists():
                    raise ModelNotFoundError(
                        f"Scorer file not found: {scorer_path}",
                        engine="coqui",
                        model_path=scorer_path
                    )
                self.model.enableExternalScorer(scorer_path)
                
                # Set scorer parameters if provided
                lm_alpha = self.config.get('lm_alpha')
                lm_beta = self.config.get('lm_beta')
                if lm_alpha and lm_beta:
                    try:
                        self.model.setScorerAlphaBeta(lm_alpha, lm_beta)
                    except Exception as e:
                        raise ConfigurationError(
                            f"Invalid scorer parameters: alpha={lm_alpha}, beta={lm_beta}",
                            engine="coqui",
                            original_error=e
                        )
                    
        except (ModelNotFoundError, EngineNotAvailableError, ConfigurationError):
            raise
        except Exception as e:
            raise EngineInitializationError(
                f"Failed to initialize Coqui STT: {str(e)}",
                engine="coqui",
                original_error=e
            )
    
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using Coqui STT"""
        expected_rate = self.model.sampleRate()
        if sample_rate != expected_rate:
            raise UnsupportedAudioFormatError(
                f"Coqui STT requires {expected_rate}Hz audio, got {sample_rate}Hz",
                engine="coqui"
            )
        
        try:
            result = self.model.stt(audio_data)
            return result
        except Exception as e:
            raise TranscriptionError(
                f"Coqui STT transcription failed: {str(e)}",
                engine="coqui",
                original_error=e
            )
    
    def _check_availability(self) -> bool:
        """Check if Coqui STT is available"""
        try:
            import STT
            # Check if model exists
            model_path = self.config.get('model_path')
            if model_path:
                return Path(model_path).exists()
            # Check default locations
            default_paths = [
                Path(__file__).parents[2] / 'coqui_model.tflite',
                Path(__file__).parents[2] / 'coqui_model.pbmm',
                Path('/app/coqui_model.tflite'),
                Path('/app/coqui_model.pbmm')
            ]
            return any(path.exists() for path in default_paths)
        except ImportError:
            return False