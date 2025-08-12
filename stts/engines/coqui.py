from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from ..base_engine import BaseSTTEngine


class CoquiEngine(BaseSTTEngine):
    """Coqui STT Engine (formerly Mozilla DeepSpeech)"""
    
    def initialize(self):
        """Initialize Coqui STT model"""
        try:
            from STT import Model
            
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
                    raise FileNotFoundError("Coqui STT model not found in default locations")
            
            self.model = Model(model_path)
            
            # Set beam width if specified
            beam_width = self.config.get('beam_width')
            if beam_width:
                self.model.setBeamWidth(beam_width)
            
            # Load scorer if available
            scorer_path = self.config.get('scorer_path')
            if scorer_path and Path(scorer_path).exists():
                self.model.enableExternalScorer(scorer_path)
                
                # Set scorer parameters if provided
                lm_alpha = self.config.get('lm_alpha')
                lm_beta = self.config.get('lm_beta')
                if lm_alpha and lm_beta:
                    self.model.setScorerAlphaBeta(lm_alpha, lm_beta)
                    
        except ImportError:
            raise ImportError("Coqui STT package not installed. Install with: pip install STT")
        except Exception as e:
            raise Exception(f"Failed to initialize Coqui STT: {e}")
    
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using Coqui STT"""
        if sample_rate != self.model.sampleRate():
            raise ValueError(f"Coqui STT requires {self.model.sampleRate()}Hz audio, got {sample_rate}Hz")
        
        return self.model.stt(audio_data)
    
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