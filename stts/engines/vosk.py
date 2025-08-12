import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from ..base_engine import BaseSTTEngine


class VoskEngine(BaseSTTEngine):
    """Vosk STT Engine - lightweight and supports many languages"""
    
    def initialize(self):
        """Initialize Vosk model"""
        try:
            import vosk
            
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
                    raise FileNotFoundError("Vosk model not found in default locations")
            
            self.model = vosk.Model(model_path)
            self.sample_rate = self.config.get('sample_rate', 16000)
            
        except ImportError:
            raise ImportError("Vosk package not installed. Install with: pip install vosk")
        except Exception as e:
            raise Exception(f"Failed to initialize Vosk: {e}")
    
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using Vosk"""
        import vosk
        
        # Create recognizer for this transcription
        rec = vosk.KaldiRecognizer(self.model, sample_rate)
        rec.SetWords(True)
        
        # Convert audio to bytes if needed
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        audio_bytes = audio_data.tobytes()
        
        # Process audio
        rec.AcceptWaveform(audio_bytes)
        result = json.loads(rec.FinalResult())
        
        return result.get('text', '')
    
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