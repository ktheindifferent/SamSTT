from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from ..base_engine import BaseSTTEngine


class WhisperEngine(BaseSTTEngine):
    """OpenAI Whisper STT Engine (offline)"""
    
    def initialize(self):
        """Initialize Whisper model"""
        try:
            import whisper
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
            
            # Import librosa once during initialization to avoid memory leaks
            # from repeated dynamic imports in transcribe_raw
            try:
                import librosa
                self.librosa = librosa
            except ImportError:
                # librosa is optional - only needed for resampling
                self.librosa = None
                
        except ImportError:
            raise ImportError("Whisper package not installed. Install with: pip install openai-whisper")
        except Exception as e:
            raise Exception(f"Failed to initialize Whisper: {e}")
    
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using Whisper"""
        # Convert to float32 and normalize
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Whisper expects audio at 16kHz
        if sample_rate != 16000:
            if self.librosa is None:
                raise RuntimeError("librosa not available for resampling. Install with: pip install librosa")
            audio_float = self.librosa.resample(audio_float, orig_sr=sample_rate, target_sr=16000)
        
        # Transcribe
        result = self.model.transcribe(audio_float, **self.transcribe_options)
        return result['text'].strip()
    
    def _check_availability(self) -> bool:
        """Check if Whisper is available"""
        try:
            import whisper
            return True
        except ImportError:
            return False