from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch
from ..base_engine import BaseSTTEngine


class SileroEngine(BaseSTTEngine):
    """Silero STT Engine - PyTorch-based lightweight models"""
    
    def initialize(self):
        """Initialize Silero model"""
        try:
            import torch
            import torchaudio
            from omegaconf import OmegaConf
            
            self.device = torch.device(self.config.get('device', 'cpu'))
            language = self.config.get('language', 'en')
            
            # Load model from torch hub or local path
            model_path = self.config.get('model_path')
            if model_path and Path(model_path).exists():
                self.model = torch.jit.load(model_path, map_location=self.device)
            else:
                # Download from torch hub
                self.model, self.decoder, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_stt',
                    language=language,
                    device=self.device
                )
                self.read_batch = utils[0]
                self.split_into_batches = utils[1]
                self.read_audio = utils[2]
            
            self.model.eval()
            
        except ImportError as e:
            raise ImportError(f"Required packages not installed for Silero: {e}. Install with: pip install torch torchaudio omegaconf")
        except Exception as e:
            raise Exception(f"Failed to initialize Silero: {e}")
    
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using Silero"""
        import torch
        
        # Convert to float32 tensor
        if audio_data.dtype == np.int16:
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32) / 32768.0)
        else:
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
        
        # Ensure correct sample rate (Silero typically expects 16kHz)
        if sample_rate != 16000:
            import torchaudio
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.unsqueeze(0), 
                sample_rate, 
                16000
            ).squeeze(0)
        
        # Add batch dimension
        audio_tensor = audio_tensor.unsqueeze(0).to(self.device)
        
        # Transcribe
        with torch.no_grad():
            if hasattr(self, 'decoder'):
                output = self.model(audio_tensor)
                transcription = self.decoder(output[0])
            else:
                # For direct model usage
                transcription = self.model(audio_tensor)
                if isinstance(transcription, tuple):
                    transcription = transcription[0]
        
        return transcription if isinstance(transcription, str) else ''
    
    def _check_availability(self) -> bool:
        """Check if Silero is available"""
        try:
            import torch
            import torchaudio
            import omegaconf
            return True
        except ImportError:
            return False