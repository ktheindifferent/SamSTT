from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from ..base_engine import BaseSTTEngine


class SpeechBrainEngine(BaseSTTEngine):
    """SpeechBrain STT Engine - PyTorch-based end-to-end speech toolkit"""
    
    def initialize(self):
        """Initialize SpeechBrain model"""
        try:
            from speechbrain.pretrained import EncoderDecoderASR
            import torch
            import os
            
            # Get model configuration - use a simpler, more reliable model
            source = self.config.get('source', 'speechbrain/asr-wav2vec2-commonvoice-en')
            savedir = self.config.get('savedir', '/app/cache/speechbrain/asr-wav2vec2')
            self.device = self.config.get('device', 'cpu')
            
            # Create savedir if it doesn't exist
            os.makedirs(savedir, exist_ok=True)
            
            # Load model with simplified approach
            self.model = EncoderDecoderASR.from_hparams(
                source=source,
                savedir=savedir,
                run_opts={"device": self.device}
            )
            
        except ImportError as e:
            raise ImportError(f"SpeechBrain not installed: {e}. Install with: pip install speechbrain")
        except Exception as e:
            raise Exception(f"Failed to initialize SpeechBrain: {e}")
    
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using SpeechBrain"""
        import torch
        
        # Convert to float32 tensor
        if audio_data.dtype == np.int16:
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32) / 32768.0)
        else:
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
        
        # Ensure correct sample rate
        if sample_rate != 16000:
            import torchaudio
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.unsqueeze(0),
                sample_rate,
                16000
            ).squeeze(0)
        
        # Add batch dimension
        audio_tensor = audio_tensor.unsqueeze(0)
        
        # Transcribe
        predicted_words, predicted_tokens = self.model.transcribe_batch(
            audio_tensor,
            [torch.tensor([1.0])]  # Relative length
        )
        
        return predicted_words[0] if predicted_words else ""
    
    def _check_availability(self) -> bool:
        """Check if SpeechBrain is available"""
        try:
            import speechbrain
            import torch
            return True
        except ImportError:
            return False