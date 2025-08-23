from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch
from ..base_engine import BaseSTTEngine
from ..exceptions import (
    EngineNotAvailableError,
    EngineInitializationError,
    ModelNotFoundError,
    TranscriptionError,
    AudioProcessingError,
    UnsupportedAudioFormatError
)


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
            raise EngineNotAvailableError(
                "Required packages not installed for Silero. Install with: pip install torch torchaudio omegaconf",
                engine="silero",
                original_error=e
            )
        except Exception as e:
            if "model" in str(e).lower() or "load" in str(e).lower():
                raise ModelNotFoundError(
                    f"Failed to load Silero model: {e}",
                    engine="silero",
                    original_error=e
                )
            raise EngineInitializationError(
                f"Failed to initialize Silero engine: {e}",
                engine="silero",
                original_error=e
            )
    
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using Silero"""
        try:
            import torch
            
            # Convert to float32 tensor
            if audio_data.dtype == np.int16:
                audio_tensor = torch.from_numpy(audio_data.astype(np.float32) / 32768.0)
            else:
                audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
            
            # Ensure correct sample rate (Silero typically expects 16kHz)
            if sample_rate != 16000:
                try:
                    import torchaudio
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor.unsqueeze(0), 
                        sample_rate, 
                        16000
                    ).squeeze(0)
                except Exception as e:
                    raise AudioProcessingError(
                        f"Failed to resample audio from {sample_rate}Hz to 16000Hz: {e}",
                        engine="silero",
                        original_error=e
                    )
            
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
            
        except AudioProcessingError:
            raise  # Re-raise audio processing errors
        except Exception as e:
            raise TranscriptionError(
                f"Silero transcription failed: {e}",
                engine="silero",
                original_error=e
            )
    
    def _check_availability(self) -> bool:
        """Check if Silero is available"""
        try:
            import torch
            import torchaudio
            import omegaconf
            return True
        except ImportError:
            return False