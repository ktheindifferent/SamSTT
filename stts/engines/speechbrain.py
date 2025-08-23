from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from ..base_engine import BaseSTTEngine
from ..exceptions import (
    EngineNotAvailableError,
    EngineInitializationError,
    ModelNotFoundError,
    TranscriptionError,
    AudioProcessingError,
    UnsupportedAudioFormatError
)


class SpeechBrainEngine(BaseSTTEngine):
    """SpeechBrain STT Engine - PyTorch-based end-to-end speech toolkit"""
    
    def initialize(self):
        """Initialize SpeechBrain model"""
        try:
            from speechbrain.pretrained import EncoderDecoderASR, EncoderASR
            import torch
            
            # Get model configuration
            source = self.config.get('source', 'speechbrain/asr-crdnn-rnnlm-librispeech')
            savedir = self.config.get('savedir', 'pretrained_models/asr-crdnn')
            model_type = self.config.get('model_type', 'encoder_decoder')  # or 'encoder'
            self.device = self.config.get('device', 'cpu')
            
            # Load appropriate model type
            if model_type == 'encoder':
                self.model = EncoderASR.from_hparams(
                    source=source,
                    savedir=savedir,
                    run_opts={"device": self.device}
                )
            else:
                self.model = EncoderDecoderASR.from_hparams(
                    source=source,
                    savedir=savedir,
                    run_opts={"device": self.device}
                )
            
        except ImportError as e:
            raise EngineNotAvailableError(
                "SpeechBrain not installed. Install with: pip install speechbrain",
                engine="speechbrain",
                original_error=e
            )
        except Exception as e:
            if "model" in str(e).lower() or "pretrained" in str(e).lower() or "not found" in str(e).lower():
                raise ModelNotFoundError(
                    f"Failed to load SpeechBrain model from '{source}': {e}",
                    engine="speechbrain",
                    original_error=e
                )
            raise EngineInitializationError(
                f"Failed to initialize SpeechBrain engine: {e}",
                engine="speechbrain",
                original_error=e
            )
    
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using SpeechBrain"""
        try:
            import torch
            
            # Convert to float32 tensor
            if audio_data.dtype == np.int16:
                audio_tensor = torch.from_numpy(audio_data.astype(np.float32) / 32768.0)
            else:
                audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
            
            # Ensure correct sample rate
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
                        engine="speechbrain",
                        original_error=e
                    )
            
            # Add batch dimension
            audio_tensor = audio_tensor.unsqueeze(0)
            
            # Transcribe
            predicted_words, predicted_tokens = self.model.transcribe_batch(
                audio_tensor,
                [torch.tensor([1.0])]  # Relative length
            )
            
            return predicted_words[0] if predicted_words else ""
            
        except AudioProcessingError:
            raise  # Re-raise audio processing errors
        except Exception as e:
            raise TranscriptionError(
                f"SpeechBrain transcription failed: {e}",
                engine="speechbrain",
                original_error=e
            )
    
    def _check_availability(self) -> bool:
        """Check if SpeechBrain is available"""
        try:
            import speechbrain
            import torch
            return True
        except ImportError:
            return False