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


class Wav2Vec2Engine(BaseSTTEngine):
    """Wav2Vec2 STT Engine using HuggingFace Transformers"""
    
    def initialize(self):
        """Initialize Wav2Vec2 model"""
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
            import torch
            
            # Get model name/path
            model_name = self.config.get('model_name', 'facebook/wav2vec2-base-960h')
            self.device = self.config.get('device', 'cpu')
            
            # Load model and processor
            cache_dir = self.config.get('cache_dir', None)
            
            self.processor = Wav2Vec2Processor.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            self.model = Wav2Vec2ForCTC.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # Move model to device
            if self.device != 'cpu':
                import torch
                self.model = self.model.to(torch.device(self.device))
            
            self.model.eval()
            
        except ImportError as e:
            raise EngineNotAvailableError(
                "Required packages not installed. Install with: pip install transformers torch",
                engine="wav2vec2",
                original_error=e
            )
        except Exception as e:
            if "model" in str(e).lower() or "not found" in str(e).lower():
                raise ModelNotFoundError(
                    f"Failed to load Wav2Vec2 model '{model_name}': {e}",
                    engine="wav2vec2",
                    original_error=e
                )
            raise EngineInitializationError(
                f"Failed to initialize Wav2Vec2 engine: {e}",
                engine="wav2vec2",
                original_error=e
            )
    
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using Wav2Vec2"""
        try:
            import torch
            
            # Convert to float32 and normalize
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # Resample if needed (Wav2Vec2 expects 16kHz)
            if sample_rate != 16000:
                try:
                    import librosa
                    audio_float = librosa.resample(audio_float, orig_sr=sample_rate, target_sr=16000)
                except Exception as e:
                    raise AudioProcessingError(
                        f"Failed to resample audio from {sample_rate}Hz to 16000Hz: {e}",
                        engine="wav2vec2",
                        original_error=e
                    )
            
            # Process audio
            try:
                inputs = self.processor(
                    audio_float, 
                    sampling_rate=16000, 
                    return_tensors="pt",
                    padding=True
                )
            except Exception as e:
                raise AudioProcessingError(
                    f"Failed to process audio with Wav2Vec2 processor: {e}",
                    engine="wav2vec2",
                    original_error=e
                )
            
            # Move to device if needed
            if self.device != 'cpu':
                inputs = {k: v.to(torch.device(self.device)) for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                logits = self.model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
            
            return transcription.strip()
            
        except (AudioProcessingError, UnsupportedAudioFormatError):
            raise  # Re-raise audio processing errors
        except Exception as e:
            raise TranscriptionError(
                f"Wav2Vec2 transcription failed: {e}",
                engine="wav2vec2",
                original_error=e
            )
    
    def _check_availability(self) -> bool:
        """Check if Wav2Vec2 is available"""
        try:
            import transformers
            import torch
            return True
        except ImportError:
            return False