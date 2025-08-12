from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from ..base_engine import BaseSTTEngine


class NeMoEngine(BaseSTTEngine):
    """NVIDIA NeMo STT Engine - Neural Modules for speech processing"""
    
    def initialize(self):
        """Initialize NeMo ASR model"""
        try:
            import nemo.collections.asr as nemo_asr
            import torch
            
            # Get model configuration
            model_name = self.config.get('model_name', 'QuartzNet15x5Base-En')
            self.device = self.config.get('device', 'cpu')
            restore_from = self.config.get('restore_from', None)
            
            if restore_from and Path(restore_from).exists():
                # Load from local checkpoint
                self.model = nemo_asr.models.ASRModel.restore_from(
                    restore_from,
                    map_location=self.device
                )
            else:
                # Load pretrained model from NGC
                self.model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=model_name,
                    map_location=self.device
                )
            
            self.model.eval()
            
            # Store sample rate requirement
            self.required_sample_rate = 16000
            if hasattr(self.model, 'cfg') and hasattr(self.model.cfg, 'sample_rate'):
                self.required_sample_rate = self.model.cfg.sample_rate
                
        except ImportError as e:
            raise ImportError(f"NeMo not installed: {e}. Install with: pip install nemo_toolkit[asr]")
        except Exception as e:
            raise Exception(f"Failed to initialize NeMo: {e}")
    
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using NeMo"""
        import torch
        import soundfile as sf
        import tempfile
        import os
        
        # NeMo works best with file paths, so we'll use a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            try:
                # Convert to float32
                if audio_data.dtype == np.int16:
                    audio_float = audio_data.astype(np.float32) / 32768.0
                else:
                    audio_float = audio_data.astype(np.float32)
                
                # Write to temporary file
                sf.write(tmp_file.name, audio_float, sample_rate)
                
                # Transcribe
                transcriptions = self.model.transcribe([tmp_file.name])
                
                # Return first transcription
                if transcriptions and len(transcriptions) > 0:
                    return transcriptions[0]
                return ""
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def _check_availability(self) -> bool:
        """Check if NeMo is available"""
        try:
            import nemo.collections.asr as nemo_asr
            import torch
            import soundfile
            return True
        except ImportError:
            return False