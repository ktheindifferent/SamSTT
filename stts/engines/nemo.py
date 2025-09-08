from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import logging
import atexit
import glob
import os
from ..base_engine import BaseSTTEngine

logger = logging.getLogger(__name__)


class NeMoEngine(BaseSTTEngine):
    """NVIDIA NeMo STT Engine - Neural Modules for speech processing"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register cleanup on exit
        atexit.register(self._cleanup_orphaned_files)
        # Clean up any orphaned files from previous runs
        self._cleanup_orphaned_files()
    
    def _cleanup_orphaned_files(self):
        """Clean up orphaned temporary files from previous runs"""
        try:
            import tempfile
            temp_dir = tempfile.gettempdir()
            # Look for orphaned NeMo temp files (pattern: nemo_*.wav)
            pattern = os.path.join(temp_dir, 'nemo_*.wav')
            orphaned_files = glob.glob(pattern)
            
            for file_path in orphaned_files:
                try:
                    # Only remove files older than 1 hour to avoid conflicts
                    if os.path.exists(file_path):
                        import time
                        file_age = time.time() - os.path.getmtime(file_path)
                        if file_age > 3600:  # 1 hour in seconds
                            os.unlink(file_path)
                            logger.debug(f"Cleaned up orphaned temp file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up orphaned file {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to clean up orphaned files: {e}")
    
    def initialize(self):
        """Initialize NeMo ASR model"""
        try:
            import nemo.collections.asr as nemo_asr
            import torch
            import os
            
            # Get model configuration - use a more reliable small model
            model_name = self.config.get('model_name', 'stt_en_quartznet15x5')
            self.device = self.config.get('device', 'cpu')
            restore_from = self.config.get('restore_from', None)
            
            # Create cache directory
            cache_dir = '/app/cache/nemo'
            os.makedirs(cache_dir, exist_ok=True)
            
            if restore_from and Path(restore_from).exists():
                # Load from local checkpoint
                self.model = nemo_asr.models.ASRModel.restore_from(
                    restore_from,
                    map_location=self.device
                )
            else:
                # Load pretrained model from NGC with timeout protection
                try:
                    self.model = nemo_asr.models.ASRModel.from_pretrained(
                        model_name=model_name,
                        map_location=self.device
                    )
                except Exception as download_error:
                    # Try alternative model name format
                    if 'stt_en_quartznet15x5' in model_name:
                        alt_model = 'QuartzNet15x5Base-En'
                        logger.warning(f"Primary model {model_name} failed, trying {alt_model}: {download_error}")
                        self.model = nemo_asr.models.ASRModel.from_pretrained(
                            model_name=alt_model,
                            map_location=self.device
                        )
                    else:
                        raise download_error
            
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
        """Transcribe using NeMo with proper temp file cleanup"""
        import soundfile as sf
        import tempfile
        import os
        import uuid
        
        # Generate unique filename to avoid conflicts
        temp_filename = f"nemo_{uuid.uuid4().hex}.wav"
        temp_filepath = None
        
        try:
            # Create temporary file with specific prefix for easier cleanup
            with tempfile.NamedTemporaryFile(
                prefix='nemo_',
                suffix='.wav',
                delete=False
            ) as tmp_file:
                temp_filepath = tmp_file.name
                logger.debug(f"Created temp file for NeMo: {temp_filepath}")
                
                # Convert to float32
                if audio_data.dtype == np.int16:
                    audio_float = audio_data.astype(np.float32) / 32768.0
                else:
                    audio_float = audio_data.astype(np.float32)
                
                # Write audio data to temporary file
                sf.write(temp_filepath, audio_float, sample_rate)
                logger.debug(f"Written audio data to temp file: {temp_filepath}")
            
            # File is closed here, now transcribe
            transcriptions = self.model.transcribe([temp_filepath])
            
            # Return first transcription
            if transcriptions and len(transcriptions) > 0:
                result = transcriptions[0]
                logger.debug(f"Transcription successful for {temp_filepath}")
                return result
            return ""
            
        except Exception as e:
            logger.error(f"Error during NeMo transcription: {e}")
            raise
        finally:
            # Always clean up temporary file
            if temp_filepath and os.path.exists(temp_filepath):
                try:
                    os.unlink(temp_filepath)
                    logger.debug(f"Deleted temp file: {temp_filepath}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_filepath}: {e}")
    
    def _check_availability(self) -> bool:
        """Check if NeMo is available"""
        try:
            import nemo.collections.asr as nemo_asr
            import torch
            import soundfile
            return True
        except ImportError:
            return False