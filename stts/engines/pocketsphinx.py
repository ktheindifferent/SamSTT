from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import os
import logging
from ..base_engine import BaseSTTEngine

logger = logging.getLogger(__name__)


class PocketSphinxEngine(BaseSTTEngine):
    """PocketSphinx STT Engine - Lightweight speech recognition from CMU"""
    
    def initialize(self):
        """Initialize PocketSphinx decoder"""
        try:
            from pocketsphinx import Pocketsphinx, get_model_path, get_data_path
            
            # Get configuration
            try:
                model_path = self.config.get('model_path', get_model_path())
                data_path = self.config.get('data_path', get_data_path())
            except Exception as path_error:
                # Fallback to manual paths if get_model_path fails
                logger.warning(f"Could not get default paths: {path_error}, using fallback")
                model_path = self.config.get('model_path', '/usr/local/share/pocketsphinx/model')
                data_path = self.config.get('data_path', '/usr/local/share/pocketsphinx/model')
            
            # Configure decoder parameters with fallback paths
            hmm_path = self.config.get('hmm', os.path.join(model_path, 'en-us'))
            lm_path = self.config.get('lm', os.path.join(model_path, 'en-us.lm.bin'))
            dict_path = self.config.get('dict', os.path.join(model_path, 'cmudict-en-us.dict'))
            
            # Check if paths exist, use simpler config if not
            config = {}
            if os.path.exists(hmm_path):
                config['hmm'] = hmm_path
            if os.path.exists(lm_path):
                config['lm'] = lm_path
            if os.path.exists(dict_path):
                config['dict'] = dict_path
            
            # Add optional parameters
            if 'keyphrase' in self.config:
                config['keyphrase'] = self.config['keyphrase']
                config['kws_threshold'] = self.config.get('kws_threshold', 1e-20)
            
            # If no model files found, try basic initialization
            if not config:
                logger.warning("No PocketSphinx model files found, trying basic initialization")
                config = {}  # Let PocketSphinx use defaults
            
            self.ps = Pocketsphinx(**config)
            self.sample_rate = 16000  # PocketSphinx expects 16kHz
            
        except ImportError as e:
            raise ImportError(f"PocketSphinx not installed: {e}. Install with: pip install pocketsphinx")
        except Exception as e:
            raise Exception(f"Failed to initialize PocketSphinx: {e}")
    
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe using PocketSphinx"""
        import os
        
        if sample_rate != 16000:
            # PocketSphinx requires 16kHz audio
            import scipy.signal
            # Resample to 16kHz
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = scipy.signal.resample(audio_data, num_samples)
        
        # Convert to int16 if needed
        if audio_data.dtype != np.int16:
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
        
        # Decode audio
        self.ps.start_utt()
        self.ps.process_raw(audio_data.tobytes(), False, True)
        self.ps.end_utt()
        
        # Get hypothesis
        hypothesis = self.ps.hypothesis()
        if hypothesis:
            return hypothesis
        
        # Try to get partial result if no final hypothesis
        return self.ps.partial_hypothesis() or ""
    
    def _check_availability(self) -> bool:
        """Check if PocketSphinx is available"""
        try:
            import pocketsphinx
            return True
        except ImportError:
            return False