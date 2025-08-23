from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional
import wave
from io import BytesIO
import numpy as np
import ffmpeg
import logging

logger = logging.getLogger(__name__)


class BaseSTTEngine(ABC):
    """Base class for all STT engine implementations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.initialize()
    
    @abstractmethod
    def initialize(self):
        """Initialize the STT engine with configuration"""
        pass
    
    @abstractmethod
    def transcribe_raw(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe raw audio data
        
        Args:
            audio_data: NumPy array of audio samples
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text
        """
        pass
    
    def normalize_audio(self, audio: bytes) -> bytes:
        """Normalize audio to 16kHz mono WAV format
        
        Args:
            audio: Raw audio bytes in any format
            
        Returns:
            Normalized WAV audio bytes
        """
        out, err = ffmpeg.input('pipe:0') \
            .output('pipe:1', f='WAV', acodec='pcm_s16le', ac=1, ar='16k', loglevel='error', hide_banner=None) \
            .run(input=audio, capture_stdout=True, capture_stderr=True)
        if err:
            raise Exception(f"Audio normalization error: {err}")
        return out
    
    def transcribe(self, audio: bytes) -> str:
        """Transcribe audio from bytes
        
        Args:
            audio: Audio bytes in any format supported by ffmpeg
            
        Returns:
            Transcribed text
        """
        normalized_audio = self.normalize_audio(audio)
        audio_io = BytesIO(normalized_audio)
        
        with wave.Wave_read(audio_io) as wav:
            audio_data = np.frombuffer(wav.readframes(wav.getnframes()), np.int16)
            sample_rate = wav.getframerate()
        
        return self.transcribe_raw(audio_data, sample_rate)
    
    @property
    def name(self) -> str:
        """Return the name of the STT engine"""
        return self.__class__.__name__.replace('Engine', '')
    
    @property
    def is_available(self) -> bool:
        """Check if the engine is available and properly configured"""
        try:
            return self._check_availability()
        except (ImportError, ModuleNotFoundError) as e:
            # Module or dependency not installed
            logger.debug(f"Engine {self.name} dependency not available: {e}")
            return False
        except (FileNotFoundError, OSError) as e:
            # Model file not found or file system error
            logger.debug(f"Engine {self.name} model file error: {e}")
            return False
        except (AttributeError, TypeError, ValueError) as e:
            # Configuration or initialization errors
            logger.debug(f"Engine {self.name} configuration error: {e}")
            return False
        except RuntimeError as e:
            # Runtime errors (e.g., CUDA not available, memory issues)
            logger.debug(f"Engine {self.name} runtime error: {e}")
            return False
        except Exception as e:
            # Catch any other unexpected exceptions
            logger.warning(f"Engine {self.name} unexpected error during availability check: {type(e).__name__}: {e}")
            return False
    
    def _check_availability(self) -> bool:
        """Override this to check if the engine is available"""
        return True