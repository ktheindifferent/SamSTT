from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional
import wave
from io import BytesIO
import numpy as np
import ffmpeg
from .exceptions import (
    AudioProcessingError,
    InvalidAudioError,
    EngineInitializationError
)


class BaseSTTEngine(ABC):
    """Base class for all STT engine implementations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        try:
            self.initialize()
        except EngineInitializationError:
            # Re-raise engine initialization errors as-is
            raise
        except Exception as e:
            # Wrap any other initialization errors
            raise EngineInitializationError(
                f"Failed to initialize {self.name} engine: {str(e)}",
                engine=self.name.lower(),
                original_error=e
            )
    
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
        try:
            out, err = ffmpeg.input('pipe:0') \
                .output('pipe:1', f='WAV', acodec='pcm_s16le', ac=1, ar='16k', loglevel='error', hide_banner=None) \
                .run(input=audio, capture_stdout=True, capture_stderr=True)
            if err:
                # Check for common error patterns
                err_str = err.decode('utf-8') if isinstance(err, bytes) else str(err)
                if 'invalid data' in err_str.lower() or 'could not find' in err_str.lower():
                    raise InvalidAudioError(
                        f"Invalid or corrupted audio data: {err_str}",
                        engine=self.name.lower()
                    )
                raise AudioProcessingError(
                    f"Audio normalization failed: {err_str}",
                    engine=self.name.lower()
                )
            return out
        except ffmpeg.Error as e:
            raise AudioProcessingError(
                f"FFmpeg error during audio normalization: {str(e)}",
                engine=self.name.lower(),
                original_error=e
            )
        except Exception as e:
            if isinstance(e, (AudioProcessingError, InvalidAudioError)):
                raise
            raise AudioProcessingError(
                f"Unexpected error during audio normalization: {str(e)}",
                engine=self.name.lower(),
                original_error=e
            )
    
    def transcribe(self, audio: bytes) -> str:
        """Transcribe audio from bytes
        
        Args:
            audio: Audio bytes in any format supported by ffmpeg
            
        Returns:
            Transcribed text
        """
        # Normalize audio (errors will be raised by normalize_audio)
        normalized_audio = self.normalize_audio(audio)
        
        try:
            audio_io = BytesIO(normalized_audio)
            
            with wave.Wave_read(audio_io) as wav:
                audio_data = np.frombuffer(wav.readframes(wav.getnframes()), np.int16)
                sample_rate = wav.getframerate()
        except wave.Error as e:
            raise AudioProcessingError(
                f"Failed to read normalized WAV data: {str(e)}",
                engine=self.name.lower(),
                original_error=e
            )
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to process audio data: {str(e)}",
                engine=self.name.lower(),
                original_error=e
            )
        
        # Call the engine-specific transcription (errors handled by implementation)
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
        except:
            return False
    
    def _check_availability(self) -> bool:
        """Override this to check if the engine is available"""
        return True