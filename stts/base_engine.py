from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional, Tuple
import wave
from io import BytesIO
import numpy as np
import ffmpeg
import logging
import os
from .validators import sanitize_ffmpeg_input, MAX_FILE_SIZE
from .audio_validators import validate_audio_security
from .security.ffmpeg_sandbox import secure_normalize_audio, FFmpegSecurityConfig

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
    
    def normalize_audio(self, audio: bytes, client_id: Optional[str] = None) -> bytes:
        """Normalize audio to 16kHz mono WAV format with comprehensive security checks
        
        Args:
            audio: Raw audio bytes in any format
            client_id: Optional client identifier for rate limiting
            
        Returns:
            Normalized WAV audio bytes
            
        Raises:
            ValueError: If audio validation fails
            Exception: If normalization fails
        """
        # Perform comprehensive security validation
        is_valid, error_msg, metadata = validate_audio_security(audio)
        if not is_valid:
            logger.error(f"Audio security validation failed: {error_msg}")
            raise ValueError(f"Audio security validation failed: {error_msg}")
        
        logger.info(f"Audio validation passed: size={metadata.get('file_size', 0)} bytes, "
                   f"hash={metadata.get('file_hash', 'unknown')[:8]}...")
        
        # Check if we should use legacy mode (for backward compatibility)
        use_legacy_mode = os.getenv('FFMPEG_LEGACY_MODE', 'false').lower() == 'true'
        
        if use_legacy_mode:
            # Legacy mode - use old implementation with basic security
            return self._normalize_audio_legacy(audio)
        
        # Use secure sandboxed FFmpeg execution
        success, normalized_audio, error, stats = secure_normalize_audio(audio, client_id=client_id)
        
        if not success:
            logger.error(f"Secure audio normalization failed: {error}")
            logger.debug(f"Normalization stats: {stats}")
            
            # Check if it's a circuit breaker issue
            if stats.get('circuit_breaker_open'):
                raise Exception("Audio processing temporarily unavailable due to system protection")
            
            raise Exception(f"Audio normalization failed: {error}")
        
        # Log performance stats
        if stats:
            logger.info(f"Audio normalized successfully in {stats.get('execution_time', 0):.2f}s, "
                       f"peak memory: {stats.get('peak_memory_mb', 0):.1f}MB")
        
        # Final validation of output
        if not normalized_audio:
            raise Exception("Audio normalization produced no output")
        
        if len(normalized_audio) > MAX_FILE_SIZE * 2:  # Allow some expansion for WAV format
            raise Exception(f"Normalized audio exceeds reasonable size")
        
        return normalized_audio
    
    def _normalize_audio_legacy(self, audio: bytes) -> bytes:
        """Legacy audio normalization method (less secure, for compatibility)
        
        Args:
            audio: Raw audio bytes in any format
            
        Returns:
            Normalized WAV audio bytes
        """
        # Sanitize input before processing
        try:
            sanitized_audio = sanitize_ffmpeg_input(audio)
        except ValueError as e:
            logger.error(f"Audio sanitization failed: {e}")
            raise
        
        # Check size after sanitization
        if len(sanitized_audio) > MAX_FILE_SIZE:
            raise ValueError(f"Audio file exceeds maximum size of {MAX_FILE_SIZE/1024/1024:.1f}MB")
        
        try:
            # Use FFmpeg with security parameters (but not sandboxed)
            # Get config from environment
            config = FFmpegSecurityConfig()
            
            process = (
                ffmpeg
                .input('pipe:0', 
                       threads=config.max_threads,
                       analyzeduration=config.max_analyzeduration,
                       probesize=config.max_probesize)
                .output('pipe:1', 
                        f='WAV',  # Force WAV format
                        acodec='pcm_s16le',  # Force PCM codec
                        ac=1,  # Mono
                        ar='16k',  # 16kHz sample rate
                        loglevel='error',  # Only log errors
                        hide_banner=None,  # Hide banner
                        threads=config.max_threads,
                        t=config.max_duration_seconds)  # Limit duration
                .overwrite_output()  # Don't prompt for overwrite
            )
            
            out, err = process.run(
                input=sanitized_audio,
                capture_stdout=True,
                capture_stderr=True,
                quiet=True,  # Suppress ffmpeg output to console
                timeout=config.timeout_seconds  # Add timeout
            )
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
            logger.error(f"FFmpeg processing error: {error_msg}")
            raise Exception(f"Audio normalization failed: {error_msg}")
        except Exception as e:
            logger.error(f"Unexpected error during audio normalization: {e}")
            raise Exception(f"Audio normalization failed: {str(e)}")
        
        # Validate output
        if not out:
            raise Exception("Audio normalization produced no output")
        
        if len(out) > config.max_output_size_mb * 1024 * 1024:
            raise Exception(f"Normalized audio exceeds maximum size")
        
        # Log any warnings from FFmpeg
        if err:
            err_str = err.decode('utf-8', errors='ignore')
            if err_str and not err_str.isspace():
                logger.warning(f"FFmpeg warnings: {err_str}")
        
        return out
    
    def transcribe(self, audio: bytes, client_id: Optional[str] = None) -> str:
        """Transcribe audio from bytes with comprehensive security validation
        
        Args:
            audio: Audio bytes in any format supported by ffmpeg
            client_id: Optional client identifier for rate limiting
            
        Returns:
            Transcribed text
        """
        # Input validation
        if not audio:
            raise ValueError("Empty audio data provided")
        
        if not isinstance(audio, bytes):
            raise TypeError("Audio data must be bytes")
        
        # Normalize audio with security checks
        normalized_audio = self.normalize_audio(audio, client_id=client_id)
        
        # Parse normalized WAV
        audio_io = BytesIO(normalized_audio)
        
        try:
            with wave.Wave_read(audio_io) as wav:
                # Validate WAV parameters
                nchannels = wav.getnchannels()
                if nchannels != 1:
                    raise ValueError(f"Expected mono audio, got {nchannels} channels")
                
                framerate = wav.getframerate()
                if framerate != 16000:
                    logger.warning(f"Expected 16kHz audio, got {framerate}Hz")
                
                nframes = wav.getnframes()
                if nframes == 0:
                    raise ValueError("WAV file contains no audio frames")
                
                # Read audio data
                audio_data = np.frombuffer(wav.readframes(nframes), np.int16)
                
                # Validate audio data
                if len(audio_data) == 0:
                    raise ValueError("No audio data after conversion")
                
                # Check for reasonable audio length (e.g., max 10 minutes)
                max_duration = 600  # seconds
                duration = len(audio_data) / framerate
                if duration > max_duration:
                    raise ValueError(f"Audio duration {duration:.1f}s exceeds maximum of {max_duration}s")
                
                sample_rate = framerate
        except wave.Error as e:
            logger.error(f"Failed to read WAV data: {e}")
            raise ValueError(f"Invalid WAV format: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error reading audio: {e}")
            raise
        
        # Transcribe with the engine
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