from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional
import wave
from io import BytesIO
import numpy as np
import ffmpeg
import logging
import subprocess
import os
import signal
import threading
import time
from contextlib import contextmanager
from .validators import sanitize_ffmpeg_input, MAX_FILE_SIZE

logger = logging.getLogger(__name__)

# Configuration for FFmpeg process handling
FFMPEG_TIMEOUT = int(os.getenv('FFMPEG_TIMEOUT', 30))  # seconds
FFMPEG_KILL_TIMEOUT = 5  # seconds to wait after SIGTERM before SIGKILL


class ProcessTimeoutError(Exception):
    """Raised when a process exceeds the timeout limit"""
    pass


class FFmpegProcessManager:
    """Manages FFmpeg subprocess lifecycle with proper cleanup"""
    
    # Class-level tracking of active processes
    _active_processes = set()
    _process_lock = threading.Lock()
    
    @classmethod
    def get_active_process_count(cls):
        """Get count of currently active FFmpeg processes"""
        with cls._process_lock:
            # Clean up any terminated processes
            cls._active_processes = {p for p in cls._active_processes if p.poll() is None}
            return len(cls._active_processes)
    
    @classmethod
    def terminate_all_processes(cls):
        """Terminate all active FFmpeg processes (for emergency cleanup)"""
        with cls._process_lock:
            for process in cls._active_processes:
                if process.poll() is None:
                    try:
                        process.terminate()
                        logger.warning(f"Terminated orphaned FFmpeg process PID={process.pid}")
                    except (OSError, ProcessLookupError):
                        pass
            cls._active_processes.clear()
    
    def __init__(self, timeout: int = FFMPEG_TIMEOUT):
        self.timeout = timeout
        self.process = None
        self.timer = None
        self.timed_out = False
        self.lock = threading.Lock()
        self.start_time = None
        self.end_time = None
    
    def _kill_process(self, force=False):
        """Kill the process with optional force"""
        with self.lock:
            if self.process and self.process.poll() is None:
                try:
                    if force:
                        # Force kill with SIGKILL
                        self.process.kill()
                        logger.warning("FFmpeg process forcefully killed (SIGKILL)")
                    else:
                        # Graceful termination with SIGTERM
                        self.process.terminate()
                        logger.info("FFmpeg process terminated (SIGTERM)")
                except (OSError, ProcessLookupError) as e:
                    logger.debug(f"Process already terminated: {e}")
    
    def _timeout_handler(self):
        """Handle process timeout"""
        self.timed_out = True
        logger.error(f"FFmpeg process timed out after {self.timeout} seconds")
        
        # Try graceful termination first
        self._kill_process(force=False)
        
        # Wait a bit for graceful termination
        time.sleep(0.5)
        
        # Force kill if still running
        if self.process and self.process.poll() is None:
            self._kill_process(force=True)
    
    @contextmanager
    def run_process(self, cmd: list, input_data: bytes):
        """Context manager for running FFmpeg process with cleanup
        
        Args:
            cmd: Command and arguments as list
            input_data: Input data to pipe to process
            
        Yields:
            tuple: (stdout, stderr) from the process
            
        Raises:
            ProcessTimeoutError: If process exceeds timeout
            Exception: For other process errors
        """
        stdout_data = None
        stderr_data = None
        
        try:
            self.start_time = time.time()
            
            # Start the process
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # Prevent process from inheriting signals
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Track the process
            with self._process_lock:
                self._active_processes.add(self.process)
            
            logger.debug(f"Started FFmpeg process PID={self.process.pid}, active processes: {self.get_active_process_count()}")
            
            # Start timeout timer
            self.timer = threading.Timer(self.timeout, self._timeout_handler)
            self.timer.start()
            
            # Communicate with the process
            try:
                stdout_data, stderr_data = self.process.communicate(
                    input=input_data,
                    timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                self.timed_out = True
                raise ProcessTimeoutError(
                    f"FFmpeg process timed out after {self.timeout} seconds"
                )
            
            # Check if process was killed due to timeout
            if self.timed_out:
                raise ProcessTimeoutError(
                    f"FFmpeg process was terminated due to timeout ({self.timeout}s)"
                )
            
            # Check return code
            if self.process.returncode != 0:
                error_msg = stderr_data.decode('utf-8', errors='ignore') if stderr_data else "Unknown error"
                raise subprocess.CalledProcessError(
                    self.process.returncode, cmd, output=stdout_data, stderr=stderr_data
                )
            
            yield stdout_data, stderr_data
            
        finally:
            self.end_time = time.time()
            
            # Cancel timer if still active
            if self.timer:
                self.timer.cancel()
            
            # Ensure process is terminated
            if self.process:
                pid = self.process.pid
                
                # Wait a bit for process to finish naturally
                try:
                    self.process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    # Process still running, terminate it
                    self._kill_process(force=False)
                    
                    try:
                        self.process.wait(timeout=FFMPEG_KILL_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        # Still not dead, force kill
                        self._kill_process(force=True)
                        try:
                            self.process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            logger.error(f"Failed to kill FFmpeg process PID={pid}")
                
                # Close all file descriptors
                for stream in [self.process.stdin, self.process.stdout, self.process.stderr]:
                    if stream:
                        try:
                            stream.close()
                        except (OSError, IOError) as e:
                            logger.debug(f"Error closing stream: {e}")
                
                # Remove from tracking
                with self._process_lock:
                    self._active_processes.discard(self.process)
                
                # Log final process state and timing
                if self.process.poll() is not None:
                    duration = self.end_time - self.start_time if self.start_time else 0
                    logger.debug(f"FFmpeg process PID={pid} exited with code {self.process.returncode}, duration: {duration:.2f}s")
                else:
                    logger.warning(f"FFmpeg process PID={pid} may still be running")
                
                logger.debug(f"Active FFmpeg processes remaining: {self.get_active_process_count()}")
                
                # Clear process reference
                self.process = None


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
        """Normalize audio to 16kHz mono WAV format with security checks and proper process cleanup
        
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
        
        # Build FFmpeg command using ffmpeg-python for command construction
        # but execute with our ProcessManager for proper cleanup
        try:
            # Build the FFmpeg command
            stream = (
                ffmpeg
                .input('pipe:0', 
                       threads=1,  # Limit thread usage
                       analyzeduration=100000000,  # 100 seconds max analysis
                       probesize=50000000)  # 50MB max probe size
                .output('pipe:1', 
                        f='WAV',  # Force WAV format
                        acodec='pcm_s16le',  # Force PCM codec
                        ac=1,  # Mono
                        ar='16k',  # 16kHz sample rate
                        loglevel='error',  # Only log errors
                        hide_banner=None,  # Hide banner
                        threads=1)  # Limit thread usage
                .overwrite_output()  # Don't prompt for overwrite
            )
            
            # Get the command as a list
            cmd = stream.compile()
            
            # Execute with our process manager for proper cleanup
            process_manager = FFmpegProcessManager(timeout=FFMPEG_TIMEOUT)
            
            with process_manager.run_process(cmd, sanitized_audio) as (out, err):
                # Process completed successfully
                pass
            
        except ProcessTimeoutError as e:
            logger.error(f"FFmpeg timeout: {e}")
            raise Exception(f"Audio normalization timed out after {FFMPEG_TIMEOUT} seconds")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
            logger.error(f"FFmpeg processing error: {error_msg}")
            raise Exception(f"Audio normalization failed: {error_msg}")
        except Exception as e:
            logger.error(f"Unexpected error during audio normalization: {e}")
            raise Exception(f"Audio normalization failed: {str(e)}")
        
        # Validate output
        if not out:
            raise Exception("Audio normalization produced no output")
        
        if len(out) > MAX_FILE_SIZE * 2:  # Allow some expansion for WAV format
            raise Exception(f"Normalized audio exceeds reasonable size")
        
        # Log any warnings from FFmpeg
        if err:
            err_str = err.decode('utf-8', errors='ignore')
            if err_str and not err_str.isspace():
                logger.warning(f"FFmpeg warnings: {err_str}")
        
        return out
    
    def transcribe(self, audio: bytes) -> str:
        """Transcribe audio from bytes with security validation
        
        Args:
            audio: Audio bytes in any format supported by ffmpeg
            
        Returns:
            Transcribed text
        """
        # Input validation
        if not audio:
            raise ValueError("Empty audio data provided")
        
        if not isinstance(audio, bytes):
            raise TypeError("Audio data must be bytes")
        
        # Normalize audio with security checks
        normalized_audio = self.normalize_audio(audio)
        
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
    
    def get_detailed_config(self) -> dict:
        """Get detailed configuration including available models"""
        config = self.config.copy()
        
        # Add available models if the engine supports it (even if empty)
        if hasattr(self, 'available_models'):
            config['available_models'] = self.available_models
            
        # Add model name if the engine has one
        if hasattr(self, 'model_name'):
            config['model_name'] = self.model_name
            
        return config