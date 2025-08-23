"""Custom exception hierarchy for the Unified STT Service

This module provides a consistent exception hierarchy for all STT engines,
enabling better error handling, debugging, and fallback mechanisms.
"""


class STTException(Exception):
    """Base exception for all STT-related errors
    
    Attributes:
        message: Error message
        engine: Name of the engine that raised the error (optional)
        original_error: The original exception that was caught (optional)
    """
    
    def __init__(self, message: str, engine: str = None, original_error: Exception = None):
        self.message = message
        self.engine = engine
        self.original_error = original_error
        super().__init__(self.message)
    
    def __str__(self):
        if self.engine:
            return f"[{self.engine}] {self.message}"
        return self.message


class EngineNotAvailableError(STTException):
    """Raised when an STT engine is not available or cannot be imported"""
    pass


class EngineInitializationError(STTException):
    """Raised when an STT engine fails to initialize
    
    This includes model loading failures, configuration errors, etc.
    """
    pass


class ModelNotFoundError(EngineInitializationError):
    """Raised when a required model file is not found
    
    Attributes:
        model_path: Path to the missing model file
    """
    
    def __init__(self, message: str, model_path: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.model_path = model_path


class AudioProcessingError(STTException):
    """Raised when audio preprocessing or normalization fails
    
    This includes format conversion errors, resampling errors, etc.
    """
    pass


class TranscriptionError(STTException):
    """Raised when the actual transcription process fails
    
    This is for errors during the model inference/transcription step.
    """
    pass


class InvalidAudioError(AudioProcessingError):
    """Raised when the provided audio data is invalid or corrupted"""
    pass


class UnsupportedAudioFormatError(AudioProcessingError):
    """Raised when the audio format is not supported by the engine"""
    pass


class EngineTimeoutError(TranscriptionError):
    """Raised when transcription times out"""
    
    def __init__(self, message: str, timeout: float = None, **kwargs):
        super().__init__(message, **kwargs)
        self.timeout = timeout


class InsufficientResourcesError(STTException):
    """Raised when there are insufficient resources (memory, GPU, etc.)"""
    pass


class ConfigurationError(STTException):
    """Raised when there's an error in engine configuration"""
    pass