"""
Security and resource configuration for the STT service.
All limits are configurable via environment variables with sensible defaults.
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SecurityConfig:
    """
    Security configuration with validation and bounds checking.
    All values are configurable via environment variables.
    """
    
    # Audio processing limits
    MAX_AUDIO_DURATION = int(os.getenv('MAX_AUDIO_DURATION', 600))  # 10 minutes default (in seconds)
    MIN_AUDIO_DURATION_LIMIT = 10  # Minimum allowed value for MAX_AUDIO_DURATION
    MAX_AUDIO_DURATION_LIMIT = 3600  # Maximum allowed value (1 hour)
    
    # File size limits
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 50 * 1024 * 1024))  # 50MB default
    MIN_FILE_SIZE_LIMIT = 1024  # 1KB minimum
    MAX_FILE_SIZE_LIMIT = 500 * 1024 * 1024  # 500MB maximum
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = int(os.getenv('MAX_REQUESTS_PER_MINUTE', 60))
    MAX_REQUESTS_PER_HOUR = int(os.getenv('MAX_REQUESTS_PER_HOUR', 600))
    MIN_RATE_LIMIT = 1
    MAX_RATE_LIMIT = 10000
    
    # Request handling
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 60))  # 60 seconds default
    MIN_TIMEOUT = 1
    MAX_TIMEOUT = 600  # 10 minutes
    
    # Concurrency limits
    MAX_ENGINE_WORKERS = int(os.getenv('MAX_ENGINE_WORKERS', 2))
    MIN_WORKERS = 1
    MAX_WORKERS = 100
    
    # Audio processing safety
    MAX_WAV_EXPANSION_FACTOR = float(os.getenv('MAX_WAV_EXPANSION_FACTOR', 2.0))  # WAV can be larger than input
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate all configuration values are within acceptable bounds.
        Returns True if all values are valid, False otherwise.
        Logs warnings for values outside recommended ranges.
        """
        valid = True
        
        # Validate audio duration
        if not cls.MIN_AUDIO_DURATION_LIMIT <= cls.MAX_AUDIO_DURATION <= cls.MAX_AUDIO_DURATION_LIMIT:
            logger.error(
                f"MAX_AUDIO_DURATION ({cls.MAX_AUDIO_DURATION}) must be between "
                f"{cls.MIN_AUDIO_DURATION_LIMIT} and {cls.MAX_AUDIO_DURATION_LIMIT} seconds"
            )
            valid = False
        elif cls.MAX_AUDIO_DURATION > 1800:  # 30 minutes
            logger.warning(
                f"MAX_AUDIO_DURATION ({cls.MAX_AUDIO_DURATION}s) is very high. "
                "This may lead to memory issues and slow processing."
            )
        
        # Validate file size
        if not cls.MIN_FILE_SIZE_LIMIT <= cls.MAX_FILE_SIZE <= cls.MAX_FILE_SIZE_LIMIT:
            logger.error(
                f"MAX_FILE_SIZE ({cls.MAX_FILE_SIZE}) must be between "
                f"{cls.MIN_FILE_SIZE_LIMIT} and {cls.MAX_FILE_SIZE_LIMIT} bytes"
            )
            valid = False
        elif cls.MAX_FILE_SIZE > 100 * 1024 * 1024:  # 100MB
            logger.warning(
                f"MAX_FILE_SIZE ({cls.MAX_FILE_SIZE / 1024 / 1024:.1f}MB) is very high. "
                "This may lead to memory issues."
            )
        
        # Validate rate limits
        if not cls.MIN_RATE_LIMIT <= cls.MAX_REQUESTS_PER_MINUTE <= cls.MAX_RATE_LIMIT:
            logger.error(
                f"MAX_REQUESTS_PER_MINUTE ({cls.MAX_REQUESTS_PER_MINUTE}) must be between "
                f"{cls.MIN_RATE_LIMIT} and {cls.MAX_RATE_LIMIT}"
            )
            valid = False
        
        if not cls.MIN_RATE_LIMIT <= cls.MAX_REQUESTS_PER_HOUR <= cls.MAX_RATE_LIMIT:
            logger.error(
                f"MAX_REQUESTS_PER_HOUR ({cls.MAX_REQUESTS_PER_HOUR}) must be between "
                f"{cls.MIN_RATE_LIMIT} and {cls.MAX_RATE_LIMIT}"
            )
            valid = False
        
        # Check rate limit consistency
        if cls.MAX_REQUESTS_PER_HOUR < cls.MAX_REQUESTS_PER_MINUTE:
            logger.error(
                f"MAX_REQUESTS_PER_HOUR ({cls.MAX_REQUESTS_PER_HOUR}) must be >= "
                f"MAX_REQUESTS_PER_MINUTE ({cls.MAX_REQUESTS_PER_MINUTE})"
            )
            valid = False
        
        # Validate timeout
        if not cls.MIN_TIMEOUT <= cls.REQUEST_TIMEOUT <= cls.MAX_TIMEOUT:
            logger.error(
                f"REQUEST_TIMEOUT ({cls.REQUEST_TIMEOUT}) must be between "
                f"{cls.MIN_TIMEOUT} and {cls.MAX_TIMEOUT} seconds"
            )
            valid = False
        
        # Validate workers
        if not cls.MIN_WORKERS <= cls.MAX_ENGINE_WORKERS <= cls.MAX_WORKERS:
            logger.error(
                f"MAX_ENGINE_WORKERS ({cls.MAX_ENGINE_WORKERS}) must be between "
                f"{cls.MIN_WORKERS} and {cls.MAX_WORKERS}"
            )
            valid = False
        elif cls.MAX_ENGINE_WORKERS > 10:
            logger.warning(
                f"MAX_ENGINE_WORKERS ({cls.MAX_ENGINE_WORKERS}) is high. "
                "Ensure your system has sufficient resources."
            )
        
        # Validate WAV expansion factor
        if not 1.0 <= cls.MAX_WAV_EXPANSION_FACTOR <= 10.0:
            logger.error(
                f"MAX_WAV_EXPANSION_FACTOR ({cls.MAX_WAV_EXPANSION_FACTOR}) must be between "
                "1.0 and 10.0"
            )
            valid = False
        
        return valid
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """
        Get a summary of all configuration values.
        Useful for logging and debugging.
        """
        return {
            'audio': {
                'max_duration_seconds': cls.MAX_AUDIO_DURATION,
                'max_file_size_mb': cls.MAX_FILE_SIZE / 1024 / 1024,
                'wav_expansion_factor': cls.MAX_WAV_EXPANSION_FACTOR,
            },
            'rate_limits': {
                'per_minute': cls.MAX_REQUESTS_PER_MINUTE,
                'per_hour': cls.MAX_REQUESTS_PER_HOUR,
            },
            'processing': {
                'request_timeout_seconds': cls.REQUEST_TIMEOUT,
                'max_workers': cls.MAX_ENGINE_WORKERS,
            }
        }
    
    @classmethod
    def apply_safe_defaults(cls):
        """
        Apply safe defaults if validation fails.
        This ensures the service can still run even with invalid config.
        """
        # Apply bounds to audio duration
        if cls.MAX_AUDIO_DURATION < cls.MIN_AUDIO_DURATION_LIMIT:
            logger.warning(f"Adjusting MAX_AUDIO_DURATION from {cls.MAX_AUDIO_DURATION} to {cls.MIN_AUDIO_DURATION_LIMIT}")
            cls.MAX_AUDIO_DURATION = cls.MIN_AUDIO_DURATION_LIMIT
        elif cls.MAX_AUDIO_DURATION > cls.MAX_AUDIO_DURATION_LIMIT:
            logger.warning(f"Adjusting MAX_AUDIO_DURATION from {cls.MAX_AUDIO_DURATION} to {cls.MAX_AUDIO_DURATION_LIMIT}")
            cls.MAX_AUDIO_DURATION = cls.MAX_AUDIO_DURATION_LIMIT
        
        # Apply bounds to file size
        if cls.MAX_FILE_SIZE < cls.MIN_FILE_SIZE_LIMIT:
            logger.warning(f"Adjusting MAX_FILE_SIZE from {cls.MAX_FILE_SIZE} to {cls.MIN_FILE_SIZE_LIMIT}")
            cls.MAX_FILE_SIZE = cls.MIN_FILE_SIZE_LIMIT
        elif cls.MAX_FILE_SIZE > cls.MAX_FILE_SIZE_LIMIT:
            logger.warning(f"Adjusting MAX_FILE_SIZE from {cls.MAX_FILE_SIZE} to {cls.MAX_FILE_SIZE_LIMIT}")
            cls.MAX_FILE_SIZE = cls.MAX_FILE_SIZE_LIMIT
        
        # Apply bounds to rate limits
        cls.MAX_REQUESTS_PER_MINUTE = max(cls.MIN_RATE_LIMIT, min(cls.MAX_RATE_LIMIT, cls.MAX_REQUESTS_PER_MINUTE))
        cls.MAX_REQUESTS_PER_HOUR = max(cls.MIN_RATE_LIMIT, min(cls.MAX_RATE_LIMIT, cls.MAX_REQUESTS_PER_HOUR))
        
        # Ensure hour limit >= minute limit
        if cls.MAX_REQUESTS_PER_HOUR < cls.MAX_REQUESTS_PER_MINUTE:
            cls.MAX_REQUESTS_PER_HOUR = cls.MAX_REQUESTS_PER_MINUTE * 60
            logger.warning(f"Adjusted MAX_REQUESTS_PER_HOUR to {cls.MAX_REQUESTS_PER_HOUR} to match minute limit")
        
        # Apply bounds to timeout
        cls.REQUEST_TIMEOUT = max(cls.MIN_TIMEOUT, min(cls.MAX_TIMEOUT, cls.REQUEST_TIMEOUT))
        
        # Apply bounds to workers
        cls.MAX_ENGINE_WORKERS = max(cls.MIN_WORKERS, min(cls.MAX_WORKERS, cls.MAX_ENGINE_WORKERS))
        
        # Apply bounds to WAV expansion
        cls.MAX_WAV_EXPANSION_FACTOR = max(1.0, min(10.0, cls.MAX_WAV_EXPANSION_FACTOR))


# Initialize and validate configuration on module import
if not SecurityConfig.validate():
    logger.warning("Configuration validation failed. Applying safe defaults.")
    SecurityConfig.apply_safe_defaults()
    # Re-validate after applying defaults
    if not SecurityConfig.validate():
        logger.error("Configuration still invalid after applying defaults!")
else:
    logger.info(f"Security configuration loaded: {SecurityConfig.get_config_summary()}")