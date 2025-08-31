"""Configuration manager for centralized config handling with proper resource management"""
import json
import logging
import os
from contextlib import closing
from pathlib import Path
from typing import Dict, Any, Optional
from threading import Lock
import time

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading with proper file handle management and caching"""
    
    _instance = None
    _lock = Lock()
    _cache: Dict[str, Dict[str, Any]] = {}
    _cache_times: Dict[str, float] = {}
    _cache_ttl = 300  # 5 minutes cache TTL
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the configuration manager"""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._file_handle_count = 0
            self._monitor_lock = Lock()
    
    def _increment_file_handle_count(self):
        """Increment file handle counter for monitoring"""
        with self._monitor_lock:
            self._file_handle_count += 1
            if self._file_handle_count > 100:
                logger.warning(f"High file handle count detected: {self._file_handle_count}")
    
    def _decrement_file_handle_count(self):
        """Decrement file handle counter for monitoring"""
        with self._monitor_lock:
            self._file_handle_count -= 1
    
    def get_file_handle_count(self) -> int:
        """Get current file handle count for monitoring"""
        with self._monitor_lock:
            return self._file_handle_count
    
    def clear_cache(self, path: Optional[str] = None):
        """Clear configuration cache
        
        Args:
            path: Optional specific path to clear, otherwise clears all
        """
        with self._lock:
            if path:
                self._cache.pop(str(path), None)
                self._cache_times.pop(str(path), None)
            else:
                self._cache.clear()
                self._cache_times.clear()
    
    def _is_cache_valid(self, path: str) -> bool:
        """Check if cached config is still valid
        
        Args:
            path: Path to check cache for
            
        Returns:
            True if cache is valid, False otherwise
        """
        if path not in self._cache_times:
            return False
        
        elapsed = time.time() - self._cache_times[path]
        return elapsed < self._cache_ttl
    
    def load_json_config(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """Load JSON configuration with proper resource management
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary or None if loading fails
        """
        path_str = str(config_path)
        
        # Check cache first
        with self._lock:
            if path_str in self._cache and self._is_cache_valid(path_str):
                logger.debug(f"Using cached configuration for {path_str}")
                return self._cache[path_str].copy()
        
        if not config_path.exists():
            logger.debug(f"Configuration file does not exist: {config_path}")
            return None
        
        if not config_path.is_file():
            logger.warning(f"Configuration path is not a file: {config_path}")
            return None
        
        # Read file content first, then parse JSON separately
        file_content = None
        config = None
        
        try:
            # Step 1: Read file content with guaranteed cleanup
            self._increment_file_handle_count()
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            finally:
                self._decrement_file_handle_count()
            
            # Step 2: Parse JSON (no file handle involved)
            if file_content:
                config = json.loads(file_content)
                
                # Cache the successful load
                with self._lock:
                    self._cache[path_str] = config.copy()
                    self._cache_times[path_str] = time.time()
                
                logger.info(f"Successfully loaded configuration from {config_path}")
                return config
                
        except PermissionError as e:
            logger.error(f"Permission denied reading configuration file {config_path}: {e}")
        except OSError as e:
            logger.error(f"OS error reading configuration file {config_path}: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file {config_path}: {e}")
            logger.debug(f"File content that failed to parse: {file_content[:200] if file_content else 'None'}...")
        except Exception as e:
            logger.error(f"Unexpected error loading configuration from {config_path}: {type(e).__name__}: {e}")
        
        return None
    
    def load_config(self, config: Optional[Dict[str, Any]] = None, 
                   config_file: Optional[str] = None,
                   default_engine: Optional[str] = None) -> tuple[Dict[str, Any], str]:
        """Load configuration from various sources with fallback
        
        Args:
            config: Pre-loaded configuration dictionary
            config_file: Path to configuration file
            default_engine: Default engine name
            
        Returns:
            Tuple of (configuration dictionary, engine name)
        """
        engine_name = default_engine or os.getenv('STT_ENGINE', 'deepspeech')
        
        # If config is already provided, use it
        if config is not None:
            if 'default_engine' in config:
                engine_name = config.pop('default_engine')
            return config, engine_name
        
        # Try to load from specified config file
        if config_file:
            config_path = Path(config_file)
            loaded_config = self.load_json_config(config_path)
            if loaded_config:
                if 'default_engine' in loaded_config:
                    engine_name = loaded_config.pop('default_engine')
                return loaded_config, engine_name
            else:
                logger.warning(f"Could not load specified config file: {config_file}")
        
        # Try default config locations
        default_config_paths = [
            Path('/app/config.json'),
            Path('./config.json'),
        ]
        
        # Add environment variable path if set
        env_config = os.getenv('STT_CONFIG_FILE')
        if env_config:
            default_config_paths.insert(0, Path(env_config))
        
        for config_path in default_config_paths:
            if config_path:
                loaded_config = self.load_json_config(config_path)
                if loaded_config:
                    if 'default_engine' in loaded_config:
                        engine_name = loaded_config.pop('default_engine')
                    return loaded_config, engine_name
        
        # No config found, return empty config
        logger.debug("No configuration file found, using default configuration")
        return {}, engine_name
    
    def build_default_config(self) -> Dict[str, Any]:
        """Build default configuration from environment and file paths
        
        Returns:
            Default configuration dictionary
        """
        config = {}
        
        # DeepSpeech configuration
        deepspeech_model_paths = [
            Path(__file__).parents[1].joinpath('model.tflite'),
            Path(__file__).parents[1].joinpath('model.pbmm'),
            Path('/app/model.pbmm'),
            Path('/app/model.tflite')
        ]
        for path in deepspeech_model_paths:
            if path.exists():
                config['deepspeech'] = {'model_path': str(path.absolute())}
                break
        
        # Whisper configuration
        config['whisper'] = {
            'model_size': os.getenv('WHISPER_MODEL_SIZE', 'base'),
            'device': os.getenv('WHISPER_DEVICE', 'cpu'),
            'language': os.getenv('WHISPER_LANGUAGE', None)
        }
        
        # Coqui configuration
        coqui_model_paths = [
            Path(__file__).parents[1].joinpath('coqui_model.tflite'),
            Path(__file__).parents[1].joinpath('coqui_model.pbmm'),
            Path('/app/coqui_model.tflite'),
            Path('/app/coqui_model.pbmm')
        ]
        for path in coqui_model_paths:
            if path.exists():
                config['coqui'] = {'model_path': str(path.absolute())}
                break
        
        # Vosk configuration
        vosk_model_path = os.getenv('VOSK_MODEL_PATH')
        if vosk_model_path:
            config['vosk'] = {'model_path': vosk_model_path}
        
        # Silero configuration
        config['silero'] = {
            'language': os.getenv('SILERO_LANGUAGE', 'en'),
            'device': os.getenv('SILERO_DEVICE', 'cpu')
        }
        
        # Wav2Vec2 configuration
        config['wav2vec2'] = {
            'model_name': os.getenv('WAV2VEC2_MODEL', 'facebook/wav2vec2-base-960h'),
            'device': os.getenv('WAV2VEC2_DEVICE', 'cpu')
        }
        
        return config


# Global instance
_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the singleton ConfigManager instance
    
    Returns:
        ConfigManager instance
    """
    return _config_manager