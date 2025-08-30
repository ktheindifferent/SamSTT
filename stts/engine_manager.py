from typing import Dict, Any, Optional, List
import logging
import threading
import weakref
import time
from pathlib import Path
from .base_engine import BaseSTTEngine
from .engines.deepspeech import DeepSpeechEngine
from .engines.whisper import WhisperEngine
from .engines.coqui import CoquiEngine
from .engines.vosk import VoskEngine
from .engines.silero import SileroEngine
from .engines.wav2vec2 import Wav2Vec2Engine
from .engines.speechbrain import SpeechBrainEngine
from .engines.nemo import NeMoEngine
from .engines.pocketsphinx import PocketSphinxEngine


logger = logging.getLogger(__name__)


class STTEngineManager:
    """Manager for multiple STT engine backends"""
    
    # Registry of available engines
    ENGINES = {
        'deepspeech': DeepSpeechEngine,
        'whisper': WhisperEngine,
        'coqui': CoquiEngine,
        'vosk': VoskEngine,
        'silero': SileroEngine,
        'wav2vec2': Wav2Vec2Engine,
        'speechbrain': SpeechBrainEngine,
        'nemo': NeMoEngine,
        'pocketsphinx': PocketSphinxEngine
    }
    
    def __init__(self, default_engine: str = 'whisper', config: Optional[Dict[str, Any]] = None):
        """Initialize the STT Engine Manager
        
        Args:
            default_engine: Name of the default engine to use
            config: Configuration dict with engine-specific settings
        """
        self.config = config or {}
        self.engines: Dict[str, BaseSTTEngine] = {}
        self.default_engine_name = default_engine
        # Thread-safe initialization locks with automatic cleanup
        # RESOURCE LEAK FIX: Implements automatic cleanup of unused engine locks to prevent
        # memory leaks in long-running deployments. The cleanup mechanism uses:
        # 1. TTL-based cleanup: Locks unused for _lock_cleanup_interval seconds are removed
        # 2. Weak references: Track lock lifecycle without preventing garbage collection
        # 3. Thread-safe cleanup: Safe to clean up while other threads are using locks
        self._engine_locks: Dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()  # Lock for managing the locks dictionary
        self._lock_refs: Dict[str, weakref.ref] = {}  # Weak references to track lock usage
        self._lock_last_used: Dict[str, float] = {}  # Track last usage time for TTL cleanup
        self._lock_cleanup_interval = 300  # Cleanup locks unused for 5 minutes
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all configured engines"""
        # Try to initialize the default engine first
        if self.default_engine_name in self.ENGINES:
            engine_config = self.config.get(self.default_engine_name, {})
            try:
                engine = self.ENGINES[self.default_engine_name](engine_config)
                if engine.is_available:
                    self.engines[self.default_engine_name] = engine
                    logger.info(f"Initialized {self.default_engine_name} as default engine")
                else:
                    logger.warning(f"Default engine {self.default_engine_name} is not available")
            except Exception as e:
                logger.error(f"Failed to initialize default engine {self.default_engine_name}: {e}")
        
        # Try to initialize other configured engines
        for engine_name, engine_class in self.ENGINES.items():
            if engine_name in self.engines:
                continue  # Already initialized
            
            if engine_name in self.config or self.config.get('initialize_all', False):
                engine_config = self.config.get(engine_name, {})
                try:
                    engine = engine_class(engine_config)
                    if engine.is_available:
                        self.engines[engine_name] = engine
                        logger.info(f"Initialized {engine_name} engine")
                except Exception as e:
                    logger.debug(f"Could not initialize {engine_name}: {e}")
    
    def add_engine(self, name: str, engine: BaseSTTEngine):
        """Add a custom engine instance
        
        Args:
            name: Name for the engine
            engine: Engine instance
        """
        self.engines[name] = engine
        logger.info(f"Added custom engine: {name}")
    
    def _cleanup_unused_locks(self):
        """Clean up locks that haven't been used recently
        
        This method implements TTL-based cleanup to prevent the _engine_locks dictionary
        from growing indefinitely. Locks are removed if:
        1. They haven't been used for longer than _lock_cleanup_interval seconds
        2. They are not currently locked (not in use by any thread)
        
        This is thread-safe and can be called while other threads are using locks.
        """
        current_time = time.time()
        locks_to_remove = []
        
        with self._locks_lock:
            for engine_name, last_used in list(self._lock_last_used.items()):
                # Remove locks that haven't been used for the cleanup interval
                # and are not currently in use (not locked)
                if current_time - last_used > self._lock_cleanup_interval:
                    lock = self._engine_locks.get(engine_name)
                    if lock and not lock.locked():
                        locks_to_remove.append(engine_name)
            
            # Remove identified locks
            for engine_name in locks_to_remove:
                if engine_name in self._engine_locks:
                    del self._engine_locks[engine_name]
                if engine_name in self._lock_refs:
                    del self._lock_refs[engine_name]
                if engine_name in self._lock_last_used:
                    del self._lock_last_used[engine_name]
                logger.debug(f"Cleaned up unused lock for engine: {engine_name}")
        
        if locks_to_remove:
            logger.info(f"Cleaned up {len(locks_to_remove)} unused engine locks")
    
    def _get_or_create_lock(self, name: str) -> threading.Lock:
        """Get or create a lock for the given engine name with cleanup tracking
        
        This method ensures thread-safe lazy initialization of engines while preventing
        resource leaks through automatic cleanup. It:
        1. Creates a new lock if one doesn't exist for the engine
        2. Updates the last-used timestamp for TTL tracking
        3. Periodically triggers cleanup to remove stale locks
        
        Args:
            name: Name of the engine to get/create a lock for
            
        Returns:
            Threading lock for the specified engine
        """
        with self._locks_lock:
            if name not in self._engine_locks:
                lock = threading.Lock()
                self._engine_locks[name] = lock
                self._lock_refs[name] = weakref.ref(lock)
                logger.debug(f"Created new lock for engine: {name}")
            
            # Update last used time
            self._lock_last_used[name] = time.time()
            
            # Periodically cleanup unused locks (every 10 engine requests)
            if len(self._engine_locks) > 10 and len(self._engine_locks) % 10 == 0:
                self._cleanup_unused_locks()
            
            return self._engine_locks[name]
    
    def get_engine(self, name: Optional[str] = None) -> BaseSTTEngine:
        """Get a specific engine or the default engine with thread-safe initialization
        
        Args:
            name: Name of the engine to get. If None, returns default engine
            
        Returns:
            The requested engine
            
        Raises:
            ValueError: If the requested engine is not available
        """
        if name is None:
            name = self.default_engine_name
        
        # Double-checked locking pattern for thread-safe lazy initialization
        if name not in self.engines:
            # Get or create lock for this specific engine with cleanup tracking
            engine_lock = self._get_or_create_lock(name)
            
            # Try to initialize on-demand with proper locking
            with engine_lock:
                # Double-check after acquiring lock
                if name not in self.engines:
                    if name in self.ENGINES:
                        engine_config = self.config.get(name, {})
                        logger.info(f"Thread {threading.current_thread().name}: Attempting to initialize {name} engine")
                        try:
                            engine = self.ENGINES[name](engine_config)
                            if engine.is_available:
                                self.engines[name] = engine
                                logger.info(f"Thread {threading.current_thread().name}: Successfully initialized {name} engine on-demand")
                            else:
                                logger.warning(f"Thread {threading.current_thread().name}: Engine {name} is not available")
                                raise ValueError(f"Engine {name} is not available")
                        except Exception as e:
                            logger.error(f"Thread {threading.current_thread().name}: Failed to initialize engine {name}: {e}")
                            raise ValueError(f"Failed to initialize engine {name}: {e}")
                    else:
                        raise ValueError(f"Unknown engine: {name}")
                else:
                    logger.debug(f"Thread {threading.current_thread().name}: Engine {name} already initialized by another thread")
        
        return self.engines[name]
    
    def list_available_engines(self) -> List[str]:
        """List all available and initialized engines"""
        return list(self.engines.keys())
    
    def list_all_engines(self) -> List[str]:
        """List all registered engines (including non-initialized)"""
        return list(self.ENGINES.keys())
    
    def transcribe(self, audio: bytes, engine: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio using specified or default engine
        
        Args:
            audio: Audio bytes to transcribe
            engine: Name of engine to use (optional)
            
        Returns:
            Dict with transcription result and metadata
        """
        stt_engine = self.get_engine(engine)
        
        try:
            text = stt_engine.transcribe(audio)
            return {
                'text': text,
                'engine': stt_engine.name,
                'success': True
            }
        except Exception as e:
            logger.error(f"Transcription failed with {stt_engine.name}: {e}")
            
            # Try fallback engines if available
            for fallback_name, fallback_engine in self.engines.items():
                if fallback_name != (engine or self.default_engine_name):
                    try:
                        text = fallback_engine.transcribe(audio)
                        logger.info(f"Fallback to {fallback_name} succeeded")
                        return {
                            'text': text,
                            'engine': fallback_engine.name,
                            'success': True,
                            'fallback': True
                        }
                    except Exception as fallback_error:
                        logger.error(f"Fallback {fallback_name} also failed: {fallback_error}")
            
            # All engines failed
            raise Exception(f"All STT engines failed. Last error: {e}")
    
    def get_engine_info(self, engine_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about an engine
        
        Args:
            engine_name: Name of engine to get info for
            
        Returns:
            Dict with engine information
        """
        if engine_name:
            if engine_name in self.engines:
                engine = self.engines[engine_name]
                return {
                    'name': engine.name,
                    'available': engine.is_available,
                    'initialized': True,
                    'config': engine.config
                }
            elif engine_name in self.ENGINES:
                return {
                    'name': engine_name,
                    'available': False,
                    'initialized': False,
                    'config': self.config.get(engine_name, {})
                }
            else:
                raise ValueError(f"Unknown engine: {engine_name}")
        else:
            # Return info for all engines
            info = {}
            for name in self.ENGINES:
                if name in self.engines:
                    engine = self.engines[name]
                    info[name] = {
                        'available': engine.is_available,
                        'initialized': True,
                        'config': engine.config
                    }
                else:
                    info[name] = {
                        'available': False,
                        'initialized': False,
                        'config': self.config.get(name, {})
                    }
            return info
    
    def get_lock_stats(self) -> Dict[str, Any]:
        """Get statistics about engine locks (for monitoring/debugging)
        
        Provides visibility into the lock cleanup mechanism's state, useful for:
        - Monitoring lock lifecycle in production
        - Debugging potential lock leaks
        - Verifying cleanup is working correctly
        
        Returns:
            Dict with lock statistics including:
            - total_locks: Number of active locks
            - locks: Per-engine lock details (locked state, age, weak ref status)
        """
        with self._locks_lock:
            current_time = time.time()
            stats = {
                'total_locks': len(self._engine_locks),
                'locks': {}
            }
            
            for engine_name, lock in self._engine_locks.items():
                last_used = self._lock_last_used.get(engine_name, 0)
                age = current_time - last_used if last_used else 0
                
                stats['locks'][engine_name] = {
                    'locked': lock.locked(),
                    'age_seconds': age,
                    'weak_ref_alive': self._lock_refs.get(engine_name, lambda: None)() is not None
                }
            
            return stats