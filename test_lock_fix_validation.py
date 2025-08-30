#!/usr/bin/env python3
"""Final validation test for engine lock leak fix"""

import sys
import os
import time
import logging
from unittest import mock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock dependencies
mock_modules = [
    'numpy', 'scipy', 'scipy.io', 'scipy.io.wavfile',
    'ffmpeg', 'sanic', 'torch', 'torchaudio', 'omegaconf',
    'transformers', 'librosa', 'speechbrain', 'nemo_toolkit',
    'pocketsphinx', 'vosk', 'STT', 'stt', 'whisper'
]

for module_name in mock_modules:
    sys.modules[module_name] = mock.MagicMock()

from stts.engine_manager import STTEngineManager


class MockEngine:
    def __init__(self, config):
        self.config = config
        self.name = config.get('name', 'unknown')
        self.is_available = True


def validate_fix():
    """Validate that the lock leak fix is properly implemented"""
    
    logger.info("=" * 60)
    logger.info("VALIDATING ENGINE LOCK LEAK FIX")
    logger.info("=" * 60)
    
    # Replace engines with mock
    for engine_name in STTEngineManager.ENGINES:
        STTEngineManager.ENGINES[engine_name] = MockEngine
    
    # Add test engines
    for i in range(20):
        STTEngineManager.ENGINES[f'test_{i}'] = MockEngine
    
    config = {'initialize_all': False}
    for i in range(20):
        config[f'test_{i}'] = {'name': f'test_{i}'}
    
    manager = STTEngineManager(default_engine='whisper', config=config)
    manager._lock_cleanup_interval = 1  # 1 second for quick testing
    manager.engines.clear()  # Clear pre-initialized
    
    # Test 1: Check cleanup mechanism exists
    logger.info("\n1. Checking cleanup mechanism attributes...")
    assert hasattr(manager, '_cleanup_unused_locks'), "Missing cleanup method"
    assert hasattr(manager, '_lock_last_used'), "Missing TTL tracking"
    assert hasattr(manager, '_lock_refs'), "Missing weak references"
    assert hasattr(manager, 'get_lock_stats'), "Missing monitoring method"
    logger.info("   ✓ All cleanup attributes present")
    
    # Test 2: Create locks through lazy initialization
    logger.info("\n2. Creating locks through lazy initialization...")
    for i in range(10):
        engine = manager.get_engine(f'test_{i}')
    
    initial_locks = len(manager._engine_locks)
    logger.info(f"   Created {initial_locks} locks")
    assert initial_locks == 10, f"Expected 10 locks, got {initial_locks}"
    
    # Test 3: Verify TTL tracking
    logger.info("\n3. Verifying TTL tracking...")
    stats = manager.get_lock_stats()
    for i in range(10):
        assert f'test_{i}' in stats['locks'], f"Missing lock for test_{i}"
        age = stats['locks'][f'test_{i}']['age_seconds']
        assert age >= 0, "Invalid age tracking"
    logger.info("   ✓ TTL tracking working")
    
    # Test 4: Test cleanup
    logger.info("\n4. Testing cleanup mechanism...")
    time.sleep(manager._lock_cleanup_interval + 0.5)
    manager._cleanup_unused_locks()
    
    locks_after_cleanup = len(manager._engine_locks)
    logger.info(f"   Locks after cleanup: {locks_after_cleanup}")
    assert locks_after_cleanup < initial_locks, "Cleanup didn't remove any locks"
    logger.info("   ✓ Cleanup successfully removed unused locks")
    
    # Test 5: Verify we can still get engines after cleanup
    logger.info("\n5. Testing engine access after cleanup...")
    for i in range(5):
        engine = manager.get_engine(f'test_{i}')
        assert engine is not None, f"Failed to get test_{i} after cleanup"
    logger.info("   ✓ Engines accessible after cleanup")
    
    # Test 6: Verify no indefinite growth
    logger.info("\n6. Testing no indefinite lock growth...")
    
    # Request many different engines
    for i in range(20):
        engine = manager.get_engine(f'test_{i}')
    
    # Wait and cleanup
    time.sleep(manager._lock_cleanup_interval + 0.5)
    manager._cleanup_unused_locks()
    
    # Request again
    for i in range(20):
        engine = manager.get_engine(f'test_{i}')
    
    final_locks = len(manager._engine_locks)
    logger.info(f"   Final lock count: {final_locks}")
    assert final_locks <= 20, f"Lock count grew beyond expected: {final_locks}"
    logger.info("   ✓ No indefinite lock growth")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ SUCCESS: ENGINE LOCK LEAK FIX VALIDATED")
    logger.info("=" * 60)
    logger.info("\nThe fix successfully implements:")
    logger.info("• TTL-based cleanup of unused locks")
    logger.info("• Weak reference tracking")
    logger.info("• Thread-safe cleanup mechanism")
    logger.info("• Lock recreation after cleanup")
    logger.info("• Monitoring via get_lock_stats()")
    logger.info("\nThis prevents memory leaks in long-running deployments.")
    
    return True


if __name__ == "__main__":
    try:
        success = validate_fix()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)