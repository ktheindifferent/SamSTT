#!/usr/bin/env python3
"""
Integration test for configurable security limits
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test 1: Verify default configuration
print("Test 1: Default configuration values")
from stts.config import SecurityConfig

print(f"  MAX_AUDIO_DURATION: {SecurityConfig.MAX_AUDIO_DURATION}s (default: 600s)")
print(f"  MAX_FILE_SIZE: {SecurityConfig.MAX_FILE_SIZE / 1024 / 1024:.1f}MB (default: 50MB)")
print(f"  MAX_REQUESTS_PER_MINUTE: {SecurityConfig.MAX_REQUESTS_PER_MINUTE} (default: 60)")
print(f"  MAX_ENGINE_WORKERS: {SecurityConfig.MAX_ENGINE_WORKERS} (default: 2)")
assert SecurityConfig.MAX_AUDIO_DURATION == 600, "Default audio duration should be 600s"
print("  ✅ Default values correct\n")

# Test 2: Verify environment variable override
print("Test 2: Environment variable override")
os.environ['MAX_AUDIO_DURATION'] = '300'
os.environ['MAX_FILE_SIZE'] = '104857600'  # 100MB
os.environ['MAX_ENGINE_WORKERS'] = '4'

# Reload the module
import importlib
import stts.config
importlib.reload(stts.config)
from stts.config import SecurityConfig as ReloadedConfig

print(f"  MAX_AUDIO_DURATION: {ReloadedConfig.MAX_AUDIO_DURATION}s (expected: 300s)")
print(f"  MAX_FILE_SIZE: {ReloadedConfig.MAX_FILE_SIZE / 1024 / 1024:.1f}MB (expected: 100MB)")
print(f"  MAX_ENGINE_WORKERS: {ReloadedConfig.MAX_ENGINE_WORKERS} (expected: 4)")

assert ReloadedConfig.MAX_AUDIO_DURATION == 300, "Audio duration override failed"
assert ReloadedConfig.MAX_FILE_SIZE == 104857600, "File size override failed"
assert ReloadedConfig.MAX_ENGINE_WORKERS == 4, "Worker count override failed"
print("  ✅ Environment overrides working\n")

# Test 3: Verify bounds checking
print("Test 3: Bounds checking and safe defaults")

# Set invalid values
os.environ['MAX_AUDIO_DURATION'] = '5'  # Too low
os.environ['MAX_FILE_SIZE'] = '100'  # Too low
os.environ['MAX_REQUESTS_PER_MINUTE'] = '0'  # Invalid
os.environ['MAX_ENGINE_WORKERS'] = '500'  # Too high

# Reload again
importlib.reload(stts.config)
from stts.config import SecurityConfig as BoundedConfig

print(f"  MAX_AUDIO_DURATION after bounds: {BoundedConfig.MAX_AUDIO_DURATION}s")
print(f"  MAX_FILE_SIZE after bounds: {BoundedConfig.MAX_FILE_SIZE} bytes")
print(f"  MAX_REQUESTS_PER_MINUTE after bounds: {BoundedConfig.MAX_REQUESTS_PER_MINUTE}")
print(f"  MAX_ENGINE_WORKERS after bounds: {BoundedConfig.MAX_ENGINE_WORKERS}")

# Verify bounds are applied
assert BoundedConfig.MAX_AUDIO_DURATION >= BoundedConfig.MIN_AUDIO_DURATION_LIMIT
assert BoundedConfig.MAX_FILE_SIZE >= BoundedConfig.MIN_FILE_SIZE_LIMIT
assert BoundedConfig.MAX_REQUESTS_PER_MINUTE >= BoundedConfig.MIN_RATE_LIMIT
assert BoundedConfig.MAX_ENGINE_WORKERS <= BoundedConfig.MAX_WORKERS

print("  ✅ Bounds checking working\n")

# Test 4: Verify backward compatibility
print("Test 4: Backward compatibility")

# Reset to normal values
os.environ['MAX_FILE_SIZE'] = '52428800'  # 50MB
os.environ['MAX_REQUESTS_PER_MINUTE'] = '60'
os.environ['MAX_REQUESTS_PER_HOUR'] = '600'

# Reload validators module
import stts.validators
importlib.reload(stts.validators)
from stts.validators import MAX_FILE_SIZE, MAX_REQUESTS_PER_MINUTE, MAX_REQUESTS_PER_HOUR

print(f"  validators.MAX_FILE_SIZE: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
print(f"  validators.MAX_REQUESTS_PER_MINUTE: {MAX_REQUESTS_PER_MINUTE}")
print(f"  validators.MAX_REQUESTS_PER_HOUR: {MAX_REQUESTS_PER_HOUR}")

assert MAX_FILE_SIZE == 52428800, "Backward compat for MAX_FILE_SIZE failed"
assert MAX_REQUESTS_PER_MINUTE == 60, "Backward compat for rate limit failed"
print("  ✅ Backward compatibility maintained\n")

# Test 5: Verify configuration summary
print("Test 5: Configuration summary")
summary = BoundedConfig.get_config_summary()
print(f"  Summary structure: {list(summary.keys())}")
assert 'audio' in summary
assert 'rate_limits' in summary
assert 'processing' in summary
print("  ✅ Configuration summary working\n")

print("=" * 50)
print("✅ All integration tests passed!")
print("=" * 50)