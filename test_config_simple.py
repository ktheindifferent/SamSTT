#!/usr/bin/env python3
"""
Simple test to verify configuration is working
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Configuration Test Results")
print("=" * 50)

# Test with custom environment variable
os.environ['MAX_AUDIO_DURATION'] = '300'
print("\n1. Setting MAX_AUDIO_DURATION=300 via environment")

from stts.config import SecurityConfig
print(f"   SecurityConfig.MAX_AUDIO_DURATION = {SecurityConfig.MAX_AUDIO_DURATION}")

if SecurityConfig.MAX_AUDIO_DURATION == 300:
    print("   ✅ Environment variable override working!")
else:
    print(f"   ❌ Expected 300, got {SecurityConfig.MAX_AUDIO_DURATION}")

# Test configuration summary
print("\n2. Configuration Summary:")
summary = SecurityConfig.get_config_summary()
for category, settings in summary.items():
    print(f"\n   {category}:")
    for key, value in settings.items():
        print(f"     - {key}: {value}")

# Test that base_engine uses the config
print("\n3. Testing base_engine integration:")
from stts.base_engine import BaseSTTEngine

class DummyEngine(BaseSTTEngine):
    def initialize(self):
        pass
    def transcribe_raw(self, audio_data, sample_rate=16000):
        return "dummy"

try:
    engine = DummyEngine()
    print("   ✅ Base engine created successfully")
    
    # The engine should use SecurityConfig internally
    # We can't directly test the duration limit without creating audio,
    # but we can verify the module imported correctly
    import stts.base_engine
    if hasattr(stts.base_engine, 'SecurityConfig'):
        print("   ✅ Base engine has SecurityConfig imported")
    else:
        print("   ❌ Base engine missing SecurityConfig import")
        
except Exception as e:
    print(f"   ❌ Error creating engine: {e}")

# Test validators backward compatibility
print("\n4. Testing backward compatibility:")
from stts.validators import MAX_FILE_SIZE, MAX_REQUESTS_PER_MINUTE

print(f"   validators.MAX_FILE_SIZE = {MAX_FILE_SIZE}")
print(f"   SecurityConfig.MAX_FILE_SIZE = {SecurityConfig.MAX_FILE_SIZE}")

if MAX_FILE_SIZE == SecurityConfig.MAX_FILE_SIZE:
    print("   ✅ Validators exports match SecurityConfig")
else:
    print("   ❌ Mismatch between validators and SecurityConfig")

print("\n" + "=" * 50)
print("Configuration test complete!")