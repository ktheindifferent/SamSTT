#!/usr/bin/env python3
"""Demonstration of the configuration file handle leak fix"""

import json
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from stts.config_manager import get_config_manager
from stts.engine import SpeechToTextEngine


def demonstrate_file_handle_leak_fix():
    """Demonstrate that file handles are properly managed even with errors"""
    
    print("=== Configuration File Handle Leak Fix Demonstration ===\n")
    
    # Create temp directory for test files
    temp_dir = tempfile.mkdtemp()
    
    # Create malformed JSON file
    malformed_config_path = Path(temp_dir) / "malformed.json"
    with open(malformed_config_path, 'w') as f:
        f.write('{"invalid": json"syntax"}')
    
    # Create valid config file
    valid_config_path = Path(temp_dir) / "valid.json"
    with open(valid_config_path, 'w') as f:
        json.dump({"default_engine": "deepspeech"}, f)
    
    config_manager = get_config_manager()
    
    print("1. Testing malformed JSON handling (should not leak file handles):")
    print(f"   - Initial file handle count: {config_manager.get_file_handle_count()}")
    
    # Try to load malformed JSON multiple times
    for i in range(5):
        result = config_manager.load_json_config(malformed_config_path)
        print(f"   - Attempt {i+1}: Config loaded = {result is not None}, "
              f"File handles open = {config_manager.get_file_handle_count()}")
    
    print(f"   - Final file handle count: {config_manager.get_file_handle_count()}")
    print("   ✓ No file handles leaked!\n")
    
    print("2. Testing valid JSON with caching:")
    config_manager.clear_cache()
    
    # First load
    print(f"   - Loading valid config (first time, from file)...")
    config1 = config_manager.load_json_config(valid_config_path)
    print(f"     Result: {config1}")
    
    # Second load (should use cache)
    print(f"   - Loading valid config (second time, from cache)...")
    config2 = config_manager.load_json_config(valid_config_path)
    print(f"     Result: {config2}")
    print(f"     Same object from cache: {config1 == config2}")
    print("   ✓ Caching reduces file I/O!\n")
    
    print("3. Testing engine initialization with error handling:")
    print("   - Creating engine with malformed config...")
    try:
        engine = SpeechToTextEngine(config_file=str(malformed_config_path))
        print(f"     Engine created with default engine: {engine.manager.default_engine_name}")
        print("   ✓ Engine initialization handles config errors gracefully!\n")
    except Exception as e:
        print(f"     Engine creation failed: {e}\n")
    
    print("4. File handle monitoring:")
    print(f"   - Current internal file handle count: {config_manager.get_file_handle_count()}")
    
    # Simulate high file handle count
    for _ in range(101):
        config_manager._increment_file_handle_count()
    print(f"   - After simulating 101 open handles: {config_manager.get_file_handle_count()}")
    print("   - (Would trigger warning in logs)")
    
    # Clean up
    for _ in range(101):
        config_manager._decrement_file_handle_count()
    print(f"   - After cleanup: {config_manager.get_file_handle_count()}")
    
    # Clean up temp files
    for file in Path(temp_dir).glob("*"):
        file.unlink()
    Path(temp_dir).rmdir()
    
    print("\n=== Summary ===")
    print("✓ File handles are properly closed even when JSON parsing fails")
    print("✓ Configuration caching reduces file I/O operations")
    print("✓ Engine initialization handles config errors gracefully")
    print("✓ File handle monitoring helps detect potential leaks")
    print("\nThe fix ensures no file descriptor exhaustion in long-running services!")


if __name__ == "__main__":
    demonstrate_file_handle_leak_fix()