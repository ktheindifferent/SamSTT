#!/usr/bin/env python3
"""
Test script for the unified STT service
"""

import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_engine_manager():
    """Test the engine manager directly"""
    from stts.engine_manager import STTEngineManager
    
    print("\n=== Testing Engine Manager ===")
    
    # Initialize manager
    manager = STTEngineManager(default_engine='deepspeech')
    
    # List all registered engines
    all_engines = manager.list_all_engines()
    print(f"All registered engines: {all_engines}")
    
    # List available engines
    available = manager.list_available_engines()
    print(f"Available engines: {available}")
    
    # Get engine info
    info = manager.get_engine_info()
    print(f"\nEngine information:")
    for engine_name, engine_info in info.items():
        print(f"  {engine_name}: available={engine_info['available']}, initialized={engine_info['initialized']}")
    
    return manager


def test_transcription(manager, audio_file='test.mp3'):
    """Test transcription with available engines"""
    print(f"\n=== Testing Transcription with {audio_file} ===")
    
    if not Path(audio_file).exists():
        print(f"Audio file {audio_file} not found, skipping transcription test")
        return
    
    # Read audio file
    with open(audio_file, 'rb') as f:
        audio_data = f.read()
    
    available_engines = manager.list_available_engines()
    
    if not available_engines:
        print("No engines available for transcription")
        return
    
    # Test each available engine
    for engine_name in available_engines[:3]:  # Test up to 3 engines
        try:
            print(f"\nTesting engine: {engine_name}")
            result = manager.transcribe(audio_data, engine=engine_name)
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")


def test_api_endpoints():
    """Test the API endpoints (requires server to be running)"""
    try:
        import requests
    except ImportError:
        print("\nRequests library not installed, skipping API tests")
        return
    
    print("\n=== Testing API Endpoints ===")
    
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print(f"Health check: {response.json()}")
        else:
            print(f"Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("Server not running, skipping API tests")
        return
    
    # Test engines endpoint
    try:
        response = requests.get(f"{base_url}/api/v1/engines")
        if response.status_code == 200:
            engines_info = response.json()
            print(f"Available engines via API: {engines_info['available']}")
            print(f"Default engine: {engines_info['default']}")
    except Exception as e:
        print(f"Error getting engines info: {e}")
    
    # Test transcription if test file exists
    if Path('test.mp3').exists():
        try:
            with open('test.mp3', 'rb') as f:
                files = {'speech': f}
                response = requests.post(f"{base_url}/api/v1/stt", files=files)
                if response.status_code == 200:
                    result = response.json()
                    print(f"Transcription result: {result}")
        except Exception as e:
            print(f"Error during transcription: {e}")


def main():
    """Main test function"""
    print("=" * 50)
    print("Unified STT Service Test Suite")
    print("=" * 50)
    
    # Test engine manager
    manager = test_engine_manager()
    
    # Test transcription if we have engines
    if manager.list_available_engines():
        test_transcription(manager)
    else:
        print("\nNo engines available, skipping transcription tests")
        print("Install at least one engine to test transcription:")
        print("  pip install stt                    # for DeepSpeech")
        print("  pip install openai-whisper         # for Whisper")
        print("  pip install vosk                   # for Vosk")
    
    # Test API if server is running
    test_api_endpoints()
    
    print("\n" + "=" * 50)
    print("Test suite completed")
    print("=" * 50)


if __name__ == "__main__":
    main()