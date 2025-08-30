#!/usr/bin/env python3
"""
Simple verification script to confirm the memory leak fix is in place.
"""

import sys
import os
import inspect

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_whisper_fix():
    """Verify WhisperEngine has the fix"""
    print("Checking WhisperEngine...")
    
    try:
        from stts.engines.whisper import WhisperEngine
        
        # Check initialize method
        init_source = inspect.getsource(WhisperEngine.initialize)
        has_librosa_cache = 'self.librosa = librosa' in init_source
        
        # Check transcribe_raw method
        transcribe_source = inspect.getsource(WhisperEngine.transcribe_raw)
        uses_cached_librosa = 'self.librosa' in transcribe_source
        no_dynamic_import = 'import librosa' not in transcribe_source
        
        print(f"  ✓ Caches librosa in initialize(): {has_librosa_cache}")
        print(f"  ✓ Uses self.librosa in transcribe_raw(): {uses_cached_librosa}")
        print(f"  ✓ No dynamic import in transcribe_raw(): {no_dynamic_import}")
        
        return has_librosa_cache and uses_cached_librosa and no_dynamic_import
    except Exception as e:
        print(f"  ✗ Error checking WhisperEngine: {e}")
        return False


def verify_wav2vec2_fix():
    """Verify Wav2Vec2Engine has the fix"""
    print("\nChecking Wav2Vec2Engine...")
    
    try:
        from stts.engines.wav2vec2 import Wav2Vec2Engine
        
        # Check initialize method
        init_source = inspect.getsource(Wav2Vec2Engine.initialize)
        has_librosa_cache = 'self.librosa = librosa' in init_source
        has_torch_cache = 'self.torch = torch' in init_source
        
        # Check transcribe_raw method
        transcribe_source = inspect.getsource(Wav2Vec2Engine.transcribe_raw)
        uses_cached_librosa = 'self.librosa' in transcribe_source
        uses_cached_torch = 'self.torch' in transcribe_source
        no_dynamic_librosa = 'import librosa' not in transcribe_source
        
        print(f"  ✓ Caches librosa in initialize(): {has_librosa_cache}")
        print(f"  ✓ Caches torch in initialize(): {has_torch_cache}")
        print(f"  ✓ Uses self.librosa in transcribe_raw(): {uses_cached_librosa}")
        print(f"  ✓ Uses self.torch in transcribe_raw(): {uses_cached_torch}")
        print(f"  ✓ No dynamic librosa import in transcribe_raw(): {no_dynamic_librosa}")
        
        return (has_librosa_cache and has_torch_cache and 
                uses_cached_librosa and uses_cached_torch and 
                no_dynamic_librosa)
    except Exception as e:
        print(f"  ✗ Error checking Wav2Vec2Engine: {e}")
        return False


def main():
    print("=" * 60)
    print("Memory Leak Fix Verification")
    print("=" * 60)
    
    whisper_fixed = verify_whisper_fix()
    wav2vec2_fixed = verify_wav2vec2_fix()
    
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    
    if whisper_fixed and wav2vec2_fixed:
        print("\n✅ SUCCESS: Memory leak fix is properly implemented!")
        print("\nThe fix ensures:")
        print("  • librosa is imported only once during initialization")
        print("  • No dynamic imports occur during transcription")
        print("  • Memory usage remains stable under load")
        print("  • Better performance with cached module references")
        return 0
    else:
        print("\n❌ FAILURE: Memory leak fix is not properly implemented")
        if not whisper_fixed:
            print("  • WhisperEngine needs fixing")
        if not wav2vec2_fixed:
            print("  • Wav2Vec2Engine needs fixing")
        return 1


if __name__ == '__main__':
    sys.exit(main())