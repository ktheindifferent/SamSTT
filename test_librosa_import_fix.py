#!/usr/bin/env python3
"""
Focused test to verify the librosa memory leak fix in WhisperEngine and Wav2Vec2Engine.

This test directly tests the engines without complex mocking.
"""

import sys
import os
import gc
import unittest
from unittest.mock import Mock, patch, MagicMock
import tracemalloc

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestLibrosaImportFix(unittest.TestCase):
    """Test that librosa is imported only once during initialization"""
    
    def test_whisper_engine_structure(self):
        """Test that WhisperEngine has the correct structure after our fix"""
        from stts.engines.whisper import WhisperEngine
        
        # Create mock config
        config = {'model_size': 'tiny', 'device': 'cpu'}
        
        # Create engine instance
        engine = WhisperEngine(config)
        
        # Mock whisper module to avoid actual model loading
        with patch('stts.engines.whisper.whisper') as mock_whisper:
            mock_model = Mock()
            mock_whisper.load_model.return_value = mock_model
            
            # Initialize the engine
            engine.initialize()
            
            # Verify that librosa is stored as an instance attribute
            self.assertTrue(hasattr(engine, 'librosa'), 
                          "WhisperEngine should have 'librosa' attribute after initialization")
            
            # Verify the transcribe_raw method uses self.librosa
            # Read the source to ensure it's using self.librosa
            import inspect
            source = inspect.getsource(engine.transcribe_raw)
            
            # Check that the method uses self.librosa instead of importing
            self.assertIn('self.librosa', source, 
                        "transcribe_raw should use self.librosa")
            self.assertNotIn('import librosa', source, 
                           "transcribe_raw should not import librosa dynamically")
    
    def test_wav2vec2_engine_structure(self):
        """Test that Wav2Vec2Engine has the correct structure after our fix"""
        from stts.engines.wav2vec2 import Wav2Vec2Engine
        
        # Create mock config
        config = {'model_name': 'facebook/wav2vec2-base-960h', 'device': 'cpu'}
        
        # Create engine instance
        engine = Wav2Vec2Engine(config)
        
        # Mock dependencies
        with patch('stts.engines.wav2vec2.transformers') as mock_transformers, \
             patch('stts.engines.wav2vec2.torch') as mock_torch:
            
            # Setup mocks
            mock_processor = Mock()
            mock_transformers.Wav2Vec2Processor.from_pretrained.return_value = mock_processor
            
            mock_model = Mock()
            mock_transformers.Wav2Vec2ForCTC.from_pretrained.return_value = mock_model
            
            # Initialize the engine
            engine.initialize()
            
            # Verify that librosa is stored as an instance attribute
            self.assertTrue(hasattr(engine, 'librosa'), 
                          "Wav2Vec2Engine should have 'librosa' attribute after initialization")
            
            # Verify that torch is stored as an instance attribute
            self.assertTrue(hasattr(engine, 'torch'), 
                          "Wav2Vec2Engine should have 'torch' attribute after initialization")
            
            # Verify the transcribe_raw method uses self.librosa and self.torch
            import inspect
            source = inspect.getsource(engine.transcribe_raw)
            
            # Check that the method uses self.librosa instead of importing
            self.assertIn('self.librosa', source, 
                        "transcribe_raw should use self.librosa")
            self.assertIn('self.torch', source, 
                        "transcribe_raw should use self.torch")
            self.assertNotIn('import librosa', source, 
                           "transcribe_raw should not import librosa dynamically")
            self.assertNotIn('import torch', source, 
                           "transcribe_raw should not import torch dynamically (except as part of self.torch)")
    
    def test_import_tracking(self):
        """Test that multiple transcriptions don't cause repeated imports"""
        import numpy as np
        
        # Track imports using sys.modules
        initial_modules = set(sys.modules.keys())
        
        # Create test with WhisperEngine
        from stts.engines.whisper import WhisperEngine
        
        with patch('stts.engines.whisper.whisper') as mock_whisper:
            mock_model = Mock()
            mock_model.transcribe.return_value = {'text': 'test'}
            mock_whisper.load_model.return_value = mock_model
            
            # Initialize engine
            engine = WhisperEngine({})
            engine.initialize()
            
            # Get modules after initialization
            after_init_modules = set(sys.modules.keys())
            
            # Mock librosa if it was imported
            if engine.librosa:
                engine.librosa.resample = Mock(return_value=np.zeros(16000))
            
            # Perform multiple transcriptions
            audio_data = np.random.randint(-32768, 32767, 48000, dtype=np.int16)
            
            for i in range(3):
                try:
                    engine.transcribe_raw(audio_data, 48000)
                except Exception:
                    pass
            
            # Get modules after transcriptions
            after_transcribe_modules = set(sys.modules.keys())
            
            # No new modules should be imported during transcription
            # (they should all be imported during initialization)
            new_modules = after_transcribe_modules - after_init_modules
            
            # Filter out any test-related modules
            new_modules = {m for m in new_modules if not m.startswith('_')}
            
            self.assertEqual(len(new_modules), 0,
                           f"New modules imported during transcription: {new_modules}")
    
    def test_memory_leak_pattern(self):
        """Test memory allocation patterns to detect potential leaks"""
        import numpy as np
        
        # Start memory tracking
        tracemalloc.start()
        
        from stts.engines.whisper import WhisperEngine
        
        with patch('stts.engines.whisper.whisper') as mock_whisper:
            mock_model = Mock()
            mock_model.transcribe.return_value = {'text': 'test'}
            mock_whisper.load_model.return_value = mock_model
            
            # Initialize engine
            engine = WhisperEngine({})
            engine.initialize()
            
            # Mock librosa if available
            if engine.librosa:
                engine.librosa.resample = Mock(return_value=np.zeros(16000))
            
            # Take baseline snapshot after initialization
            gc.collect()
            baseline = tracemalloc.take_snapshot()
            
            # Perform transcriptions
            audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
            
            for i in range(10):
                try:
                    result = engine.transcribe_raw(audio_data, 16000)
                except Exception:
                    pass
                
                if i % 5 == 0:
                    gc.collect()
            
            # Take final snapshot
            gc.collect()
            final = tracemalloc.take_snapshot()
            
            # Compare snapshots
            top_stats = final.compare_to(baseline, 'lineno')
            
            # Check for suspicious growth patterns
            # We're looking for lines in our code that allocate memory repeatedly
            suspicious_growth = []
            for stat in top_stats[:20]:
                if stat.size_diff > 100000:  # More than 100KB growth
                    if 'whisper.py' in str(stat) or 'wav2vec2.py' in str(stat):
                        suspicious_growth.append(stat)
            
            # There should be no significant memory growth in our engine files
            self.assertEqual(len(suspicious_growth), 0,
                           f"Suspicious memory growth detected: {suspicious_growth}")
            
            tracemalloc.stop()


class TestRegressionPrevention(unittest.TestCase):
    """Test to ensure the fix doesn't break existing functionality"""
    
    def test_whisper_basic_functionality(self):
        """Test that WhisperEngine still works correctly after the fix"""
        import numpy as np
        from stts.engines.whisper import WhisperEngine
        
        with patch('stts.engines.whisper.whisper') as mock_whisper:
            mock_model = Mock()
            mock_model.transcribe.return_value = {'text': 'Hello world'}
            mock_whisper.load_model.return_value = mock_model
            
            # Test with default config
            engine = WhisperEngine({})
            engine.initialize()
            
            # Test transcription at native sample rate
            audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
            result = engine.transcribe_raw(audio_data, 16000)
            self.assertEqual(result, 'Hello world')
            
            # Test with resampling (if librosa available)
            if engine.librosa:
                engine.librosa.resample = Mock(return_value=np.zeros(16000, dtype=np.float32))
                audio_data_48k = np.random.randint(-32768, 32767, 48000, dtype=np.int16)
                result = engine.transcribe_raw(audio_data_48k, 48000)
                self.assertEqual(result, 'Hello world')
                engine.librosa.resample.assert_called_once()
            else:
                # If librosa not available, should raise error for resampling
                audio_data_48k = np.random.randint(-32768, 32767, 48000, dtype=np.int16)
                with self.assertRaises(RuntimeError) as context:
                    engine.transcribe_raw(audio_data_48k, 48000)
                self.assertIn('librosa not available', str(context.exception))
    
    def test_wav2vec2_basic_functionality(self):
        """Test that Wav2Vec2Engine still works correctly after the fix"""
        import numpy as np
        from stts.engines.wav2vec2 import Wav2Vec2Engine
        
        with patch('stts.engines.wav2vec2.transformers') as mock_transformers:
            # Setup mocks
            mock_processor = Mock()
            mock_processor.batch_decode.return_value = ['Hello world']
            mock_processor.return_value = {'input_values': Mock()}
            mock_transformers.Wav2Vec2Processor.from_pretrained.return_value = mock_processor
            
            mock_model = Mock()
            mock_logits = Mock()
            mock_logits.logits = Mock()
            mock_model.return_value = mock_logits
            mock_transformers.Wav2Vec2ForCTC.from_pretrained.return_value = mock_model
            
            # Initialize engine - torch will be imported and stored
            engine = Wav2Vec2Engine({})
            engine.initialize()
            
            # Mock the stored torch
            if hasattr(engine, 'torch'):
                engine.torch.no_grad = Mock()
                engine.torch.no_grad.__enter__ = Mock()
                engine.torch.no_grad.__exit__ = Mock()
                engine.torch.argmax = Mock(return_value=Mock())
                engine.torch.device = Mock(return_value='cpu')
            
            # Test transcription at native sample rate
            audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
            result = engine.transcribe_raw(audio_data, 16000)
            self.assertEqual(result, 'Hello world')


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)