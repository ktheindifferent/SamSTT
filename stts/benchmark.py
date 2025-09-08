"""
Benchmark module for testing STT engine performance
"""
import time
import logging
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import json
import os

logger = logging.getLogger(__name__)


class STTBenchmark:
    """Benchmark STT engines on startup"""
    
    def __init__(self, engine_manager):
        self.engine_manager = engine_manager
        self.results = {}
        self.benchmark_audio = None
        
    def generate_test_audio(self, duration: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
        """Generate a test audio signal (sine wave with noise)"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Generate a 440Hz sine wave (A note) with some noise
        audio = np.sin(2 * np.pi * 440 * t) * 0.3
        audio += np.random.normal(0, 0.01, audio.shape)  # Add some noise
        # Convert to int16 format
        audio = (audio * 32767).astype(np.int16)
        return audio
    
    def load_test_audio(self) -> Optional[np.ndarray]:
        """Try to load a real test audio file if available"""
        test_files = [
            Path('/app/test.wav'),
            Path('/app/test.mp3'),
            Path(__file__).parent.parent / 'test.wav',
            Path(__file__).parent.parent / 'test.mp3',
        ]
        
        for test_file in test_files:
            if test_file.exists():
                try:
                    # Use the engine's normalize_audio method to load the file
                    # We'll use the first available engine for this
                    engines = self.engine_manager.list_engines()
                    if engines:
                        engine = self.engine_manager.get_engine(engines[0])
                        if engine:
                            audio, sr = engine.normalize_audio(str(test_file))
                            return audio
                except Exception as e:
                    logger.warning(f"Failed to load test file {test_file}: {e}")
        
        # If no test file found, generate synthetic audio
        logger.info("No test audio file found, using generated audio for benchmark")
        return self.generate_test_audio()
    
    def benchmark_engine(self, engine_name: str, audio: np.ndarray, warmup: bool = True) -> Dict[str, Any]:
        """Benchmark a single engine"""
        result = {
            'available': False,
            'error': None,
            'init_time': None,
            'transcribe_time': None,
            'total_time': None,
            'transcript': None
        }
        
        try:
            # Measure initialization time
            start_init = time.time()
            engine = self.engine_manager.get_engine(engine_name)
            if not engine:
                result['error'] = f"Engine {engine_name} not available"
                return result
            init_time = time.time() - start_init
            result['init_time'] = round(init_time, 3)
            
            # Warmup run (first run is often slower)
            if warmup:
                try:
                    _ = engine.transcribe_raw(audio, sample_rate=16000)
                except:
                    pass  # Ignore warmup errors
            
            # Measure transcription time (average of 3 runs)
            transcribe_times = []
            transcript = None
            
            for _ in range(3):
                start_transcribe = time.time()
                try:
                    transcript = engine.transcribe_raw(audio, sample_rate=16000)
                    transcribe_time = time.time() - start_transcribe
                    transcribe_times.append(transcribe_time)
                except Exception as e:
                    logger.warning(f"Transcription failed for {engine_name}: {e}")
                    break
            
            if transcribe_times:
                avg_transcribe_time = sum(transcribe_times) / len(transcribe_times)
                result['transcribe_time'] = round(avg_transcribe_time, 3)
                result['total_time'] = round(init_time + avg_transcribe_time, 3)
                result['transcript'] = transcript[:50] if transcript else None  # First 50 chars
                result['available'] = True
            else:
                result['error'] = "Transcription failed"
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Benchmark failed for {engine_name}: {e}")
        
        return result
    
    def run_benchmarks(self, engines: Optional[list] = None) -> Dict[str, Any]:
        """Run benchmarks on all available engines"""
        logger.info("Starting STT engine benchmarks...")
        
        # Load or generate test audio
        self.benchmark_audio = self.load_test_audio()
        if self.benchmark_audio is None:
            logger.error("Failed to create benchmark audio")
            return {}
        
        # Get list of engines to benchmark
        if engines is None:
            engines = self.engine_manager.list_engines()
        
        logger.info(f"Benchmarking {len(engines)} engines...")
        
        # Run benchmarks
        for engine_name in engines:
            logger.info(f"Benchmarking {engine_name}...")
            self.results[engine_name] = self.benchmark_engine(engine_name, self.benchmark_audio)
        
        # Find fastest engine
        fastest_engine = None
        fastest_time = float('inf')
        
        for engine_name, result in self.results.items():
            if result['available'] and result['transcribe_time']:
                if result['transcribe_time'] < fastest_time:
                    fastest_time = result['transcribe_time']
                    fastest_engine = engine_name
        
        # Summary
        summary = {
            'engines_tested': len(engines),
            'engines_available': sum(1 for r in self.results.values() if r['available']),
            'fastest_engine': fastest_engine,
            'fastest_time': round(fastest_time, 3) if fastest_engine else None,
            'results': self.results
        }
        
        logger.info(f"Benchmark complete. Fastest engine: {fastest_engine} ({fastest_time:.3f}s)" if fastest_engine else "No engines available")
        
        # Save results to file if path is specified
        benchmark_file = os.environ.get('BENCHMARK_RESULTS_FILE', '/app/benchmark_results.json')
        try:
            with open(benchmark_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Benchmark results saved to {benchmark_file}")
        except Exception as e:
            logger.warning(f"Failed to save benchmark results: {e}")
        
        return summary
    
    def get_results(self) -> Dict[str, Any]:
        """Get the latest benchmark results"""
        return self.results
    
    def get_fastest_engine(self) -> Optional[str]:
        """Get the name of the fastest engine from benchmarks"""
        fastest_engine = None
        fastest_time = float('inf')
        
        for engine_name, result in self.results.items():
            if result.get('available') and result.get('transcribe_time'):
                if result['transcribe_time'] < fastest_time:
                    fastest_time = result['transcribe_time']
                    fastest_engine = engine_name
        
        return fastest_engine