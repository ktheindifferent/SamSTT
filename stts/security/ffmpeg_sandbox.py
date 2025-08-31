"""
FFmpeg security sandboxing and resource control module

This module provides comprehensive security hardening for FFmpeg operations
to prevent DOS attacks through crafted audio files.
"""

import os
import signal
import resource
import subprocess
import threading
import time
import psutil
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager
from queue import Queue, Empty
import tempfile
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class FFmpegSecurityConfig:
    """Configuration for FFmpeg security limits"""
    
    # Resource limits
    max_memory_mb: int = int(os.getenv('FFMPEG_MAX_MEMORY_MB', 512))  # Max memory in MB
    max_cpu_seconds: int = int(os.getenv('FFMPEG_MAX_CPU_SECONDS', 30))  # Max CPU time
    max_output_size_mb: int = int(os.getenv('FFMPEG_MAX_OUTPUT_SIZE_MB', 100))  # Max output size
    max_duration_seconds: int = int(os.getenv('FFMPEG_MAX_DURATION_SECONDS', 600))  # Max audio duration (10 min)
    timeout_seconds: int = int(os.getenv('FFMPEG_TIMEOUT_SECONDS', 10))  # Process timeout
    
    # FFmpeg specific limits
    max_analyzeduration: int = int(os.getenv('FFMPEG_MAX_ANALYZEDURATION', 10000000))  # 10 seconds in microseconds
    max_probesize: int = int(os.getenv('FFMPEG_MAX_PROBESIZE', 10000000))  # 10MB probe size
    max_threads: int = int(os.getenv('FFMPEG_MAX_THREADS', 1))  # Number of threads
    
    # Circuit breaker settings
    circuit_breaker_threshold: int = int(os.getenv('FFMPEG_CIRCUIT_BREAKER_THRESHOLD', 5))  # Failures before opening
    circuit_breaker_reset_time: int = int(os.getenv('FFMPEG_CIRCUIT_BREAKER_RESET_TIME', 60))  # Reset time in seconds
    
    # Rate limiting per client
    max_concurrent_per_client: int = int(os.getenv('FFMPEG_MAX_CONCURRENT_PER_CLIENT', 2))
    max_requests_per_minute: int = int(os.getenv('FFMPEG_MAX_REQUESTS_PER_MINUTE', 10))


class CircuitBreaker:
    """Circuit breaker pattern for FFmpeg operations"""
    
    def __init__(self, threshold: int = 5, reset_time: int = 60):
        self.threshold = threshold
        self.reset_time = reset_time
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self._lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == 'open':
                if time.time() - self.last_failure_time > self.reset_time:
                    self.state = 'half-open'
                    logger.info("Circuit breaker entering half-open state")
                else:
                    raise RuntimeError(f"Circuit breaker is open. Retry after {self.reset_time}s")
            
            try:
                result = func(*args, **kwargs)
                if self.state == 'half-open':
                    self.state = 'closed'
                    self.failure_count = 0
                    logger.info("Circuit breaker closed after successful operation")
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.threshold:
                    self.state = 'open'
                    logger.error(f"Circuit breaker opened after {self.failure_count} failures")
                
                raise e


class ResourceMonitor:
    """Monitor resource usage of a process"""
    
    def __init__(self, pid: int, config: FFmpegSecurityConfig):
        self.pid = pid
        self.config = config
        self.start_time = time.time()
        self.peak_memory = 0
        self.total_cpu = 0
        self._stop_monitoring = threading.Event()
        self._monitor_thread = None
    
    def start(self):
        """Start monitoring in a separate thread"""
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self):
        """Stop monitoring"""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Monitor loop that runs in separate thread"""
        try:
            process = psutil.Process(self.pid)
            
            while not self._stop_monitoring.is_set():
                try:
                    # Check if process is still running
                    if not process.is_running():
                        break
                    
                    # Get memory usage
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    self.peak_memory = max(self.peak_memory, memory_mb)
                    
                    # Check memory limit
                    if memory_mb > self.config.max_memory_mb:
                        logger.error(f"Process {self.pid} exceeded memory limit: {memory_mb:.2f}MB > {self.config.max_memory_mb}MB")
                        process.terminate()
                        time.sleep(0.5)
                        if process.is_running():
                            process.kill()
                        break
                    
                    # Get CPU usage
                    cpu_percent = process.cpu_percent(interval=0.1)
                    self.total_cpu += cpu_percent * 0.1
                    
                    # Check CPU time limit
                    if self.total_cpu > self.config.max_cpu_seconds:
                        logger.error(f"Process {self.pid} exceeded CPU time limit: {self.total_cpu:.2f}s > {self.config.max_cpu_seconds}s")
                        process.terminate()
                        time.sleep(0.5)
                        if process.is_running():
                            process.kill()
                        break
                    
                    # Check runtime limit
                    runtime = time.time() - self.start_time
                    if runtime > self.config.timeout_seconds:
                        logger.error(f"Process {self.pid} exceeded timeout: {runtime:.2f}s > {self.config.timeout_seconds}s")
                        process.terminate()
                        time.sleep(0.5)
                        if process.is_running():
                            process.kill()
                        break
                    
                    time.sleep(0.1)  # Check every 100ms
                    
                except psutil.NoSuchProcess:
                    break
                except Exception as e:
                    logger.warning(f"Error monitoring process {self.pid}: {e}")
                    break
                    
        except psutil.NoSuchProcess:
            pass  # Process already terminated
        except Exception as e:
            logger.error(f"Fatal error in resource monitor: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics"""
        return {
            'peak_memory_mb': self.peak_memory,
            'total_cpu_seconds': self.total_cpu,
            'runtime_seconds': time.time() - self.start_time
        }


class FFmpegSandbox:
    """Secure FFmpeg execution sandbox"""
    
    def __init__(self, config: Optional[FFmpegSecurityConfig] = None):
        self.config = config or FFmpegSecurityConfig()
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold,
            reset_time=self.config.circuit_breaker_reset_time
        )
        self._client_counters = {}  # Track per-client requests
        self._client_lock = threading.Lock()
    
    def validate_audio_metadata(self, audio_bytes: bytes) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Validate audio metadata before processing
        
        Returns:
            Tuple of (is_valid, error_message, metadata)
        """
        metadata = {}
        
        # Check basic size limits
        size_mb = len(audio_bytes) / (1024 * 1024)
        if size_mb > self.config.max_output_size_mb:
            return False, f"Input size {size_mb:.2f}MB exceeds limit {self.config.max_output_size_mb}MB", metadata
        
        metadata['input_size_mb'] = size_mb
        
        # Quick probe to get duration and format info (with timeout)
        try:
            import ffmpeg
            
            # Create temporary file for probe (safer than pipe for probe)
            with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False) as tmp:
                tmp.write(audio_bytes[:min(len(audio_bytes), 1024*1024)])  # Write first 1MB for probe
                tmp_path = tmp.name
            
            try:
                # Probe with strict limits
                probe_result = ffmpeg.probe(
                    tmp_path,
                    cmd='ffprobe',
                    timeout=2,  # 2 second timeout for probe
                    v='error'
                )
                
                # Extract duration if available
                if 'format' in probe_result and 'duration' in probe_result['format']:
                    duration = float(probe_result['format']['duration'])
                    metadata['duration_seconds'] = duration
                    
                    if duration > self.config.max_duration_seconds:
                        return False, f"Audio duration {duration:.1f}s exceeds limit {self.config.max_duration_seconds}s", metadata
                
                # Extract format info
                if 'format' in probe_result:
                    metadata['format_name'] = probe_result['format'].get('format_name', 'unknown')
                    metadata['bit_rate'] = probe_result['format'].get('bit_rate', 0)
                
                # Extract stream info
                if 'streams' in probe_result:
                    for stream in probe_result['streams']:
                        if stream.get('codec_type') == 'audio':
                            metadata['codec_name'] = stream.get('codec_name', 'unknown')
                            metadata['sample_rate'] = stream.get('sample_rate', 0)
                            metadata['channels'] = stream.get('channels', 0)
                            break
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            return False, "Audio probe timeout - potentially malicious file", metadata
        except Exception as e:
            logger.warning(f"Could not probe audio metadata: {e}")
            # Continue without metadata - will rely on processing limits
        
        return True, None, metadata
    
    def check_client_limits(self, client_id: str) -> Tuple[bool, Optional[str]]:
        """Check per-client rate limits
        
        Returns:
            Tuple of (is_allowed, error_message)
        """
        with self._client_lock:
            now = time.time()
            
            # Clean old entries
            for cid in list(self._client_counters.keys()):
                self._client_counters[cid] = [
                    t for t in self._client_counters.get(cid, [])
                    if now - t < 60  # Keep last minute
                ]
                if not self._client_counters[cid]:
                    del self._client_counters[cid]
            
            # Check client's recent requests
            client_requests = self._client_counters.get(client_id, [])
            
            if len(client_requests) >= self.config.max_requests_per_minute:
                return False, f"Client exceeded {self.config.max_requests_per_minute} requests per minute"
            
            # Count concurrent requests (requests started in last timeout period)
            concurrent = sum(1 for t in client_requests if now - t < self.config.timeout_seconds)
            if concurrent >= self.config.max_concurrent_per_client:
                return False, f"Client has {concurrent} concurrent requests (max {self.config.max_concurrent_per_client})"
            
            # Record this request
            client_requests.append(now)
            self._client_counters[client_id] = client_requests
            
            return True, None
    
    def execute_ffmpeg_secure(self, input_data: bytes, output_format: str = 'wav',
                            additional_args: Optional[Dict[str, Any]] = None,
                            client_id: Optional[str] = None) -> Tuple[bool, Optional[bytes], Optional[str], Dict[str, Any]]:
        """Execute FFmpeg with comprehensive security controls
        
        Args:
            input_data: Input audio bytes
            output_format: Output format (e.g., 'wav')
            additional_args: Additional FFmpeg arguments
            client_id: Client identifier for rate limiting
            
        Returns:
            Tuple of (success, output_bytes, error_message, stats)
        """
        stats = {'start_time': time.time()}
        
        # Check client limits
        if client_id:
            allowed, error = self.check_client_limits(client_id)
            if not allowed:
                return False, None, error, stats
        
        # Validate input
        valid, error, metadata = self.validate_audio_metadata(input_data)
        if not valid:
            stats['validation_error'] = error
            return False, None, error, stats
        
        stats['metadata'] = metadata
        
        # Build secure FFmpeg command
        try:
            import ffmpeg
            
            # Build input stream with security limits
            stream = ffmpeg.input(
                'pipe:0',
                threads=self.config.max_threads,
                analyzeduration=self.config.max_analyzeduration,
                probesize=self.config.max_probesize,
                fflags='+discardcorrupt',  # Discard corrupt packets
                err_detect='aggressive',  # Aggressive error detection
                max_delay=1000000  # Max 1 second delay
            )
            
            # Apply output settings
            output_args = {
                'f': output_format,
                'threads': self.config.max_threads,
                'loglevel': 'error',
                'hide_banner': None,
                't': self.config.max_duration_seconds  # Limit output duration
            }
            
            # Add user-specified args (filtered for safety)
            if additional_args:
                safe_args = self._filter_safe_args(additional_args)
                output_args.update(safe_args)
            
            stream = stream.output('pipe:1', **output_args)
            
            # Compile command
            cmd = stream.compile()
            
        except Exception as e:
            return False, None, f"Failed to build FFmpeg command: {e}", stats
        
        # Execute with circuit breaker
        try:
            success, output, error, exec_stats = self.circuit_breaker.call(
                self._execute_with_limits,
                cmd, input_data
            )
            stats.update(exec_stats)
            stats['execution_time'] = time.time() - stats['start_time']
            
            if success:
                # Validate output size
                if output and len(output) > self.config.max_output_size_mb * 1024 * 1024:
                    return False, None, "Output size exceeds limit", stats
                
                return True, output, None, stats
            else:
                return False, None, error, stats
                
        except RuntimeError as e:
            # Circuit breaker is open
            stats['circuit_breaker_open'] = True
            return False, None, str(e), stats
        except Exception as e:
            logger.error(f"FFmpeg execution failed: {e}")
            return False, None, f"FFmpeg execution failed: {e}", stats
    
    def _filter_safe_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Filter FFmpeg arguments for safety"""
        # Whitelist of safe output arguments
        safe_keys = {
            'ac', 'ar', 'acodec', 'ab', 'aq',  # Audio settings
            'vn',  # No video
            'sn',  # No subtitles
            'dn',  # No data streams
            'map_metadata',  # Metadata handling
        }
        
        filtered = {}
        for key, value in args.items():
            if key in safe_keys:
                filtered[key] = value
            else:
                logger.warning(f"Filtered potentially unsafe FFmpeg argument: {key}")
        
        return filtered
    
    def _execute_with_limits(self, cmd: list, input_data: bytes) -> Tuple[bool, Optional[bytes], Optional[str], Dict[str, Any]]:
        """Execute FFmpeg with resource limits and monitoring"""
        stats = {}
        monitor = None
        process = None
        
        try:
            # Set resource limits for the subprocess
            def set_limits():
                if hasattr(resource, 'RLIMIT_AS'):
                    # Virtual memory limit
                    resource.setrlimit(resource.RLIMIT_AS, 
                                     (self.config.max_memory_mb * 1024 * 1024,
                                      self.config.max_memory_mb * 1024 * 1024))
                
                if hasattr(resource, 'RLIMIT_CPU'):
                    # CPU time limit
                    resource.setrlimit(resource.RLIMIT_CPU,
                                     (self.config.max_cpu_seconds,
                                      self.config.max_cpu_seconds))
                
                # Set nice value to lower priority
                os.nice(10)
            
            # Start FFmpeg process
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=set_limits if os.name != 'nt' else None,  # Unix only
                # Additional security on Linux
                **({'user': 'nobody'} if os.name != 'nt' and os.getuid() == 0 else {})
            )
            
            # Start resource monitor
            monitor = ResourceMonitor(process.pid, self.config)
            monitor.start()
            
            # Execute with timeout
            try:
                stdout, stderr = process.communicate(
                    input=input_data,
                    timeout=self.config.timeout_seconds
                )
                
                # Get monitoring stats
                if monitor:
                    stats.update(monitor.get_stats())
                
                # Check return code
                if process.returncode != 0:
                    error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown error"
                    logger.error(f"FFmpeg failed with code {process.returncode}: {error_msg}")
                    return False, None, error_msg, stats
                
                return True, stdout, None, stats
                
            except subprocess.TimeoutExpired:
                logger.error(f"FFmpeg process timeout after {self.config.timeout_seconds}s")
                process.kill()
                process.wait(timeout=1)
                return False, None, f"Process timeout after {self.config.timeout_seconds}s", stats
                
        except Exception as e:
            logger.error(f"FFmpeg execution error: {e}")
            return False, None, str(e), stats
            
        finally:
            # Clean up
            if monitor:
                monitor.stop()
            
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=1)
                except:
                    process.kill()


# Global sandbox instance
_sandbox = None
_sandbox_lock = threading.Lock()


def get_sandbox(config: Optional[FFmpegSecurityConfig] = None) -> FFmpegSandbox:
    """Get or create global FFmpeg sandbox instance"""
    global _sandbox
    
    with _sandbox_lock:
        if _sandbox is None:
            _sandbox = FFmpegSandbox(config)
        return _sandbox


def secure_normalize_audio(audio_bytes: bytes, client_id: Optional[str] = None) -> Tuple[bool, Optional[bytes], Optional[str], Dict[str, Any]]:
    """Convenience function for secure audio normalization
    
    Args:
        audio_bytes: Input audio bytes
        client_id: Client identifier for rate limiting
        
    Returns:
        Tuple of (success, normalized_audio, error_message, stats)
    """
    sandbox = get_sandbox()
    
    # FFmpeg args for normalization to 16kHz mono WAV
    additional_args = {
        'acodec': 'pcm_s16le',
        'ac': 1,
        'ar': '16k',
        'vn': None  # No video
    }
    
    return sandbox.execute_ffmpeg_secure(
        audio_bytes,
        output_format='WAV',
        additional_args=additional_args,
        client_id=client_id
    )


@contextmanager
def ffmpeg_security_context(config: Optional[FFmpegSecurityConfig] = None):
    """Context manager for FFmpeg security settings"""
    sandbox = FFmpegSandbox(config)
    try:
        yield sandbox
    finally:
        # Any cleanup if needed
        pass