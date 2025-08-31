"""
Input validation and security utilities for the STT service
"""
import os
import hashlib
import time
from typing import Optional, Dict, Tuple, Deque
from collections import defaultdict, deque
from threading import Lock, RLock
import logging
import mimetypes
from io import BytesIO

logger = logging.getLogger(__name__)

# Configuration from environment variables
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 50 * 1024 * 1024))  # 50MB default
MAX_REQUESTS_PER_MINUTE = int(os.getenv('MAX_REQUESTS_PER_MINUTE', 60))
MAX_REQUESTS_PER_HOUR = int(os.getenv('MAX_REQUESTS_PER_HOUR', 600))
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 60))  # 60 seconds default

# Allowed MIME types for audio files
ALLOWED_MIME_TYPES = {
    'audio/wav',
    'audio/x-wav',
    'audio/wave',
    'audio/mpeg',
    'audio/mp3',
    'audio/mp4',
    'audio/ogg',
    'audio/flac',
    'audio/x-flac',
    'audio/webm',
    'audio/aac',
    'audio/m4a',
    'audio/x-m4a',
    'audio/opus',
    'audio/vorbis',
    'audio/speex',
    'audio/amr',
    'audio/3gpp',
    'audio/3gpp2',
    'application/octet-stream',  # Generic binary, will validate with magic numbers
}

# Magic numbers for common audio formats (first few bytes)
AUDIO_MAGIC_NUMBERS = {
    b'RIFF': 'wav',  # WAV files start with RIFF
    b'ID3': 'mp3',   # MP3 with ID3 tag
    b'\xff\xfb': 'mp3',  # MP3 without ID3
    b'\xff\xf3': 'mp3',  # MP3 without ID3
    b'\xff\xf2': 'mp3',  # MP3 without ID3
    b'fLaC': 'flac',  # FLAC
    b'OggS': 'ogg',   # OGG
    b'\x00\x00\x00\x20ftypM4A': 'm4a',  # M4A
    b'\x00\x00\x00\x18ftyp': 'mp4',  # MP4
    b'\x1a\x45\xdf\xa3': 'webm',  # WebM
    b'#!AMR': 'amr',  # AMR
    b'#!AMR-WB': 'amr',  # AMR-WB
}


class RateLimiter:
    """Thread-safe rate limiter using sliding window algorithm with deque for efficiency
    
    This implementation uses:
    - RLock for reentrant locking to prevent deadlocks
    - deque for efficient O(1) append/popleft operations
    - Atomic in-place operations to prevent race conditions
    - Proper cleanup of old entries to prevent memory leaks
    """
    
    def __init__(self):
        # Use deque instead of list for O(1) append/popleft operations
        # defaultdict ensures thread-safe initialization of new client entries
        self.requests: Dict[str, Deque[float]] = defaultdict(deque)
        # Use RLock for reentrant locking (safer for complex scenarios)
        self.lock = RLock()
        # Track cleanup times to avoid excessive cleanup operations
        self.last_cleanup: Dict[str, float] = {}
        self.cleanup_interval = 60.0  # Cleanup old entries every 60 seconds
    
    def is_allowed(self, client_id: str) -> Tuple[bool, Optional[str]]:
        """Check if request from client is allowed based on rate limits
        
        Thread-safe implementation with atomic operations.
        
        Args:
            client_id: Unique identifier for the client (IP address)
            
        Returns:
            Tuple of (is_allowed, error_message)
        """
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            hour_ago = now - 3600
            
            # Get or create deque for this client
            client_requests = self.requests[client_id]
            
            # Cleanup old entries if needed (with optimization to avoid excessive cleanup)
            last_cleanup = self.last_cleanup.get(client_id, 0)
            if now - last_cleanup > self.cleanup_interval:
                # Remove entries older than 1 hour using atomic in-place operation
                # This avoids creating a new list and potential race conditions
                while client_requests and client_requests[0] <= hour_ago:
                    client_requests.popleft()
                self.last_cleanup[client_id] = now
            
            # Count requests in different time windows
            # Use generator expressions for memory efficiency
            minute_requests = sum(1 for t in client_requests if t > minute_ago)
            hour_requests = len(client_requests)
            
            # Check rate limits
            if minute_requests >= MAX_REQUESTS_PER_MINUTE:
                logger.warning(f"Rate limit exceeded for {client_id}: {minute_requests} requests per minute")
                return False, f"Rate limit exceeded: {MAX_REQUESTS_PER_MINUTE} requests per minute"
            
            if hour_requests >= MAX_REQUESTS_PER_HOUR:
                logger.warning(f"Rate limit exceeded for {client_id}: {hour_requests} requests per hour")
                return False, f"Rate limit exceeded: {MAX_REQUESTS_PER_HOUR} requests per hour"
            
            # Add current request timestamp
            # Using append on deque is atomic and thread-safe within the lock
            client_requests.append(now)
            
            # Optional: Clean up clients with no recent requests to prevent memory leaks
            # This is done periodically, not on every request
            if len(self.requests) > 1000 and now % 300 < 1:  # Every ~5 minutes
                self._cleanup_inactive_clients(hour_ago)
            
            return True, None
    
    def _cleanup_inactive_clients(self, cutoff_time: float):
        """Remove clients with no requests after cutoff time
        
        This method must be called within a lock context.
        
        Args:
            cutoff_time: Remove clients with no requests after this time
        """
        # Create list of clients to remove (can't modify dict during iteration)
        clients_to_remove = [
            client_id for client_id, requests in self.requests.items()
            if not requests or requests[-1] <= cutoff_time
        ]
        
        # Remove inactive clients
        for client_id in clients_to_remove:
            del self.requests[client_id]
            self.last_cleanup.pop(client_id, None)
        
        if clients_to_remove:
            logger.debug(f"Cleaned up {len(clients_to_remove)} inactive clients")
    
    def reset(self, client_id: Optional[str] = None):
        """Reset rate limit counters for a client or all clients
        
        Thread-safe reset operation.
        
        Args:
            client_id: Client to reset, or None to reset all
        """
        with self.lock:
            if client_id:
                # Clear specific client's request history
                if client_id in self.requests:
                    self.requests[client_id].clear()
                self.last_cleanup.pop(client_id, None)
            else:
                # Clear all clients
                self.requests.clear()
                self.last_cleanup.clear()
    
    def get_stats(self, client_id: str) -> Dict[str, int]:
        """Get current request stats for a client
        
        Thread-safe stats retrieval.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dictionary with minute_requests and hour_requests counts
        """
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            client_requests = self.requests.get(client_id, deque())
            
            minute_requests = sum(1 for t in client_requests if t > minute_ago)
            hour_requests = len(client_requests)
            
            return {
                'minute_requests': minute_requests,
                'hour_requests': hour_requests,
                'minute_remaining': MAX_REQUESTS_PER_MINUTE - minute_requests,
                'hour_remaining': MAX_REQUESTS_PER_HOUR - hour_requests
            }


# Global rate limiter instance
rate_limiter = RateLimiter()


def validate_file_size(file_bytes: bytes) -> Tuple[bool, Optional[str]]:
    """Validate that file size is within limits
    
    Args:
        file_bytes: The file content as bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    file_size = len(file_bytes)
    
    if file_size == 0:
        return False, "File is empty"
    
    if file_size > MAX_FILE_SIZE:
        size_mb = file_size / (1024 * 1024)
        max_mb = MAX_FILE_SIZE / (1024 * 1024)
        return False, f"File size {size_mb:.2f}MB exceeds maximum allowed size of {max_mb:.2f}MB"
    
    return True, None


def validate_mime_type(file_bytes: bytes, filename: Optional[str] = None, 
                      content_type: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Validate MIME type and file format
    
    Args:
        file_bytes: The file content as bytes
        filename: Optional filename for MIME type guessing
        content_type: Optional Content-Type header value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check Content-Type header if provided
    if content_type:
        # Extract base MIME type (ignore parameters like charset)
        base_type = content_type.split(';')[0].strip().lower()
        if base_type not in ALLOWED_MIME_TYPES:
            return False, f"Invalid MIME type: {base_type}"
    
    # Check filename extension if provided
    if filename:
        guessed_type, _ = mimetypes.guess_type(filename)
        if guessed_type and guessed_type.lower() not in ALLOWED_MIME_TYPES:
            return False, f"Invalid file type based on extension: {guessed_type}"
    
    # Validate magic numbers (file signature)
    if not validate_magic_number(file_bytes):
        return False, "File format validation failed: not a valid audio file"
    
    return True, None


def validate_magic_number(file_bytes: bytes) -> bool:
    """Check if file has valid audio file magic number
    
    Args:
        file_bytes: The file content as bytes
        
    Returns:
        True if file appears to be a valid audio file
    """
    if len(file_bytes) < 4:
        return False
    
    # Check for known audio file signatures
    for magic, format_name in AUDIO_MAGIC_NUMBERS.items():
        if file_bytes.startswith(magic):
            logger.debug(f"Detected {format_name} format from magic number")
            return True
    
    # Check for more complex signatures
    # MP4/M4A files have 'ftyp' at offset 4
    if len(file_bytes) > 8 and file_bytes[4:8] == b'ftyp':
        logger.debug("Detected MP4/M4A format from ftyp box")
        return True
    
    # Don't allow unknown formats for better security
    logger.warning("Unknown file format detected")
    return False


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import os.path
    import re
    
    # Remove any path components (gets just the filename)
    # This handles ../../../etc/passwd -> passwd
    filename = os.path.basename(filename)
    
    # Remove dangerous characters but keep alphanumeric, spaces, hyphens, underscores, and dots
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    
    # Remove leading dots to prevent hidden files
    filename = filename.lstrip('.')
    
    # Limit length
    max_length = 255
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext
    
    return filename or 'unnamed_file'


def calculate_file_hash(file_bytes: bytes) -> str:
    """Calculate SHA256 hash of file for integrity checking
    
    Args:
        file_bytes: The file content as bytes
        
    Returns:
        Hex string of SHA256 hash
    """
    return hashlib.sha256(file_bytes).hexdigest()


def validate_audio_file(file_bytes: bytes, filename: Optional[str] = None,
                       content_type: Optional[str] = None, 
                       client_id: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """Comprehensive audio file validation
    
    Args:
        file_bytes: The file content as bytes
        filename: Optional filename
        content_type: Optional Content-Type header
        client_id: Optional client identifier for rate limiting
        
    Returns:
        Tuple of (is_valid, error_message, metadata)
    """
    metadata = {}
    
    # Rate limiting check
    if client_id:
        allowed, error = rate_limiter.is_allowed(client_id)
        if not allowed:
            return False, error, None
    
    # File size validation
    valid, error = validate_file_size(file_bytes)
    if not valid:
        return False, error, None
    
    metadata['size'] = len(file_bytes)
    
    # MIME type and format validation
    valid, error = validate_mime_type(file_bytes, filename, content_type)
    if not valid:
        return False, error, None
    
    # Calculate file hash for logging/tracking
    metadata['hash'] = calculate_file_hash(file_bytes)
    
    # Sanitize filename if provided
    if filename:
        metadata['original_filename'] = filename
        metadata['sanitized_filename'] = sanitize_filename(filename)
    
    logger.info(f"File validation successful: {metadata}")
    return True, None, metadata


def sanitize_ffmpeg_input(audio_bytes: bytes) -> bytes:
    """Sanitize audio input before passing to FFmpeg
    
    Args:
        audio_bytes: Raw audio bytes
        
    Returns:
        Sanitized audio bytes
    """
    # Ensure we have bytes
    if not isinstance(audio_bytes, bytes):
        raise ValueError("Audio input must be bytes")
    
    # Check for reasonable size limits
    if len(audio_bytes) > MAX_FILE_SIZE:
        raise ValueError(f"Audio file exceeds maximum size of {MAX_FILE_SIZE} bytes")
    
    # Return the bytes as-is (FFmpeg will handle from pipe)
    # The actual sanitization happens through proper FFmpeg invocation
    return audio_bytes


class SecurityMiddleware:
    """Sanic middleware for security checks"""
    
    @staticmethod
    def get_client_id(request) -> str:
        """Extract client identifier from request"""
        # Try to get real IP from proxy headers
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        # Fall back to direct connection IP
        return request.ip
    
    @staticmethod
    async def validate_request(request):
        """Validate incoming request
        
        Args:
            request: Sanic request object
            
        Returns:
            Tuple of (is_valid, error_message, metadata)
        """
        # Skip validation for non-upload endpoints
        if request.method != 'POST' or not request.path.startswith('/api/'):
            return True, None, {}
        
        # Check for file upload
        if not request.files:
            return True, None, {}  # No file to validate
        
        # Get the audio file
        speech_file = request.files.get('speech')
        if not speech_file:
            return True, None, {}  # No speech file
        
        # Get client ID for rate limiting
        client_id = SecurityMiddleware.get_client_id(request)
        
        # Get content type
        content_type = request.headers.get('Content-Type')
        
        # Validate the audio file
        return validate_audio_file(
            speech_file.body,
            filename=speech_file.name,
            content_type=content_type,
            client_id=client_id
        )