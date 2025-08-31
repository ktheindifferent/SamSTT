"""
Advanced audio validation module for security hardening

This module provides comprehensive validation of audio files to prevent
various attack vectors including zip bombs, infinite loops, and resource exhaustion.
"""

import os
import struct
import logging
from typing import Tuple, Optional, Dict, Any
from io import BytesIO
import hashlib
import wave

logger = logging.getLogger(__name__)


class AudioSecurityValidator:
    """Advanced audio file security validator"""
    
    # Maximum compression ratio to detect potential zip bombs
    MAX_COMPRESSION_RATIO = 100  # Typical audio compression ratios are < 20
    
    # Maximum number of metadata tags (to prevent metadata bombs)
    MAX_METADATA_TAGS = 1000
    
    # Maximum metadata size
    MAX_METADATA_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Suspicious patterns in file headers
    SUSPICIOUS_PATTERNS = [
        b'%PDF',  # PDF file
        b'PK\x03\x04',  # ZIP file
        b'\x1f\x8b\x08',  # GZIP file
        b'Rar!',  # RAR file
        b'\x42\x5a\x68',  # BZIP2 file
        b'\x37\x7a\xbc\xaf\x27\x1c',  # 7-Zip file
        b'MZ',  # PE executable
        b'\x7fELF',  # ELF executable
        b'#!/',  # Shell script
        b'<?php',  # PHP script
        b'<script',  # JavaScript
    ]
    
    @staticmethod
    def check_compression_ratio(file_bytes: bytes) -> Tuple[bool, Optional[str]]:
        """Check for potential compression bombs
        
        Args:
            file_bytes: Raw file bytes
            
        Returns:
            Tuple of (is_safe, error_message)
        """
        # For audio files, check if the file claims to contain
        # significantly more data than its actual size
        
        file_size = len(file_bytes)
        
        # Try to parse as WAV to check claimed vs actual size
        if file_bytes.startswith(b'RIFF'):
            try:
                # Parse RIFF header
                if len(file_bytes) < 44:
                    return False, "Invalid WAV header - too small"
                
                # Get claimed file size from RIFF header
                claimed_size = struct.unpack('<I', file_bytes[4:8])[0] + 8
                
                # Check if claimed size is reasonable
                if claimed_size > file_size * AudioSecurityValidator.MAX_COMPRESSION_RATIO:
                    ratio = claimed_size / file_size
                    return False, f"Potential compression bomb detected - claimed size {ratio:.1f}x actual"
                
                # Check data chunk size
                pos = 12
                while pos < min(len(file_bytes) - 8, 1024 * 1024):  # Search first 1MB
                    chunk_id = file_bytes[pos:pos+4]
                    if chunk_id == b'data':
                        data_size = struct.unpack('<I', file_bytes[pos+4:pos+8])[0]
                        if data_size > file_size * AudioSecurityValidator.MAX_COMPRESSION_RATIO:
                            ratio = data_size / file_size
                            return False, f"Potential audio bomb - data chunk claims {ratio:.1f}x file size"
                        break
                    # Skip to next chunk
                    chunk_size = struct.unpack('<I', file_bytes[pos+4:pos+8])[0]
                    pos += 8 + chunk_size
                    if chunk_size > file_size:
                        return False, "Invalid chunk size in WAV file"
                        
            except Exception as e:
                logger.warning(f"Error checking WAV compression ratio: {e}")
        
        # Check for nested containers (e.g., WAV containing ZIP)
        for pattern in AudioSecurityValidator.SUSPICIOUS_PATTERNS:
            if pattern in file_bytes[44:]:  # Skip WAV header
                return False, f"Suspicious embedded content detected"
        
        return True, None
    
    @staticmethod
    def check_metadata_bombs(file_bytes: bytes) -> Tuple[bool, Optional[str]]:
        """Check for metadata-based attacks
        
        Args:
            file_bytes: Raw file bytes
            
        Returns:
            Tuple of (is_safe, error_message)
        """
        metadata_size = 0
        metadata_count = 0
        
        # Check ID3 tags (MP3)
        if file_bytes.startswith(b'ID3'):
            try:
                # ID3v2 header is 10 bytes
                if len(file_bytes) >= 10:
                    # Get tag size (synchsafe integer)
                    size_bytes = file_bytes[6:10]
                    tag_size = ((size_bytes[0] & 0x7f) << 21 |
                               (size_bytes[1] & 0x7f) << 14 |
                               (size_bytes[2] & 0x7f) << 7 |
                               (size_bytes[3] & 0x7f))
                    
                    if tag_size > AudioSecurityValidator.MAX_METADATA_SIZE:
                        return False, f"ID3 metadata too large: {tag_size} bytes"
                    
                    metadata_size += tag_size
            except Exception as e:
                logger.warning(f"Error parsing ID3 tags: {e}")
        
        # Check Vorbis comments (OGG, FLAC)
        if b'vorbis' in file_bytes[:1024] or file_bytes.startswith(b'fLaC'):
            # Simple heuristic - count comment-like patterns
            comment_markers = [b'TITLE=', b'ARTIST=', b'ALBUM=', b'COMMENT=']
            for marker in comment_markers:
                metadata_count += file_bytes.count(marker)
            
            if metadata_count > AudioSecurityValidator.MAX_METADATA_TAGS:
                return False, f"Too many metadata tags: {metadata_count}"
        
        # Check RIFF INFO chunks (WAV)
        if file_bytes.startswith(b'RIFF'):
            info_chunks = file_bytes.count(b'INFO')
            list_chunks = file_bytes.count(b'LIST')
            
            if info_chunks + list_chunks > 100:  # Reasonable limit
                return False, f"Too many RIFF metadata chunks"
        
        return True, None
    
    @staticmethod
    def check_recursive_structures(file_bytes: bytes) -> Tuple[bool, Optional[str]]:
        """Check for recursive or cyclic structures that could cause infinite loops
        
        Args:
            file_bytes: Raw file bytes
            
        Returns:
            Tuple of (is_safe, error_message)
        """
        # Check for self-referential chunks
        if file_bytes.startswith(b'RIFF'):
            try:
                # Track chunk offsets to detect cycles
                seen_offsets = set()
                pos = 12  # Start after RIFF header
                max_chunks = 10000  # Reasonable limit
                chunk_count = 0
                
                while pos < len(file_bytes) - 8 and chunk_count < max_chunks:
                    if pos in seen_offsets:
                        return False, "Cyclic chunk structure detected"
                    
                    seen_offsets.add(pos)
                    
                    # Read chunk header
                    chunk_id = file_bytes[pos:pos+4]
                    if not chunk_id or len(chunk_id) < 4:
                        break
                    
                    try:
                        chunk_size = struct.unpack('<I', file_bytes[pos+4:pos+8])[0]
                    except:
                        break
                    
                    # Validate chunk size
                    if chunk_size > len(file_bytes) - pos - 8:
                        return False, "Invalid chunk size - potential infinite loop"
                    
                    # Check for unreasonable number of small chunks (chunk bomb)
                    if chunk_size < 8:
                        chunk_count += 1
                        if chunk_count > 1000:
                            return False, "Too many small chunks - potential chunk bomb"
                    
                    # Move to next chunk
                    pos += 8 + chunk_size
                    if chunk_size % 2 == 1:  # RIFF chunks are word-aligned
                        pos += 1
                    
                    chunk_count += 1
                
                if chunk_count >= max_chunks:
                    return False, f"Too many chunks: {chunk_count}"
                    
            except Exception as e:
                logger.warning(f"Error checking RIFF structure: {e}")
        
        return True, None
    
    @staticmethod
    def check_polyglot_files(file_bytes: bytes) -> Tuple[bool, Optional[str]]:
        """Check for polyglot files that could be interpreted differently by different parsers
        
        Args:
            file_bytes: Raw file bytes
            
        Returns:
            Tuple of (is_safe, error_message)
        """
        # Check for multiple valid file signatures
        signatures_found = []
        
        # Audio signatures
        if file_bytes.startswith(b'RIFF'):
            signatures_found.append('WAV')
        if file_bytes.startswith(b'ID3') or file_bytes.startswith(b'\xff\xfb'):
            signatures_found.append('MP3')
        if file_bytes.startswith(b'fLaC'):
            signatures_found.append('FLAC')
        if file_bytes.startswith(b'OggS'):
            signatures_found.append('OGG')
        
        # Non-audio signatures that shouldn't be present
        if b'%PDF' in file_bytes[:1024]:
            return False, "PDF signature found in audio file"
        if b'<html' in file_bytes[:1024] or b'<HTML' in file_bytes[:1024]:
            return False, "HTML content found in audio file"
        if b'<?xml' in file_bytes[:1024]:
            return False, "XML content found in audio file"
        
        # Check for multiple audio signatures (suspicious)
        if len(signatures_found) > 1:
            return False, f"Multiple file signatures found: {signatures_found}"
        
        return True, None
    
    @staticmethod
    def validate_wav_structure(file_bytes: bytes) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Detailed WAV file structure validation
        
        Args:
            file_bytes: Raw file bytes
            
        Returns:
            Tuple of (is_valid, error_message, metadata)
        """
        metadata = {}
        
        if not file_bytes.startswith(b'RIFF'):
            return False, "Not a valid RIFF file", metadata
        
        if len(file_bytes) < 44:
            return False, "File too small to be valid WAV", metadata
        
        # Check WAVE signature
        if file_bytes[8:12] != b'WAVE':
            return False, "Not a valid WAVE file", metadata
        
        try:
            # Parse fmt chunk
            fmt_pos = file_bytes.find(b'fmt ')
            if fmt_pos == -1 or fmt_pos > 1000:  # Should be near the beginning
                return False, "fmt chunk not found or too far from start", metadata
            
            fmt_size = struct.unpack('<I', file_bytes[fmt_pos+4:fmt_pos+8])[0]
            if fmt_size < 16 or fmt_size > 1000:  # Reasonable limits
                return False, f"Invalid fmt chunk size: {fmt_size}", metadata
            
            # Parse format data
            fmt_data = file_bytes[fmt_pos+8:fmt_pos+8+fmt_size]
            if len(fmt_data) < 16:
                return False, "Incomplete fmt chunk", metadata
            
            audio_format = struct.unpack('<H', fmt_data[0:2])[0]
            num_channels = struct.unpack('<H', fmt_data[2:4])[0]
            sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
            byte_rate = struct.unpack('<I', fmt_data[8:12])[0]
            block_align = struct.unpack('<H', fmt_data[12:14])[0]
            bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]
            
            metadata['audio_format'] = audio_format
            metadata['channels'] = num_channels
            metadata['sample_rate'] = sample_rate
            metadata['bits_per_sample'] = bits_per_sample
            
            # Validate format parameters
            if audio_format not in [1, 3]:  # PCM or IEEE float
                return False, f"Unsupported audio format: {audio_format}", metadata
            
            if num_channels == 0 or num_channels > 32:
                return False, f"Invalid channel count: {num_channels}", metadata
            
            if sample_rate < 1000 or sample_rate > 384000:
                return False, f"Invalid sample rate: {sample_rate}", metadata
            
            if bits_per_sample not in [8, 16, 24, 32]:
                return False, f"Invalid bits per sample: {bits_per_sample}", metadata
            
            # Validate byte rate
            expected_byte_rate = sample_rate * num_channels * bits_per_sample // 8
            if abs(byte_rate - expected_byte_rate) > 100:  # Allow small variance
                return False, f"Inconsistent byte rate: {byte_rate} vs expected {expected_byte_rate}", metadata
            
            # Find and validate data chunk
            data_pos = file_bytes.find(b'data')
            if data_pos == -1:
                return False, "data chunk not found", metadata
            
            data_size = struct.unpack('<I', file_bytes[data_pos+4:data_pos+8])[0]
            metadata['data_size'] = data_size
            
            # Calculate duration
            if byte_rate > 0:
                duration = data_size / byte_rate
                metadata['duration_seconds'] = duration
                
                # Check for unreasonable duration
                if duration > 3600:  # 1 hour
                    return False, f"Unreasonably long duration: {duration:.1f} seconds", metadata
            
            # Validate actual file size
            expected_size = data_pos + 8 + data_size
            if abs(len(file_bytes) - expected_size) > 1000:  # Allow some padding
                return False, f"File size mismatch: {len(file_bytes)} vs expected {expected_size}", metadata
            
        except Exception as e:
            return False, f"Error parsing WAV structure: {e}", metadata
        
        return True, None, metadata
    
    @staticmethod
    def comprehensive_validate(file_bytes: bytes) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Perform comprehensive security validation on audio file
        
        Args:
            file_bytes: Raw file bytes
            
        Returns:
            Tuple of (is_valid, error_message, metadata)
        """
        metadata = {
            'file_size': len(file_bytes),
            'file_hash': hashlib.sha256(file_bytes).hexdigest()
        }
        
        # Check for compression bombs
        is_safe, error = AudioSecurityValidator.check_compression_ratio(file_bytes)
        if not is_safe:
            return False, error, metadata
        
        # Check for metadata bombs
        is_safe, error = AudioSecurityValidator.check_metadata_bombs(file_bytes)
        if not is_safe:
            return False, error, metadata
        
        # Check for recursive structures
        is_safe, error = AudioSecurityValidator.check_recursive_structures(file_bytes)
        if not is_safe:
            return False, error, metadata
        
        # Check for polyglot files
        is_safe, error = AudioSecurityValidator.check_polyglot_files(file_bytes)
        if not is_safe:
            return False, error, metadata
        
        # Detailed format validation for WAV files
        if file_bytes.startswith(b'RIFF'):
            is_valid, error, wav_metadata = AudioSecurityValidator.validate_wav_structure(file_bytes)
            if not is_valid:
                return False, error, metadata
            metadata.update(wav_metadata)
        
        return True, None, metadata


def validate_audio_security(audio_bytes: bytes) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Convenience function for audio security validation
    
    Args:
        audio_bytes: Raw audio file bytes
        
    Returns:
        Tuple of (is_valid, error_message, metadata)
    """
    validator = AudioSecurityValidator()
    return validator.comprehensive_validate(audio_bytes)