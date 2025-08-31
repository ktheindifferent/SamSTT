# Security Hardening Documentation

## Overview

The Unified STT Service implements comprehensive security hardening for FFmpeg operations to prevent DoS attacks through crafted audio files. This document describes the security measures, configuration options, and best practices for secure deployment.

## Security Features

### 1. FFmpeg Sandboxing

The service uses a sandboxed FFmpeg execution environment with:

- **Resource Limits**: CPU, memory, and time limits for FFmpeg processes
- **Process Isolation**: FFmpeg runs with reduced privileges when possible
- **Timeout Protection**: Automatic termination of hanging operations
- **Circuit Breaker**: Prevents cascading failures from repeated errors

### 2. Audio File Validation

Multi-layer validation before processing:

- **File Size Limits**: Configurable maximum file size (default 50MB)
- **Format Validation**: Magic number verification for audio formats
- **Structure Validation**: Deep inspection of file structure
- **Metadata Validation**: Protection against metadata bombs
- **Compression Ratio Checks**: Detection of zip bomb attacks
- **Polyglot Detection**: Identifies files with multiple signatures

### 3. Rate Limiting

Per-client and global rate limiting:

- **Request Rate Limiting**: Max requests per minute/hour
- **Concurrent Request Limits**: Per-client concurrent processing
- **Resource-Intensive Operation Throttling**: Special limits for FFmpeg

### 4. Resource Monitoring

Real-time monitoring of FFmpeg processes:

- **Memory Usage Tracking**: Peak memory monitoring
- **CPU Time Tracking**: Total CPU seconds consumed
- **Automatic Termination**: Kill processes exceeding limits
- **Statistics Collection**: Performance metrics for analysis

## Configuration

### Environment Variables

```bash
# Security Limits
MAX_FILE_SIZE=52428800                    # 50MB max file size
MAX_REQUESTS_PER_MINUTE=60                # Global rate limit
MAX_REQUESTS_PER_HOUR=600                 # Global rate limit
REQUEST_TIMEOUT=60                         # Request timeout in seconds

# FFmpeg Security Configuration
FFMPEG_MAX_MEMORY_MB=512                  # Max memory for FFmpeg
FFMPEG_MAX_CPU_SECONDS=30                 # Max CPU time
FFMPEG_MAX_OUTPUT_SIZE_MB=100            # Max output size
FFMPEG_MAX_DURATION_SECONDS=600          # Max audio duration (10 min)
FFMPEG_TIMEOUT_SECONDS=10                # Process timeout
FFMPEG_MAX_ANALYZEDURATION=10000000      # Max probe duration (microseconds)
FFMPEG_MAX_PROBESIZE=10000000           # Max probe size (bytes)
FFMPEG_MAX_THREADS=1                     # FFmpeg thread limit

# Circuit Breaker
FFMPEG_CIRCUIT_BREAKER_THRESHOLD=5       # Failures before opening
FFMPEG_CIRCUIT_BREAKER_RESET_TIME=60    # Reset time in seconds

# Per-Client Limits
FFMPEG_MAX_CONCURRENT_PER_CLIENT=2      # Concurrent FFmpeg per client
FFMPEG_MAX_REQUESTS_PER_MINUTE=10       # FFmpeg requests per minute

# Compatibility Mode
FFMPEG_LEGACY_MODE=false                # Use legacy mode (less secure)
```

## Security Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Request                         │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Rate Limiter                              │
│         (Per-client and global limits)                      │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                Audio Validator                              │
│    (Size, format, structure, metadata checks)              │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              FFmpeg Security Sandbox                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Circuit Breaker                            │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Resource Monitor                             │  │
│  │    (Memory, CPU, timeout tracking)                   │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      Sandboxed FFmpeg Process                        │  │
│  │   (Limited resources, no shell, pipe I/O only)       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Validation Layers

1. **Input Validation**
   - File size check
   - MIME type validation
   - Magic number verification

2. **Structure Validation**
   - WAV/RIFF structure parsing
   - Chunk size verification
   - Metadata size limits

3. **Security Validation**
   - Compression ratio analysis
   - Recursive structure detection
   - Polyglot file detection
   - Embedded content scanning

4. **Runtime Protection**
   - Resource limits enforcement
   - Timeout enforcement
   - Memory monitoring
   - CPU usage tracking

## Attack Mitigation

### Protected Against

1. **Zip Bombs**: Files claiming to be much larger than actual size
2. **Metadata Bombs**: Excessive metadata causing memory exhaustion
3. **Infinite Loops**: Circular references in file structure
4. **Resource Exhaustion**: CPU/memory consumption attacks
5. **Polyglot Files**: Files with multiple valid interpretations
6. **Malformed Files**: Corrupted or crafted malicious structures
7. **Long Duration Files**: Excessive processing time attacks
8. **Concurrent Attack**: Multiple simultaneous resource-heavy requests

### Detection Mechanisms

- **Compression Ratio Check**: Detects files with claimed size >> actual size
- **Metadata Size Limits**: Prevents excessive metadata parsing
- **Chunk Validation**: Verifies RIFF/WAV chunk structures
- **Pattern Matching**: Identifies suspicious embedded content
- **Resource Monitoring**: Real-time tracking of process resources
- **Circuit Breaking**: Stops repeated failures from same source

## Deployment Best Practices

### 1. Container Security

```dockerfile
# Run as non-root user
USER nobody

# Read-only root filesystem
# Mount only necessary volumes

# Resource limits in docker-compose
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 1G
    reservations:
      memory: 512M
```

### 2. Network Security

- Use reverse proxy with rate limiting (nginx, HAProxy)
- Enable TLS/HTTPS for all connections
- Implement IP-based rate limiting at proxy level
- Use Web Application Firewall (WAF) if available

### 3. Monitoring

Essential metrics to monitor:

- FFmpeg process count and duration
- Memory usage patterns
- Circuit breaker state changes
- Rate limit violations
- File validation failures
- Error rates by type

### 4. Logging

Important events to log:

- All validation failures with details
- Resource limit violations
- Circuit breaker state changes
- Unusual file characteristics
- Client rate limit violations

### 5. Configuration Tuning

Start with conservative limits and adjust based on:

- Legitimate use case requirements
- Available system resources
- Observed attack patterns
- Performance requirements

Example production configuration:

```bash
# Conservative production settings
FFMPEG_MAX_MEMORY_MB=256          # Lower memory limit
FFMPEG_MAX_CPU_SECONDS=10         # Shorter CPU limit
FFMPEG_TIMEOUT_SECONDS=5          # Faster timeout
FFMPEG_MAX_DURATION_SECONDS=300   # 5 minute max
FFMPEG_MAX_CONCURRENT_PER_CLIENT=1 # Single concurrent per client
```

## Security Testing

### Test Scenarios

The security test suite (`test_ffmpeg_security.py`) covers:

1. **Compression bomb detection**
2. **Metadata bomb detection**
3. **Recursive structure detection**
4. **Polyglot file detection**
5. **Circuit breaker functionality**
6. **Rate limiting enforcement**
7. **Timeout protection**
8. **Resource limit enforcement**

### Running Security Tests

```bash
python3 test_ffmpeg_security.py
```

### Creating Test Files

Examples of malicious test files:

```python
# Compression bomb
fake_wav = bytearray(b'RIFF')
fake_wav.extend(struct.pack('<I', 1024*1024*1024))  # Claim 1GB
fake_wav.extend(b'WAVEdata')
fake_wav.extend(struct.pack('<I', 1024*1024*1024))  # Claim 1GB data
fake_wav.extend(b'\x00' * 100)  # Only 100 bytes actual

# Metadata bomb
fake_mp3 = bytearray(b'ID3\x04\x00\x00')
fake_mp3.extend(bytes([0x7F, 0x7F, 0x7F, 0x7F]))  # Max size claim
```

## Incident Response

### If Under Attack

1. **Immediate Actions**:
   - Check circuit breaker states
   - Review rate limit violations
   - Monitor resource usage

2. **Mitigation**:
   - Temporarily reduce limits
   - Block attacking IPs
   - Enable FFMPEG_LEGACY_MODE if needed

3. **Recovery**:
   - Reset circuit breakers
   - Clear rate limit counters
   - Review and adjust limits

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Circuit breaker open | Too many failures | Wait for reset or manually reset |
| Legitimate files rejected | Limits too strict | Increase specific limits |
| High memory usage | Large files or many concurrent | Reduce limits or add resources |
| Timeouts on valid files | Timeout too short | Increase FFMPEG_TIMEOUT_SECONDS |

## Security Updates

Stay informed about:

- FFmpeg security advisories
- Python package vulnerabilities
- Container base image updates
- New attack vectors

Regular security maintenance:

1. Update FFmpeg regularly
2. Review and update Python dependencies
3. Audit configuration settings
4. Test with new attack patterns
5. Monitor security mailing lists

## Compliance

The security implementation helps meet:

- **OWASP Top 10** protection (A04:2021 - Insecure Design)
- **CIS Controls** for input validation
- **PCI DSS** requirements for input validation
- **GDPR** data protection through resource limits

## Support

For security issues:

1. Check logs for validation failures
2. Review configuration settings
3. Run security test suite
4. Check system resources
5. Report security vulnerabilities responsibly

## References

- [FFmpeg Security Considerations](https://ffmpeg.org/security.html)
- [OWASP Input Validation](https://owasp.org/www-project-proactive-controls/)
- [CWE-400: Uncontrolled Resource Consumption](https://cwe.mitre.org/data/definitions/400.html)
- [Audio File Attack Vectors](https://www.contextis.com/en/blog/audio-file-processing-denial-of-service)