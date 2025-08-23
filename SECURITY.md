# Security Features and Best Practices

## Overview

This document describes the security features implemented in the Unified STT Service to protect against various attack vectors and ensure safe operation in production environments.

## Security Features

### 1. Input Validation

#### File Size Limits
- **Default Limit**: 50MB (configurable via `MAX_FILE_SIZE` environment variable)
- **Protection Against**: DoS attacks via large file uploads
- **Implementation**: Checked before processing in `validators.py`

#### MIME Type Validation
- **Allowed Types**: Audio formats only (wav, mp3, mp4, ogg, flac, webm, etc.)
- **Protection Against**: Upload of malicious non-audio files
- **Implementation**: Both Content-Type header and magic number validation

#### Magic Number Validation
- **What**: Validates file format by checking initial bytes (file signature)
- **Protection Against**: File type spoofing, polyglot files
- **Supported Formats**: WAV, MP3, FLAC, OGG, MP4/M4A, WebM, AMR

### 2. Rate Limiting

#### Per-Client Limits
- **Per Minute**: 60 requests (configurable via `MAX_REQUESTS_PER_MINUTE`)
- **Per Hour**: 600 requests (configurable via `MAX_REQUESTS_PER_HOUR`)
- **Protection Against**: DoS attacks, resource exhaustion
- **Implementation**: Sliding window algorithm with thread-safe counters

#### Client Identification
- Supports proxy headers (`X-Forwarded-For`, `X-Real-IP`)
- Falls back to direct connection IP
- Per-IP tracking of request rates

### 3. Request Timeouts

#### Processing Timeout
- **Default**: 60 seconds (configurable via `REQUEST_TIMEOUT`)
- **Protection Against**: Slow loris attacks, hung processes
- **Implementation**: Async timeout on transcription operations

### 4. FFmpeg Sanitization

#### Secure FFmpeg Invocation
- **No Shell Execution**: Direct process invocation without shell
- **Pipe-Only I/O**: No file system access (stdin/stdout only)
- **Resource Limits**: Thread and probe size limitations
- **Format Enforcement**: Explicit codec and format specifications

#### Input Sanitization
- Size validation before processing
- Type checking (must be bytes)
- Output size validation after conversion

### 5. Path Traversal Protection

#### Filename Sanitization
- Removes path components (`../`, `./`, etc.)
- Strips dangerous characters
- Limits filename length
- Handles null byte injection

### 6. Security Headers

#### Response Headers
- `X-Content-Type-Options: nosniff` - Prevents MIME sniffing
- `X-Frame-Options: DENY` - Prevents clickjacking
- `X-XSS-Protection: 1; mode=block` - XSS protection
- `Content-Security-Policy: default-src 'self'` - CSP policy

### 7. Error Handling

#### Safe Error Messages
- No sensitive information in error responses
- Logged errors with full details for debugging
- Generic user-facing error messages

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_FILE_SIZE` | 52428800 (50MB) | Maximum upload file size in bytes |
| `MAX_REQUESTS_PER_MINUTE` | 60 | Rate limit per minute per IP |
| `MAX_REQUESTS_PER_HOUR` | 600 | Rate limit per hour per IP |
| `REQUEST_TIMEOUT` | 60 | Request processing timeout in seconds |
| `LOG_LEVEL` | INFO | Logging verbosity |

### Sanic Configuration

```python
app.config.REQUEST_MAX_SIZE = MAX_FILE_SIZE
app.config.REQUEST_TIMEOUT = REQUEST_TIMEOUT
app.config.RESPONSE_TIMEOUT = REQUEST_TIMEOUT
```

## Testing

### Security Test Suite

Run comprehensive security tests:

```bash
python test_security.py
```

Test categories:
- File size validation
- MIME type validation
- Magic number validation
- Filename sanitization
- Rate limiting
- FFmpeg sanitization
- Malicious input handling
- Performance under load

### Manual Testing

#### Test Oversized File
```bash
# Create a 51MB file (over default limit)
dd if=/dev/zero of=large.mp3 bs=1M count=51
curl -X POST -F "speech=@large.mp3" http://localhost:8000/api/v1/stt
# Expected: Error about file size
```

#### Test Rate Limiting
```bash
# Send rapid requests
for i in {1..100}; do
  curl -X POST -F "speech=@test.mp3" http://localhost:8000/api/v1/stt &
done
# Expected: Some requests blocked with rate limit error
```

#### Test Invalid File Type
```bash
# Try uploading a text file
echo "malicious content" > evil.txt
curl -X POST -F "speech=@evil.txt" http://localhost:8000/api/v1/stt
# Expected: Error about invalid file format
```

#### Test Path Traversal
```bash
curl -X POST -F "speech=@test.mp3;filename=../../../etc/passwd" \
  http://localhost:8000/api/v1/stt
# Expected: Filename sanitized, no path traversal
```

## Security Best Practices

### Deployment Recommendations

1. **Use HTTPS**: Always deploy behind HTTPS in production
2. **Reverse Proxy**: Use nginx/Apache as reverse proxy with additional security features
3. **Container Security**: Run containers as non-root user
4. **Network Isolation**: Use private networks for internal services
5. **Monitoring**: Implement logging and alerting for security events

### Resource Limits

1. **Container Limits**: Set memory and CPU limits
2. **Disk Quotas**: Implement disk space quotas
3. **Connection Limits**: Configure max connections in reverse proxy
4. **Process Limits**: Use systemd or container limits for max processes

### Monitoring

1. **Log Analysis**: Monitor for suspicious patterns
2. **Rate Limit Hits**: Track IPs hitting rate limits
3. **Error Rates**: Monitor for unusual error patterns
4. **Resource Usage**: Track CPU/memory/disk usage

## Threat Model

### Considered Threats

1. **Denial of Service (DoS)**
   - Large file uploads
   - Rapid request flooding
   - Slow loris attacks
   - Resource exhaustion

2. **Remote Code Execution (RCE)**
   - Command injection via filenames
   - FFmpeg vulnerabilities
   - Path traversal attacks

3. **Information Disclosure**
   - Error message leakage
   - Path disclosure
   - System information exposure

4. **Data Integrity**
   - File type spoofing
   - Polyglot files
   - Malformed audio files

### Mitigation Summary

| Threat | Mitigation |
|--------|------------|
| Large file DoS | File size limits |
| Request flooding | Rate limiting |
| Slow requests | Request timeouts |
| Command injection | Input sanitization, no shell execution |
| Path traversal | Filename sanitization |
| File type attacks | Magic number validation |
| Information leakage | Generic error messages |

## Incident Response

### Security Event Handling

1. **Detection**: Monitor logs for security events
2. **Analysis**: Investigate suspicious patterns
3. **Response**: Block malicious IPs, adjust limits
4. **Recovery**: Reset rate limits, clear caches
5. **Post-Mortem**: Document and improve defenses

### Contact

For security issues, please contact the security team or file a private security advisory.

## Compliance

This service implements security controls suitable for:
- OWASP Top 10 protection
- CIS Docker Benchmark compliance
- GDPR data protection (no PII storage)
- SOC 2 Type II requirements

## Future Enhancements

- [ ] Web Application Firewall (WAF) integration
- [ ] Machine learning-based anomaly detection
- [ ] Distributed rate limiting (Redis-based)
- [ ] Certificate pinning for HTTPS
- [ ] Content Security Policy (CSP) enhancements
- [ ] Security audit logging
- [ ] Vulnerability scanning integration
- [ ] Automated security testing in CI/CD