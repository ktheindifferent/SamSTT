Unified STT Service
===================
A unified Speech-to-Text REST API service supporting multiple offline STT engines.

## Features
- **Multiple STT Engine Support**: DeepSpeech, Whisper, Coqui, Vosk, Silero, Wav2Vec2
- **Unified API**: Single API interface for all engines
- **Automatic Fallback**: Falls back to other engines if primary fails
- **Engine Selection**: Choose engine per request or use default
- **Backward Compatible**: Maintains compatibility with legacy DeepSpeech API

## Supported STT Engines

### 1. Mozilla DeepSpeech (Legacy)
- Fast, lightweight
- Requires pre-trained model files (.pbmm or .tflite)

### 2. OpenAI Whisper
- State-of-the-art accuracy
- Multiple model sizes (tiny, base, small, medium, large)
- Supports 100+ languages
- No separate model download needed

### 3. Coqui STT
- Successor to Mozilla DeepSpeech
- Compatible with DeepSpeech models
- Active development and community

### 4. Vosk
- Lightweight, supports 20+ languages
- Works offline with small models
- Good for embedded systems

### 5. Silero
- PyTorch-based models
- Fast inference
- Multiple language support

### 6. Wav2Vec2
- Facebook/Meta's transformer-based model
- High accuracy
- HuggingFace integration

## API Endpoints

### 1. Transcribe Audio (Legacy Compatible)
```bash
POST /api/v1/stt

# Basic usage - uses default engine
curl -X POST -F "speech=@test.mp3" http://127.0.0.1:8000/api/v1/stt

# Specify engine via query parameter
curl -X POST -F "speech=@test.mp3" http://127.0.0.1:8000/api/v1/stt?engine=whisper

# Specify engine via header
curl -X POST -F "speech=@test.mp3" -H "X-STT-Engine: vosk" http://127.0.0.1:8000/api/v1/stt

Response:
{
  "text": "experience proves this",
  "time": 0.9638,
  "engine": "whisper",
  "fallback": false
}
```

### 2. Transcribe with Explicit Engine (v2 API)
```bash
POST /api/v2/stt/{engine_name}

curl -X POST -F "speech=@test.mp3" http://127.0.0.1:8000/api/v2/stt/whisper
```

### 3. List Available Engines
```bash
GET /api/v1/engines

curl http://127.0.0.1:8000/api/v1/engines

Response:
{
  "available": ["deepspeech", "whisper"],
  "all": ["deepspeech", "whisper", "coqui", "vosk", "silero", "wav2vec2"],
  "default": "whisper",
  "details": {
    "deepspeech": {"available": true, "initialized": true, ...},
    "whisper": {"available": true, "initialized": true, ...}
  }
}
```

### 4. Get Engine Information
```bash
GET /api/v1/engines/{engine_name}

curl http://127.0.0.1:8000/api/v1/engines/whisper
```

### 5. Health Check
```bash
GET /health

curl http://127.0.0.1:8000/health
```

## Setup

### Quick Start with Docker

```bash
# Clone repository
git clone <repository-url>
cd stt-service

# Build and run with docker-compose
docker-compose up
```

### Manual Setup

#### 1. Install Core Dependencies
```bash
pip install -r requirements.pip
```

#### 2. Install STT Engine Dependencies

Choose and install the engines you want to use:

```bash
# For Whisper (recommended for best accuracy)
pip install openai-whisper

# For Vosk (lightweight, many languages)
pip install vosk

# For Coqui STT
pip install STT

# For Silero (requires PyTorch)
pip install torch torchaudio omegaconf

# For Wav2Vec2
pip install transformers torch
```

#### 3. Download Models (if needed)

**For DeepSpeech/Coqui:**
```bash
# DeepSpeech model
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm -O model.pbmm

# Or Coqui model
wget <coqui-model-url> -O coqui_model.tflite
```

**For Vosk:**
```bash
# Download a model for your language
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip
mv vosk-model-en-us-0.22 vosk_model
```

**For Whisper:**
Models are automatically downloaded on first use.

#### 4. Configure Environment

Set environment variables to configure the service:

```bash
# Default STT engine
export STT_ENGINE=whisper  # or deepspeech, coqui, vosk, silero, wav2vec2

# Whisper configuration
export WHISPER_MODEL_SIZE=base  # tiny, base, small, medium, large
export WHISPER_DEVICE=cpu  # or cuda
export WHISPER_LANGUAGE=en  # or auto-detect

# Vosk configuration
export VOSK_MODEL_PATH=/path/to/vosk_model

# Silero configuration
export SILERO_LANGUAGE=en
export SILERO_DEVICE=cpu

# Wav2Vec2 configuration
export WAV2VEC2_MODEL=facebook/wav2vec2-base-960h
export WAV2VEC2_DEVICE=cpu

# Service configuration
export MAX_ENGINE_WORKERS=2
export LOG_LEVEL=INFO
```

#### 5. Run the Service

```bash
python -m stts.app
```

## Docker Deployment

### Using Docker Compose

```yaml
version: '3.8'
services:
  stt:
    build: .
    ports:
      - "8000:8000"
    environment:
      - STT_ENGINE=whisper
      - WHISPER_MODEL_SIZE=base
      - MAX_ENGINE_WORKERS=2
    volumes:
      - ./models:/app/models  # For model persistence
```

### Using Docker Run

```bash
# Build image
docker build -t unified-stt:latest .

# Run with Whisper (no model volume needed)
docker run -p 8000:8000 -e STT_ENGINE=whisper unified-stt:latest

# Run with DeepSpeech (mount model)
docker run -p 8000:8000 -v $(pwd)/model.pbmm:/app/model.pbmm:ro unified-stt:latest
```

## Engine Selection Strategy

The service uses the following strategy for engine selection:

1. **Explicit Selection**: If engine specified in request, use that engine
2. **Default Engine**: Use the configured default engine (via STT_ENGINE env var)
3. **Automatic Fallback**: If primary engine fails, try other available engines
4. **Load Balancing**: Can configure multiple workers for concurrent processing

## Performance Considerations

| Engine | Speed | Accuracy | Memory | Languages | Notes |
|--------|-------|----------|---------|-----------|-------|
| DeepSpeech | Fast | Good | Low | English | Legacy, no longer maintained |
| Whisper | Slow-Medium | Excellent | Medium-High | 100+ | Best accuracy, GPU recommended for large models |
| Coqui | Fast | Good | Low | Multiple | Active development |
| Vosk | Fast | Good | Low | 20+ | Good for embedded/mobile |
| Silero | Fast | Good | Medium | Multiple | Requires PyTorch |
| Wav2Vec2 | Medium | Very Good | High | Multiple | Transformer-based |

## Recommendations

- **For best accuracy**: Use Whisper with 'base' or larger model
- **For speed**: Use Vosk or Coqui with appropriate models
- **For embedded/IoT**: Use Vosk with small models
- **For GPU acceleration**: Use Whisper or Wav2Vec2 with CUDA
- **For production**: Configure multiple engines for fallback

## API Migration Guide

### From Legacy DeepSpeech API

The service is fully backward compatible. Your existing code will continue to work:

```bash
# Old code still works
curl -X POST -F "speech=@test.mp3" http://127.0.0.1:8000/api/v1/stt
```

To use new engines, simply add the engine parameter:

```bash
# Use Whisper instead
curl -X POST -F "speech=@test.mp3" http://127.0.0.1:8000/api/v1/stt?engine=whisper
```

## Troubleshooting

### Engine Not Available

If an engine shows as unavailable:
1. Check that the required package is installed
2. Verify model files exist (for engines that need them)
3. Check logs for initialization errors

### Memory Issues

- Reduce model size (e.g., use whisper 'tiny' instead of 'base')
- Limit concurrent workers via MAX_ENGINE_WORKERS
- Use more lightweight engines (Vosk, Coqui)

### Slow Performance

- Use GPU acceleration where supported (Whisper, Wav2Vec2)
- Choose smaller models
- Increase MAX_ENGINE_WORKERS for parallel processing
- Consider using faster engines (Vosk, Coqui) if accuracy permits

## License

See LICENSE file for details.