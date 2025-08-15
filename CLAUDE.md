# Unified STT Service - Project Context

## Project Overview

This is a Unified Speech-to-Text (STT) REST API service that provides a single, consistent interface for multiple offline STT engines. The service is designed to be flexible, scalable, and backward-compatible with legacy DeepSpeech APIs while supporting modern STT engines like Whisper, Vosk, and others.

## Key Features

- **Multi-Engine Support**: Supports 9 different STT engines with automatic fallback
- **Unified API**: Single API interface regardless of underlying engine
- **Backward Compatible**: Maintains compatibility with legacy DeepSpeech API
- **Engine Selection**: Per-request engine selection via query params or headers
- **Automatic Fallback**: Falls back to other engines if primary fails
- **Concurrent Processing**: Thread pool executor for parallel processing
- **Docker Ready**: Full Docker and docker-compose configurations
- **Extensible**: Easy to add new STT engines via base class

## Architecture

### Directory Structure

```
/root/repo/
├── stts/                      # Main application package
│   ├── __init__.py
│   ├── app.py                 # Sanic web application & API endpoints
│   ├── base_engine.py         # Abstract base class for STT engines
│   ├── engine.py              # Main STT engine orchestrator
│   ├── engine_manager.py      # Engine lifecycle & selection manager
│   └── engines/               # Individual STT engine implementations
│       ├── __init__.py
│       ├── coqui.py          # Coqui STT (DeepSpeech successor)
│       ├── deepspeech.py     # Mozilla DeepSpeech (legacy)
│       ├── nemo.py           # NVIDIA NeMo
│       ├── pocketsphinx.py   # CMU PocketSphinx
│       ├── silero.py         # Silero (PyTorch-based)
│       ├── speechbrain.py    # SpeechBrain toolkit
│       ├── vosk.py           # Vosk (Kaldi-based)
│       ├── wav2vec2.py       # Facebook Wav2Vec2
│       └── whisper.py        # OpenAI Whisper
├── Dockerfile                 # Multi-stage Docker build
├── docker-compose.yml         # Multiple service configurations
├── requirements.pip           # Core dependencies
├── requirements-full.pip      # All optional dependencies
├── test_unified_stt.py       # Test script
├── test.mp3                  # Test audio file
├── config.example.json       # Configuration example
├── captain-definition        # CapRover deployment config
├── LICENSE
└── README.md                 # User documentation
```

### Core Components

1. **`stts/app.py`**: Sanic web application with REST API endpoints
   - `/api/v1/stt` - Legacy compatible endpoint
   - `/api/v2/stt/{engine}` - Engine-specific endpoint
   - `/api/v1/engines` - List available engines
   - `/api/v1/engines/{engine}` - Get engine info
   - `/health` - Health check endpoint

2. **`stts/engine.py`**: Main orchestrator that handles:
   - Audio transcription requests
   - Engine selection logic
   - Fallback mechanism
   - Result formatting

3. **`stts/engine_manager.py`**: Manages:
   - Engine registration and initialization
   - Engine lifecycle (lazy loading)
   - Engine availability checking
   - Default engine configuration

4. **`stts/base_engine.py`**: Abstract base class providing:
   - Common interface for all engines
   - Audio normalization (ffmpeg-based)
   - Standard transcription methods

## Supported STT Engines

| Engine | Package | Status | Use Case |
|--------|---------|---------|----------|
| DeepSpeech | `stt` | Legacy | Fast, English-only, discontinued |
| Whisper | `openai-whisper` | Recommended | Best accuracy, 100+ languages |
| Coqui | `STT` | Active | DeepSpeech successor, active development |
| Vosk | `vosk` | Stable | Lightweight, 20+ languages, embedded |
| Silero | `torch, torchaudio` | Stable | PyTorch-based, fast inference |
| Wav2Vec2 | `transformers` | Stable | Transformer-based, high accuracy |
| SpeechBrain | `speechbrain` | Experimental | Research toolkit |
| NeMo | `nemo_toolkit` | Enterprise | NVIDIA enterprise solution |
| PocketSphinx | `pocketsphinx` | Legacy | Ultra-lightweight, limited accuracy |

## API Design

### Request Flow
1. Client sends audio file to API endpoint
2. Engine selection: explicit > default > fallback
3. Audio normalization to 16kHz mono WAV
4. Transcription via selected engine
5. Automatic fallback on failure
6. JSON response with transcript and metadata

### Response Format
```json
{
  "text": "transcribed text here",
  "time": 0.9638,
  "engine": "whisper",
  "fallback": false
}
```

## Configuration

### Environment Variables
- `STT_ENGINE`: Default engine (deepspeech, whisper, vosk, etc.)
- `MAX_ENGINE_WORKERS`: Thread pool size (default: 2)
- `LOG_LEVEL`: Logging level (INFO, DEBUG, ERROR)
- Engine-specific:
  - `WHISPER_MODEL_SIZE`: tiny, base, small, medium, large
  - `WHISPER_DEVICE`: cpu or cuda
  - `VOSK_MODEL_PATH`: Path to Vosk model
  - `SILERO_LANGUAGE`: Language code
  - `WAV2VEC2_MODEL`: HuggingFace model name

## Deployment

### Docker Build Args
- `INSTALL_WHISPER`: Install Whisper engine
- `INSTALL_VOSK`: Install Vosk engine
- `INSTALL_COQUI`: Install Coqui STT
- `INSTALL_SILERO`: Install Silero
- `INSTALL_WAV2VEC2`: Install Wav2Vec2
- `DOWNLOAD_DEEPSPEECH_MODEL`: Download DeepSpeech model
- `DOWNLOAD_VOSK_MODEL`: Download Vosk model
- `DOWNLOAD_COQUI_MODEL`: Download Coqui model

### Docker Compose Profiles
- Default: DeepSpeech-based service
- `whisper`: Whisper-only service
- `vosk`: Vosk-only service
- `multi`: Multi-engine with fallback

## Dependencies

### Core (Required)
- `ffmpeg-python==0.2.0`: Audio processing
- `sanic==20.9.1`: Async web framework
- `numpy`: Numerical operations
- `scipy`: Scientific computing

### STT Engines (Optional)
- DeepSpeech: `stt`
- Whisper: `openai-whisper`
- Vosk: `vosk`
- Coqui: `STT`
- Silero: `torch`, `torchaudio`, `omegaconf`
- Wav2Vec2: `transformers`, `torch`, `librosa`
- SpeechBrain: `speechbrain`
- NeMo: `nemo_toolkit[asr]`
- PocketSphinx: `pocketsphinx`

### System Requirements
- Python 3.9+
- FFmpeg (for audio processing)
- 2-8 GB RAM (engine dependent)
- Optional: CUDA for GPU acceleration

## Development Guidelines

### Adding New STT Engines
1. Create new file in `stts/engines/`
2. Inherit from `BaseSTTEngine`
3. Implement `initialize()` and `transcribe_raw()`
4. Register in `engine_manager.py`
5. Add dependencies to requirements files
6. Update Docker build args if needed

### Testing
- Run unit tests: `python test_unified_stt.py`
- Test specific engine: Set `STT_ENGINE` environment variable
- API testing: Use curl or Postman with test.mp3

### Code Style
- Follow PEP 8 guidelines
- Use type hints where applicable
- Document all public methods
- Add logging for debugging
- Handle exceptions gracefully

## Performance Tuning

### Optimization Tips
1. **Model Selection**: Choose appropriate model size for accuracy/speed tradeoff
2. **Concurrency**: Adjust `MAX_ENGINE_WORKERS` based on CPU cores
3. **GPU Acceleration**: Use CUDA for Whisper/Wav2Vec2 when available
4. **Caching**: Models are cached after first initialization
5. **Audio Format**: Pre-convert to 16kHz WAV to skip normalization

### Benchmarks (Approximate)
| Engine | Speed | Memory | Accuracy |
|--------|-------|---------|----------|
| Vosk | Fast | Low | Good |
| DeepSpeech | Fast | Low | Good |
| Coqui | Fast | Low | Good |
| Silero | Fast | Medium | Good |
| Whisper (tiny) | Medium | Low | Very Good |
| Whisper (base) | Slow | Medium | Excellent |
| Wav2Vec2 | Medium | High | Very Good |

## Security Considerations

- Audio files are processed in memory (not saved to disk)
- Input validation on file uploads
- Configurable request size limits
- No external API calls (all offline processing)
- Docker container runs as non-root user
- Model files can be mounted read-only

## Known Issues & Limitations

1. **Memory Usage**: Large models (Whisper large, Wav2Vec2) require significant RAM
2. **First Request Latency**: Initial model loading can be slow
3. **Audio Format Support**: Some engines have limited format support
4. **Language Support**: Varies by engine and model
5. **Concurrent Requests**: Limited by `MAX_ENGINE_WORKERS`

## Monitoring & Debugging

### Health Checks
- `/health` endpoint for container health
- Engine availability via `/api/v1/engines`
- Detailed logging via `LOG_LEVEL` environment variable

### Common Issues
1. **Engine Not Available**: Check dependencies installed
2. **Model Not Found**: Verify model path/download
3. **OOM Errors**: Reduce model size or workers
4. **Slow Performance**: Enable GPU or use smaller models

## Future Enhancements

- [ ] Streaming transcription support
- [ ] Batch processing endpoint
- [ ] WebSocket support for real-time transcription
- [ ] Model warm-up on startup
- [ ] Prometheus metrics export
- [ ] Redis-based result caching
- [ ] Kubernetes Helm chart
- [ ] gRPC API support
- [ ] Language detection
- [ ] Speaker diarization

## License

See LICENSE file for details. The service itself is open source, but individual STT engines may have their own licenses.