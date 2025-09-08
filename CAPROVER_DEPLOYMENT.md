# CapRover Deployment Guide

## Quick Start

1. **Create a new app in CapRover**
   - Go to your CapRover dashboard
   - Click "Apps" → "One-Click Apps/Deploy"
   - Create a new app with your desired name (e.g., `stt-service`)

2. **Deploy from GitHub**
   - In the app settings, go to "Deployment"
   - Select "Deploy from GitHub"
   - Enter your repository URL
   - Set the branch (usually `master` or `main`)

3. **Configure Docker Build Arguments**
   - In "App Configs" → "Build Arguments", add:
   ```
   CACHEBUST=7
   DOWNLOAD_COQUI_MODEL=true
   ```
   - To install ALL engines for benchmarking:
   ```
   INSTALL_ALL=true
   DOWNLOAD_COQUI_MODEL=true
   DOWNLOAD_VOSK_MODEL=true
   ```
   - Or install specific engines:
   ```
   INSTALL_WHISPER=true
   INSTALL_VOSK=true
   DOWNLOAD_VOSK_MODEL=true
   ```

4. **Set Environment Variables**
   - In "App Configs" → "Environmental Variables", add:
   ```
   STT_ENGINE=coqui
   MAX_ENGINE_WORKERS=2
   LOG_LEVEL=INFO
   RUN_BENCHMARK_ON_STARTUP=true
   ```

5. **Deploy**
   - Click "Deploy Now"
   - Wait for the build to complete

## Benchmark Feature

The service can automatically benchmark all available engines on startup to determine which is fastest on your hardware. This helps optimize performance by identifying the best engine for your specific deployment.

**Enable Benchmarking:**
- Set `RUN_BENCHMARK_ON_STARTUP=true` in environment variables
- Results appear in `/api/v1/engines` response
- Run manual benchmarks with `POST /api/v1/benchmark`
- View results with `GET /api/v1/benchmark`

**Benchmark Results Include:**
- Initialization time per engine
- Average transcription time (3 runs)
- Fastest engine identification
- Success/failure status

## Recommended Configurations

### Basic Setup (Coqui STT)
**Build Args:**
```
CACHEBUST=7
DOWNLOAD_COQUI_MODEL=true
```
**Environment:**
```
STT_ENGINE=coqui
MAX_ENGINE_WORKERS=2
```

### High Accuracy Setup (Whisper)
**Build Args:**
```
INSTALL_WHISPER=true
```
**Environment:**
```
STT_ENGINE=whisper
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
MAX_ENGINE_WORKERS=2
```

### Lightweight Setup (Vosk)
**Build Args:**
```
INSTALL_VOSK=true
DOWNLOAD_VOSK_MODEL=true
```
**Environment:**
```
STT_ENGINE=vosk
VOSK_MODEL_PATH=/app/vosk_model
MAX_ENGINE_WORKERS=2
```

### Multi-Engine Setup (with fallback)
**Build Args:**
```
INSTALL_WHISPER=true
INSTALL_VOSK=true
INSTALL_COQUI=true
DOWNLOAD_COQUI_MODEL=true
DOWNLOAD_VOSK_MODEL=true
```
**Environment:**
```
STT_ENGINE=whisper
WHISPER_MODEL_SIZE=tiny
MAX_ENGINE_WORKERS=4
```

## Advanced Configuration

### Persistent Volume for Models
To avoid downloading models on every deployment:

1. Create a persistent directory in CapRover
2. Add volume mapping in "App Configs" → "Persistent Directories":
   ```
   /app/models:/var/lib/docker/volumes/captain--stt-models/_data
   ```
3. Upload your models to this directory via SSH/SFTP

### Resource Limits
In "App Configs" → "Resource Limits":
- Memory: 2GB minimum (4GB+ for Whisper/Wav2Vec2)
- CPU: 1 core minimum (2+ cores recommended)

### HTTP Settings
In "HTTP Settings":
- Enable HTTPS if needed
- Set custom domain
- Configure timeout to 60s for large audio files

### Monitoring
- Health check endpoint: `/health`
- List engines: `/api/v1/engines`
- Engine info: `/api/v1/engines/{engine}`

## Troubleshooting

### App Won't Start
- Check logs in CapRover dashboard
- Verify model download succeeded
- Ensure sufficient memory allocated

### Slow Performance
- Increase `MAX_ENGINE_WORKERS`
- Use smaller models (whisper tiny vs base)
- Add more CPU/memory resources

### Model Not Found
- Check build logs for download errors
- Verify build args are set correctly
- Consider using persistent volumes

### Out of Memory
- Reduce `MAX_ENGINE_WORKERS`
- Use smaller models
- Increase memory allocation

## API Testing

After deployment, test your API:

```bash
# Get your app URL from CapRover
APP_URL="https://stt-service.your-domain.com"

# Test health
curl $APP_URL/health

# List available engines (includes benchmark results if run)
curl $APP_URL/api/v1/engines

# Run benchmark on all available engines
curl -X POST $APP_URL/api/v1/benchmark

# Get latest benchmark results
curl $APP_URL/api/v1/benchmark

# Test transcription
curl -X POST $APP_URL/api/v1/stt \
  -F "audio=@test.mp3" \
  -F "engine=coqui"
```

## Security Notes

- CapRover handles SSL certificates automatically
- The container runs as non-root user (uid 1000)
- No external API calls (all processing is offline)
- Audio files are processed in memory only

## Optimization Tips

1. **Pre-build Images**: Build your Docker image locally and push to Docker Hub for faster deployments
2. **Use CDN**: For static model files, consider using a CDN
3. **Horizontal Scaling**: Create multiple app instances for load balancing
4. **Cache Models**: Use persistent volumes to avoid re-downloading models

## Support

- Check application logs in CapRover dashboard
- Review build logs for compilation errors
- Ensure all required build arguments are set
- Verify environment variables are configured correctly