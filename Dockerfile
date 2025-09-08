FROM python:3.9 as build

# Cache bust to ensure fresh builds
ARG CACHEBUST=9
RUN echo "Cache bust: ${CACHEBUST}"

RUN pip install -U pip virtualenv \
 && virtualenv -p `which python3` /venv/

ENV PATH=/venv/bin/:$PATH

# Install build dependencies first
RUN pip install --no-cache-dir \
    wheel \
    setuptools \
    Cython \
    "numpy<2.0.0"

ADD ./requirements.pip /requirements.pip
RUN pip install -r /requirements.pip

# Install optional STT engines based on build args
ARG INSTALL_ALL=false
ARG INSTALL_WHISPER=false
ARG INSTALL_VOSK=false
ARG INSTALL_COQUI=false
ARG INSTALL_SILERO=false
ARG INSTALL_WAV2VEC2=false
ARG INSTALL_SPEECHBRAIN=false
ARG INSTALL_NEMO=false
ARG INSTALL_POCKETSPHINX=false

# Note: stt package (Coqui STT) is installed from requirements.pip

# Install all engines if INSTALL_ALL is true (simplified to avoid build issues)
RUN if [ "$INSTALL_ALL" = "true" ]; then \
    pip install --no-cache-dir pywhispercpp vosk transformers torch librosa && \
    pip install --no-cache-dir pocketsphinx || echo "PocketSphinx install failed, continuing..." && \
    pip install --no-cache-dir torchaudio omegaconf || echo "Silero deps install failed, continuing..."; \
    fi

# Install individual engines if not installing all
RUN if [ "$INSTALL_ALL" != "true" ] && [ "$INSTALL_WHISPER" = "true" ]; then pip install --no-cache-dir pywhispercpp; fi
RUN if [ "$INSTALL_ALL" != "true" ] && [ "$INSTALL_VOSK" = "true" ]; then pip install --no-cache-dir vosk; fi
RUN if [ "$INSTALL_ALL" != "true" ] && [ "$INSTALL_SILERO" = "true" ]; then pip install --no-cache-dir torch torchaudio omegaconf; fi
RUN if [ "$INSTALL_ALL" != "true" ] && [ "$INSTALL_WAV2VEC2" = "true" ]; then pip install --no-cache-dir transformers torch librosa; fi
RUN if [ "$INSTALL_ALL" != "true" ] && [ "$INSTALL_SPEECHBRAIN" = "true" ]; then pip install --no-cache-dir speechbrain || echo "SpeechBrain install failed"; fi
RUN if [ "$INSTALL_ALL" != "true" ] && [ "$INSTALL_NEMO" = "true" ]; then pip install --no-cache-dir nemo_toolkit[asr] || echo "NeMo install failed"; fi
RUN if [ "$INSTALL_ALL" != "true" ] && [ "$INSTALL_POCKETSPHINX" = "true" ]; then pip install --no-cache-dir pocketsphinx || echo "PocketSphinx install failed"; fi

FROM python:3.9

RUN apt-get update \
 && apt-get install --no-install-recommends -y ffmpeg wget \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

# Download default models (optional - can be mounted as volumes instead)
ARG DOWNLOAD_COQUI_MODEL=true
ARG DOWNLOAD_VOSK_MODEL=false

# Download Coqui STT model
RUN if [ "$DOWNLOAD_COQUI_MODEL" = "true" ]; then \
    wget --progress=dot:giga --tries=3 --timeout=30 \
    https://coqui.gateway.scarf.sh/english/coqui/v1.0.0-huge-vocab/model.tflite \
    -O /app/model.tflite || \
    echo "Warning: Failed to download model"; \
    fi

# Vosk model
RUN if [ "$DOWNLOAD_VOSK_MODEL" = "true" ]; then \
    wget --progress=dot:giga --tries=3 --timeout=30 \
    https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip \
    -O /tmp/vosk-model.zip && \
    unzip -q /tmp/vosk-model.zip -d /app/ && \
    mv /app/vosk-model-small-en-us-0.15 /app/vosk_model && \
    rm /tmp/vosk-model.zip || \
    echo "Warning: Failed to download Vosk model"; \
    fi

# Create app user and directories with proper permissions
RUN groupadd --gid=1000 app \
 && useradd --uid=1000 --gid=1000 --system --create-home app \
 && mkdir -p /app/cache /app/results /app/models \
 && chown -R app:app /app /home/app

COPY --from=build --chown=app:app /venv/ /venv/
ENV PATH=/venv/bin/:$PATH

COPY --chown=app:app ./stts/ /app/stts/

# Switch to app user
USER app

WORKDIR /app

# Set environment variables for cache and model paths
ENV HF_HOME=/app/cache/huggingface
ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV PYWHISPERCPP_MODEL_DIR=/app/models/whisper
ENV BENCHMARK_RESULTS_FILE=/app/results/benchmark_results.json

# Default environment variables
ENV STT_ENGINE=coqui
ENV MAX_ENGINE_WORKERS=2
ENV LOG_LEVEL=INFO
ENV WHISPER_MODEL_SIZE=base
ENV WHISPER_DEVICE=cpu
ENV RUN_BENCHMARK_ON_STARTUP=true

EXPOSE 8000

# Health check for CapRover
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD python -m stts.app