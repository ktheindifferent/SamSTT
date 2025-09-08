FROM python:3.9 as build

# Cache bust to ensure fresh builds
ARG CACHEBUST=1
RUN echo "Cache bust: ${CACHEBUST}"

RUN pip install -U pip virtualenv \
 && virtualenv -p `which python3` /venv/

ENV PATH=/venv/bin/:$PATH

ADD ./requirements.pip /requirements.pip
RUN pip install -r /requirements.pip

# Install optional STT engines based on build args
ARG INSTALL_WHISPER=false
ARG INSTALL_VOSK=false
ARG INSTALL_COQUI=true
ARG INSTALL_SILERO=false
ARG INSTALL_WAV2VEC2=false
ARG DOWNLOAD_COQUI_MODEL=false

# Install Coqui by default or when explicitly requested
RUN if [ "$INSTALL_COQUI" = "true" ] || [ "$DOWNLOAD_COQUI_MODEL" = "true" ]; then pip install STT; fi
RUN if [ "$INSTALL_WHISPER" = "true" ]; then pip install openai-whisper; fi
RUN if [ "$INSTALL_VOSK" = "true" ]; then pip install vosk; fi
RUN if [ "$INSTALL_SILERO" = "true" ]; then pip install torch torchaudio omegaconf; fi
RUN if [ "$INSTALL_WAV2VEC2" = "true" ]; then pip install transformers torch librosa; fi

FROM python:3.9

RUN apt-get update \
 && apt-get install --no-install-recommends -y ffmpeg wget \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

# Download default models (optional - can be mounted as volumes instead)
ARG DOWNLOAD_DEEPSPEECH_MODEL=false
ARG DOWNLOAD_COQUI_MODEL=true
ARG DOWNLOAD_VOSK_MODEL=false

# DeepSpeech model
RUN if [ "$DOWNLOAD_DEEPSPEECH_MODEL" = "true" ]; then \
    wget --progress=dot:giga --tries=3 --timeout=30 \
    https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm \
    -O /app/model.pbmm || \
    echo "Warning: Failed to download DeepSpeech model"; \
    fi

# Coqui model - always download to model.tflite for backward compatibility
RUN if [ "$DOWNLOAD_COQUI_MODEL" = "true" ]; then \
    wget --progress=dot:giga --tries=3 --timeout=30 \
    https://coqui.gateway.scarf.sh/english/coqui/v1.0.0-huge-vocab/model.tflite \
    -O /app/model.tflite || \
    echo "Warning: Failed to download Coqui model"; \
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

RUN groupadd --gid=1000 app \
 && useradd --uid=1000 --gid=1000 --system app
USER app

COPY --from=build --chown=app:app /venv/ /venv/
ENV PATH=/venv/bin/:$PATH

COPY --chown=app:app ./stts/ /app/stts/
WORKDIR /app

# Default environment variables
ENV STT_ENGINE=coqui
ENV MAX_ENGINE_WORKERS=2
ENV LOG_LEVEL=INFO
ENV WHISPER_MODEL_SIZE=base
ENV WHISPER_DEVICE=cpu

EXPOSE 8000

# Health check for CapRover
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD python -m stts.app