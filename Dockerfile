FROM python:3.8 as build

RUN echo "0.0.01"

RUN pip install -U pip virtualenv \
 && virtualenv -p `which python3` /venv/

ENV PATH=/venv/bin/:$PATH

ADD ./requirements.pip /requirements.pip
RUN pip install -r /requirements.pip

FROM python:3.8

RUN apt-get update \
 && apt-get install --no-install-recommends -y ffmpeg wget \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
RUN wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm -O /app/model.pbmm
RUN wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.tflite -O /app/model.tflite


RUN groupadd --gid=1000 app \
 && useradd --uid=1000 --gid=1000 --system app
USER app

COPY --from=build --chown=app:app /venv/ /venv/
ENV PATH=/venv/bin/:$PATH

COPY --chown=app:app ./stt/ /app/stt/
WORKDIR /app


EXPOSE 8000

CMD python -m stt.app
