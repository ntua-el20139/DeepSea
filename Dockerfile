# syntax=docker/dockerfile:1.4

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    NLTK_DATA=/app/nltk_data

WORKDIR ${APP_HOME}

# System level deps: build tools, ffmpeg for media ingestion, glib/gl for docling OCR, poppler for PDF image rasterization
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libmagic1 \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure writable directories exist for runtime artifacts
RUN mkdir -p data/chunks ${NLTK_DATA}

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.maxUploadSize=1024"]
