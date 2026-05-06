# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps: Tesseract + image libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    tesseract-ocr-deu \
    tesseract-ocr-fra \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download HuggingFace models at build time (optional — remove to pull at runtime)
# RUN python -c "from transformers import pipeline; pipeline('ner', model='dslim/bert-base-NER')"
# RUN python -c "from transformers import pipeline; pipeline('summarization', model='facebook/bart-large-cnn')"

# ── App ───────────────────────────────────────────────────────────────────────
COPY app/ ./app/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Graceful startup with pre-fork workers
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--timeout-keep-alive", "30"]
