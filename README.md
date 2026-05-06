  # Multi-Model NLP Document Pipeline

A production-grade document processing pipeline that integrates **OCR**, **Named Entity Recognition**, and **Summarization** into a single FastAPI service.

```
Image/Document
     │
     ▼
┌─────────────┐    ┌─────────────────┐    ┌────────────────────┐
│  OCR Stage  │───▶│   NER Stage     │───▶│ Summarization Stage│
│             │    │                 │    │                    │
│ PaddleOCR   │    │ BERT-base-NER   │    │ BART-large-CNN     │
│ Tesseract   │    │ BERT-large-NER  │    │ T5-small           │
│ (auto-fall) │    │ RoBERTa NER     │    │ PEGASUS-XSum       │
└─────────────┘    └─────────────────┘    └────────────────────┘
     │                    │                        │
     ▼                    ▼                        ▼
Extracted Text      Entity List              Summary Text
+ Confidence        + Entity Groups          + Compression Ratio
+ BBox (optional)   + Entity Types           + Processing Time
```

---

## Features

| Feature | Detail |
|---|---|
| **OCR** | PaddleOCR (primary) + Tesseract (fallback). Auto-mode tries paddle first. |
| **Image preprocessing** | Denoise, contrast enhancement, sharpening, grayscale conversion. |
| **NER** | dslim/bert-base-NER, bert-large-NER, roberta-large-NER via HuggingFace Transformers. |
| **Long text chunking** | Both NER and summarization automatically split texts >450 / >800 words. |
| **Hierarchical summarization** | Long documents are chunked and summarised in two passes. |
| **Batch endpoints** | `/ner/extract/batch` (50 texts) and `/summarization/summarize/batch` (20 texts). |
| **Text-only pipeline** | `/pipeline/process/text` — skip OCR, run NER + summarization on raw text. |
| **Structured JSON** | Every endpoint returns typed, validated Pydantic schemas. |
| **Request IDs** | `X-Request-ID` and `X-Process-Time-Ms` headers on every response. |
| **Per-stage error isolation** | One stage failing doesn't block the rest of the pipeline. |

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/Tejakumar02/nlp-pipeline.git
cd nlp-pipeline
pip install -r requirements.txt
```

### 2. Install Tesseract (system package)

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract

# Windows
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

### 3. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

---

## Docker

```bash
cd docker
docker compose up --build
```

Models are downloaded on first request and cached in a named volume.

---

## API Endpoints

### OCR

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/ocr/extract` | Extract text from uploaded image |
| `GET` | `/api/v1/ocr/engines` | List available OCR engines |

```bash
# Extract text with PaddleOCR
curl -X POST http://localhost:8000/api/v1/ocr/extract \
  -F "file=@invoice.png" \
  -F "engine=paddle" \
  -F "enhance_image=true"
```

Response:
```json
{
  "text": "INVOICE\nDate: 2024-01-15\nAmount: $1,250.00",
  "confidence": 0.9734,
  "engine_used": "paddle",
  "word_count": 7,
  "char_count": 42,
  "language_detected": "Latin",
  "processing_time_ms": 312.5
}
```

### NER

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/ner/extract` | Extract entities from text |
| `POST` | `/api/v1/ner/extract/batch` | Batch entity extraction |
| `GET` | `/api/v1/ner/models` | List NER models |
| `GET` | `/api/v1/ner/entity-types` | Entity type descriptions |

```bash
curl -X POST http://localhost:8000/api/v1/ner/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Tim Cook visited Berlin to meet Angela Merkel at the Bundestag.", "threshold": 0.85}'
```

Response:
```json
{
  "entities": [
    {"text": "Tim Cook",     "label": "PER", "score": 0.9987, "start": 0,  "end": 8},
    {"text": "Berlin",       "label": "LOC", "score": 0.9945, "start": 17, "end": 23},
    {"text": "Angela Merkel","label": "PER", "score": 0.9991, "start": 32, "end": 45},
    {"text": "Bundestag",    "label": "LOC", "score": 0.9878, "start": 53, "end": 62}
  ],
  "entity_groups": {
    "PER": {"label": "PER", "count": 2, "entities": [...]},
    "LOC": {"label": "LOC", "count": 2, "entities": [...]}
  },
  "entity_count": 4,
  "unique_entity_types": ["LOC", "PER"],
  "model_used": "dslim/bert-base-NER",
  "processing_time_ms": 87.3
}
```

### Summarization

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/summarization/summarize` | Summarize text |
| `POST` | `/api/v1/summarization/summarize/batch` | Batch summarization |
| `GET` | `/api/v1/summarization/models` | List summarization models |

```bash
curl -X POST http://localhost:8000/api/v1/summarization/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "<long document>", "model": "facebook/bart-large-cnn", "max_length": 200}'
```

### Full Pipeline

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/pipeline/process` | Image → OCR → NER → Summarization |
| `POST` | `/api/v1/pipeline/process/text` | Text → NER → Summarization |

```bash
# Full pipeline on an image
curl -X POST http://localhost:8000/api/v1/pipeline/process \
  -F "file=@contract.png" \
  -F "run_ocr=true" \
  -F "run_ner_stage=true" \
  -F "run_summarization_stage=true" \
  -F "ocr_engine=auto" \
  -F "ner_model=dslim/bert-base-NER" \
  -F "summarization_model=facebook/bart-large-cnn"
```

Response:
```json
{
  "document_id": "a3f9c12e-...",
  "total_processing_time_ms": 2341.5,
  "stages_executed": ["ocr", "ner", "summarization"],
  "ocr": {
    "success": true,
    "data": {"text": "...", "confidence": 0.94, ...},
    "processing_time_ms": 310.2
  },
  "ner": {
    "success": true,
    "data": {"entities": [...], "entity_count": 12, ...},
    "processing_time_ms": 94.1
  },
  "summarization": {
    "success": true,
    "data": {"summary": "...", "compression_ratio": 0.88, ...},
    "processing_time_ms": 1937.2
  },
  "metadata": {
    "filename": "contract.png",
    "file_size": 482340,
    "language": "en"
  }
}
```

---

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## Project Structure

```
nlp-pipeline/
├── app/
│   ├── main.py                        # FastAPI app, middleware, router registration
│   ├── models/
│   │   └── schemas.py                 # All Pydantic request/response models
│   ├── routers/
│   │   ├── ocr.py                     # /api/v1/ocr
│   │   ├── ner.py                     # /api/v1/ner
│   │   ├── summarization.py           # /api/v1/summarization
│   │   └── pipeline.py                # /api/v1/pipeline
│   ├── services/
│   │   ├── ocr_service.py             # PaddleOCR + Tesseract + preprocessing
│   │   ├── ner_service.py             # HuggingFace NER + chunking + dedup
│   │   └── summarization_service.py   # HuggingFace Summarization + hierarchical
│   └── utils/
│       └── logger.py                  # Centralised logging
├── tests/
│   └── test_pipeline.py               # Full test suite (mocked ML calls)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Model Notes

| Task | Recommended Model | Notes |
|---|---|---|
| OCR | PaddleOCR (auto) | Best for dense text. Falls back to Tesseract. |
| NER | `dslim/bert-base-NER` | Fast, 4-class CoNLL (PER/ORG/LOC/MISC). |
| NER (high quality) | `dslim/bert-large-NER` | Slower, better accuracy. |
| Summarization | `facebook/bart-large-cnn` | Best overall quality. |
| Summarization (fast) | `t5-small` | 4× faster, lower quality. |
| Summarization (concise) | `google/pegasus-xsum` | Very short abstracts. |

All HuggingFace models are downloaded on first use and cached in `~/.cache/huggingface/`.
