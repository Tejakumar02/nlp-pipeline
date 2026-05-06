"""
Test suite for the Multi-Model NLP Document Pipeline.

Uses httpx AsyncClient so tests run without a live server.
Run with: pytest tests/ -v --asyncio-mode=auto
"""

from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_test_image(text: str = "Hello World") -> bytes:
    """Create a simple in-memory PNG with text."""
    img = Image.new("RGB", (400, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


SAMPLE_TEXT = (
    "Apple Inc. announced today that CEO Tim Cook will visit Berlin next Tuesday. "
    "The company plans to open a new European headquarters in Germany, "
    "according to a statement released in New York. "
    "The investment of $3 billion marks Apple's largest European expansion to date."
)

LONG_TEXT = SAMPLE_TEXT * 20  # ~1400 words for summarization tests


# ── Health endpoints ──────────────────────────────────────────────────────────

def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "endpoints" in data


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


# ── OCR Router ────────────────────────────────────────────────────────────────

class TestOCRRouter:
    def test_list_engines(self):
        resp = client.get("/api/v1/ocr/engines")
        assert resp.status_code == 200
        data = resp.json()
        assert "engines" in data
        assert "paddle" in data["engines"]
        assert "tesseract" in data["engines"]

    @patch("app.routers.ocr.extract_text")
    def test_extract_success(self, mock_extract):
        mock_extract.return_value = {
            "text": "Hello World",
            "confidence": 0.97,
            "engine_used": "paddle",
            "word_count": 2,
            "char_count": 11,
            "language_detected": "Latin",
            "processing_time_ms": 123.4,
            "bounding_boxes": None,
        }

        image_bytes = make_test_image()
        resp = client.post(
            "/api/v1/ocr/extract",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Hello World"
        assert data["engine_used"] == "paddle"
        assert data["word_count"] == 2

    def test_extract_empty_file(self):
        resp = client.post(
            "/api/v1/ocr/extract",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 400

    def test_extract_unsupported_type(self):
        resp = client.post(
            "/api/v1/ocr/extract",
            files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert resp.status_code == 415

    @patch("app.routers.ocr.extract_text", side_effect=RuntimeError("PaddleOCR unavailable"))
    def test_extract_engine_error(self, _):
        image_bytes = make_test_image()
        resp = client.post(
            "/api/v1/ocr/extract",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert resp.status_code == 422


# ── NER Router ────────────────────────────────────────────────────────────────

class TestNERRouter:
    def test_list_models(self):
        resp = client.get("/api/v1/ner/models")
        assert resp.status_code == 200
        assert "models" in resp.json()

    def test_entity_types(self):
        resp = client.get("/api/v1/ner/entity-types")
        assert resp.status_code == 200
        data = resp.json()
        assert "entity_types" in data
        assert "PER" in data["entity_types"]

    @patch("app.routers.ner.run_ner")
    def test_extract_success(self, mock_run):
        mock_run.return_value = {
            "entities": [
                {"text": "Tim Cook", "label": "PER", "score": 0.99, "start": 0, "end": 8, "normalized": "Tim Cook"},
                {"text": "Apple", "label": "ORG", "score": 0.98, "start": 10, "end": 15, "normalized": "Apple"},
            ],
            "entity_groups": {
                "PER": {"label": "PER", "entities": [], "count": 1},
                "ORG": {"label": "ORG", "entities": [], "count": 1},
            },
            "entity_count": 2,
            "unique_entity_types": ["ORG", "PER"],
            "model_used": "dslim/bert-base-NER",
            "processing_time_ms": 45.0,
        }

        resp = client.post(
            "/api/v1/ner/extract",
            json={"text": SAMPLE_TEXT, "model": "dslim/bert-base-NER"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_count"] == 2
        assert "PER" in data["unique_entity_types"]

    def test_extract_empty_text(self):
        resp = client.post(
            "/api/v1/ner/extract",
            json={"text": ""},
        )
        assert resp.status_code == 422

    def test_extract_invalid_aggregation(self):
        resp = client.post(
            "/api/v1/ner/extract",
            json={"text": SAMPLE_TEXT, "aggregation_strategy": "invalid_strategy"},
        )
        assert resp.status_code == 422

    @patch("app.routers.ner.run_ner")
    def test_batch_success(self, mock_run):
        mock_run.return_value = {
            "entities": [], "entity_groups": {}, "entity_count": 0,
            "unique_entity_types": [], "model_used": "dslim/bert-base-NER",
            "processing_time_ms": 10.0,
        }
        resp = client.post(
            "/api/v1/ner/extract/batch",
            json=["Text one.", "Text two."],
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 2

    def test_batch_too_many(self):
        texts = [f"text {i}" for i in range(51)]
        resp = client.post("/api/v1/ner/extract/batch", json=texts)
        assert resp.status_code == 400


# ── Summarization Router ──────────────────────────────────────────────────────

class TestSummarizationRouter:
    def test_list_models(self):
        resp = client.get("/api/v1/summarization/models")
        assert resp.status_code == 200
        assert "models" in resp.json()

    @patch("app.routers.summarization.run_summarization")
    def test_summarize_success(self, mock_sum):
        mock_sum.return_value = {
            "summary": "Apple plans European expansion.",
            "original_length": len(SAMPLE_TEXT),
            "summary_length": 30,
            "compression_ratio": 0.87,
            "model_used": "facebook/bart-large-cnn",
            "processing_time_ms": 800.0,
        }
        resp = client.post(
            "/api/v1/summarization/summarize",
            json={"text": SAMPLE_TEXT, "model": "facebook/bart-large-cnn"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert data["compression_ratio"] == 0.87

    def test_summarize_text_too_short(self):
        resp = client.post(
            "/api/v1/summarization/summarize",
            json={"text": "Too short."},
        )
        assert resp.status_code == 422

    def test_summarize_min_gt_max(self):
        resp = client.post(
            "/api/v1/summarization/summarize",
            json={"text": SAMPLE_TEXT, "min_length": 200, "max_length": 100},
        )
        assert resp.status_code == 422


# ── Full Pipeline Router ──────────────────────────────────────────────────────

class TestPipelineRouter:
    @patch("app.routers.pipeline.extract_text")
    @patch("app.routers.pipeline.run_ner")
    @patch("app.routers.pipeline.run_summarization")
    def test_full_pipeline_success(self, mock_sum, mock_ner, mock_ocr):
        mock_ocr.return_value = {
            "text": SAMPLE_TEXT,
            "confidence": 0.95,
            "engine_used": "paddle",
            "word_count": 50,
            "char_count": len(SAMPLE_TEXT),
            "language_detected": "Latin",
            "processing_time_ms": 100.0,
            "bounding_boxes": None,
        }
        mock_ner.return_value = {
            "entities": [{"text": "Apple", "label": "ORG", "score": 0.99, "start": 0, "end": 5, "normalized": "Apple"}],
            "entity_groups": {"ORG": {"label": "ORG", "entities": [], "count": 1}},
            "entity_count": 1,
            "unique_entity_types": ["ORG"],
            "model_used": "dslim/bert-base-NER",
            "processing_time_ms": 50.0,
        }
        mock_sum.return_value = {
            "summary": "Apple expands in Europe.",
            "original_length": len(SAMPLE_TEXT),
            "summary_length": 24,
            "compression_ratio": 0.88,
            "model_used": "facebook/bart-large-cnn",
            "processing_time_ms": 900.0,
        }

        image_bytes = make_test_image()
        resp = client.post(
            "/api/v1/pipeline/process",
            data={"run_ocr": "true", "run_ner_stage": "true", "run_summarization_stage": "true"},
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ocr"]["success"] is True
        assert data["ner"]["success"] is True
        assert data["summarization"]["success"] is True
        assert "document_id" in data
        assert set(data["stages_executed"]) == {"ocr", "ner", "summarization"}

    @patch("app.routers.pipeline.extract_text", side_effect=RuntimeError("OCR failed"))
    def test_pipeline_ocr_failure_doesnt_crash(self, _):
        """OCR failure should be captured; pipeline returns gracefully."""
        image_bytes = make_test_image()
        resp = client.post(
            "/api/v1/pipeline/process",
            data={"run_ocr": "true", "run_ner_stage": "false", "run_summarization_stage": "false"},
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ocr"]["success"] is False
        assert "OCR failed" in data["ocr"]["error"]

    @patch("app.routers.pipeline.run_ner")
    @patch("app.routers.pipeline.run_summarization")
    def test_process_text_endpoint(self, mock_sum, mock_ner):
        mock_ner.return_value = {
            "entities": [], "entity_groups": {}, "entity_count": 0,
            "unique_entity_types": [], "model_used": "dslim/bert-base-NER",
            "processing_time_ms": 5.0,
        }
        mock_sum.return_value = {
            "summary": "Brief summary.",
            "original_length": len(SAMPLE_TEXT),
            "summary_length": 14,
            "compression_ratio": 0.9,
            "model_used": "facebook/bart-large-cnn",
            "processing_time_ms": 200.0,
        }
        resp = client.post(
            "/api/v1/pipeline/process/text",
            data={"text": SAMPLE_TEXT, "run_ner_stage": "true", "run_summarization_stage": "true"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ocr"] is None
        assert data["ner"]["success"] is True
        assert data["summarization"]["success"] is True


# ── Service unit tests ────────────────────────────────────────────────────────

class TestOCRService:
    @patch("app.services.ocr_service._get_paddle_ocr")
    def test_paddle_extraction(self, mock_get):
        mock_ocr = MagicMock()
        mock_ocr.ocr.return_value = [[
            ([[0,0],[100,0],[100,20],[0,20]], ("Hello World", 0.97)),
        ]]
        mock_get.return_value = mock_ocr

        import asyncio
        from app.services.ocr_service import extract_text

        image = Image.new("RGB", (200, 50), "white")
        buf = io.BytesIO()
        image.save(buf, format="PNG")

        result = asyncio.get_event_loop().run_until_complete(
            extract_text(buf.getvalue(), engine="paddle")
        )
        assert "text" in result
        assert result["engine_used"] == "paddle"


class TestNERService:
    @patch("app.services.ner_service._get_ner_pipeline")
    def test_ner_extraction(self, mock_get):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"entity_group": "PER", "word": "Tim Cook", "score": 0.99, "start": 0, "end": 8},
            {"entity_group": "ORG", "word": "Apple", "score": 0.97, "start": 26, "end": 31},
        ]
        mock_get.return_value = mock_pipeline

        import asyncio
        from app.services.ner_service import run_ner

        result = asyncio.get_event_loop().run_until_complete(
            run_ner("Tim Cook is the CEO of Apple.")
        )
        assert result["entity_count"] == 2
        assert "PER" in result["entity_groups"]
        assert "ORG" in result["entity_groups"]

    @patch("app.services.ner_service._get_ner_pipeline")
    def test_ner_threshold_filtering(self, mock_get):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"entity_group": "PER", "word": "John", "score": 0.60, "start": 0, "end": 4},
            {"entity_group": "ORG", "word": "Google", "score": 0.95, "start": 10, "end": 16},
        ]
        mock_get.return_value = mock_pipeline

        import asyncio
        from app.services.ner_service import run_ner

        result = asyncio.get_event_loop().run_until_complete(
            run_ner("John works at Google.", threshold=0.85)
        )
        # John (0.60) should be filtered out; Google (0.95) passes
        assert result["entity_count"] == 1
        assert result["entities"][0]["text"] == "Google"


class TestSummarizationService:
    @patch("app.services.summarization_service._get_summarizer")
    def test_summarization(self, mock_get):
        mock_summarizer = MagicMock()
        mock_summarizer.return_value = [{"summary_text": "Apple expands to Europe."}]
        mock_get.return_value = mock_summarizer

        import asyncio
        from app.services.summarization_service import run_summarization

        result = asyncio.get_event_loop().run_until_complete(
            run_summarization(SAMPLE_TEXT)
        )
        assert result["summary"] == "Apple expands to Europe."
        assert result["compression_ratio"] > 0

    def test_text_cleaning(self):
        from app.services.summarization_service import _clean_text
        dirty = "Hello\x00World  extra   spaces"
        clean = _clean_text(dirty)
        assert "\x00" not in clean
        assert "  " not in clean

    def test_text_chunking_short(self):
        from app.services.summarization_service import _split_into_chunks
        short = "word " * 100
        chunks = _split_into_chunks(short, max_words=800)
        assert len(chunks) == 1

    def test_text_chunking_long(self):
        from app.services.summarization_service import _split_into_chunks
        long = "word " * 2000
        chunks = _split_into_chunks(long, max_words=800)
        assert len(chunks) > 1
