"""
Pydantic schemas for all API request/response models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────────

class OCREngine(str, Enum):
    paddle = "paddle"
    tesseract = "tesseract"
    auto = "auto"          # tries paddle first, falls back to tesseract


class NERModel(str, Enum):
    bert_base = "dslim/bert-base-NER"
    bert_large = "dslim/bert-large-NER"
    roberta = "Jean-Baptiste/roberta-large-ner-english"


class SummarizationModel(str, Enum):
    bart = "facebook/bart-large-cnn"
    t5_small = "t5-small"
    pegasus = "google/pegasus-xsum"


# ── OCR ─────────────────────────────────────────────────────────────────────

class OCRRequest(BaseModel):
    engine: OCREngine = OCREngine.auto
    language: str = Field(default="en", description="ISO 639-1 language code")
    enhance_image: bool = Field(default=True, description="Apply preprocessing (denoise, deskew)")

    model_config = {"use_enum_values": True}


class OCRResponse(BaseModel):
    text: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    engine_used: str
    word_count: int
    char_count: int
    language_detected: Optional[str] = None
    processing_time_ms: float
    bounding_boxes: Optional[list[dict[str, Any]]] = None


# ── NER ─────────────────────────────────────────────────────────────────────

class NEREntity(BaseModel):
    text: str
    label: str
    score: float = Field(ge=0.0, le=1.0)
    start: int
    end: int
    normalized: Optional[str] = None


class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)
    model: NERModel = NERModel.bert_base
    aggregation_strategy: str = Field(
        default="simple",
        description="Token aggregation strategy: none | simple | first | average | max",
    )
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    model_config = {"use_enum_values": True}

    @field_validator("aggregation_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"none", "simple", "first", "average", "max"}
        if v not in allowed:
            raise ValueError(f"aggregation_strategy must be one of {allowed}")
        return v


class EntityGroup(BaseModel):
    label: str
    entities: list[NEREntity]
    count: int


class NERResponse(BaseModel):
    entities: list[NEREntity]
    entity_groups: dict[str, EntityGroup]
    entity_count: int
    unique_entity_types: list[str]
    model_used: str
    processing_time_ms: float


# ── Summarization ────────────────────────────────────────────────────────────

class SummarizationRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=100_000)
    model: SummarizationModel = SummarizationModel.bart
    max_length: int = Field(default=150, ge=30, le=1024)
    min_length: int = Field(default=40, ge=10, le=512)
    do_sample: bool = False
    num_beams: int = Field(default=4, ge=1, le=10)
    length_penalty: float = Field(default=2.0, ge=0.0, le=5.0)

    model_config = {"use_enum_values": True}

    @field_validator("min_length")
    @classmethod
    def min_less_than_max(cls, v: int, info) -> int:
        max_len = info.data.get("max_length", 150)
        if v >= max_len:
            raise ValueError("min_length must be less than max_length")
        return v


class SummarizationResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    model_used: str
    processing_time_ms: float


# ── Full Pipeline ────────────────────────────────────────────────────────────

class PipelineStages(BaseModel):
    ocr: bool = True
    ner: bool = True
    summarization: bool = True


class PipelineRequest(BaseModel):
    stages: PipelineStages = PipelineStages()
    ocr_options: OCRRequest = OCRRequest()
    ner_options: NERRequest = Field(
        default_factory=lambda: NERRequest(text="placeholder")
    )
    summarization_options: SummarizationRequest = Field(
        default_factory=lambda: SummarizationRequest(text="placeholder text for defaults")
    )


class PipelineStageResult(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    processing_time_ms: float


class PipelineResponse(BaseModel):
    document_id: str
    total_processing_time_ms: float
    stages_executed: list[str]
    ocr: Optional[PipelineStageResult] = None
    ner: Optional[PipelineStageResult] = None
    summarization: Optional[PipelineStageResult] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Error ────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None
