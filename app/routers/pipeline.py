"""
Full Pipeline Router — /api/v1/pipeline

Combines OCR → NER → Summarization in a single API call.
Each stage is independently togglable; errors in one stage don't block others.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.models.schemas import (
    NERModel,
    OCREngine,
    PipelineResponse,
    PipelineStageResult,
    SummarizationModel,
)
from app.services.ner_service import run_ner
from app.services.ocr_service import extract_text
from app.services.summarization_service import run_summarization
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@router.post(
    "/process",
    response_model=PipelineResponse,
    summary="Full document processing pipeline",
    description=(
        "Upload an image document and run it through the full pipeline:\n\n"
        "1. **OCR** — Extract text (PaddleOCR / Tesseract)\n"
        "2. **NER** — Identify named entities (HuggingFace BERT/RoBERTa)\n"
        "3. **Summarization** — Generate a concise summary (BART / T5 / PEGASUS)\n\n"
        "Each stage can be individually enabled or disabled. "
        "Stage errors are captured and do not block subsequent stages."
    ),
)
async def process_document(
    file: UploadFile = File(..., description="Image file to process"),
    # OCR options
    run_ocr: bool = Form(True),
    ocr_engine: OCREngine = Form(OCREngine.auto),
    language: str = Form("en"),
    enhance_image: bool = Form(True),
    # NER options
    run_ner_stage: bool = Form(True),
    ner_model: NERModel = Form(NERModel.bert_base),
    ner_threshold: float = Form(0.85),
    # Summarization options
    run_summarization_stage: bool = Form(True),
    summarization_model: SummarizationModel = Form(SummarizationModel.bart),
    summary_max_length: int = Form(150),
    summary_min_length: int = Form(40),
):
    pipeline_start = time.perf_counter()
    document_id = str(uuid.uuid4())
    stages_executed: list[str] = []

    # ── Read file ──────────────────────────────────────────────────────────
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB).")
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    logger.info(
        f"[{document_id}] Pipeline started. File: {file.filename}, "
        f"size: {len(image_bytes)} bytes. "
        f"Stages: ocr={run_ocr}, ner={run_ner_stage}, summarization={run_summarization_stage}"
    )

    # ── Stage 1: OCR ──────────────────────────────────────────────────────
    ocr_result: PipelineStageResult | None = None
    extracted_text: str = ""

    if run_ocr:
        stages_executed.append("ocr")
        t_ocr = time.perf_counter()
        try:
            ocr_data = await extract_text(
                image_bytes=image_bytes,
                engine=ocr_engine.value if hasattr(ocr_engine, "value") else ocr_engine,
                language=language,
                enhance_image=enhance_image,
            )
            extracted_text = ocr_data.get("text", "")
            ocr_result = PipelineStageResult(
                success=True,
                data=ocr_data,
                processing_time_ms=round((time.perf_counter() - t_ocr) * 1000, 2),
            )
            logger.info(f"[{document_id}] OCR complete: {len(extracted_text)} chars extracted.")
        except Exception as exc:
            logger.error(f"[{document_id}] OCR failed: {exc}")
            ocr_result = PipelineStageResult(
                success=False,
                error=str(exc),
                processing_time_ms=round((time.perf_counter() - t_ocr) * 1000, 2),
            )

    # ── Stage 2: NER ──────────────────────────────────────────────────────
    ner_result: PipelineStageResult | None = None

    if run_ner_stage:
        stages_executed.append("ner")
        t_ner = time.perf_counter()

        text_for_ner = extracted_text
        if not text_for_ner.strip():
            ner_result = PipelineStageResult(
                success=False,
                error="No text available for NER (OCR produced no output or OCR was disabled).",
                processing_time_ms=0.0,
            )
        else:
            try:
                ner_data = await run_ner(
                    text=text_for_ner,
                    model_name=ner_model.value if hasattr(ner_model, "value") else ner_model,
                    threshold=ner_threshold,
                )
                ner_result = PipelineStageResult(
                    success=True,
                    data=ner_data,
                    processing_time_ms=round((time.perf_counter() - t_ner) * 1000, 2),
                )
                logger.info(
                    f"[{document_id}] NER complete: {ner_data['entity_count']} entities found."
                )
            except Exception as exc:
                logger.error(f"[{document_id}] NER failed: {exc}")
                ner_result = PipelineStageResult(
                    success=False,
                    error=str(exc),
                    processing_time_ms=round((time.perf_counter() - t_ner) * 1000, 2),
                )

    # ── Stage 3: Summarization ────────────────────────────────────────────
    summarization_result: PipelineStageResult | None = None

    if run_summarization_stage:
        stages_executed.append("summarization")
        t_sum = time.perf_counter()

        text_for_sum = extracted_text
        if not text_for_sum.strip():
            summarization_result = PipelineStageResult(
                success=False,
                error="No text available for summarization.",
                processing_time_ms=0.0,
            )
        elif len(text_for_sum.split()) < 30:
            summarization_result = PipelineStageResult(
                success=False,
                error=f"Text too short for summarization ({len(text_for_sum.split())} words; minimum 30).",
                processing_time_ms=0.0,
            )
        else:
            try:
                sum_data = await run_summarization(
                    text=text_for_sum,
                    model_name=(
                        summarization_model.value
                        if hasattr(summarization_model, "value")
                        else summarization_model
                    ),
                    max_length=summary_max_length,
                    min_length=summary_min_length,
                )
                summarization_result = PipelineStageResult(
                    success=True,
                    data=sum_data,
                    processing_time_ms=round((time.perf_counter() - t_sum) * 1000, 2),
                )
                logger.info(
                    f"[{document_id}] Summarization complete: "
                    f"compression_ratio={sum_data['compression_ratio']}"
                )
            except Exception as exc:
                logger.error(f"[{document_id}] Summarization failed: {exc}")
                summarization_result = PipelineStageResult(
                    success=False,
                    error=str(exc),
                    processing_time_ms=round((time.perf_counter() - t_sum) * 1000, 2),
                )

    total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
    logger.info(f"[{document_id}] Pipeline complete in {total_ms} ms.")

    return PipelineResponse(
        document_id=document_id,
        total_processing_time_ms=total_ms,
        stages_executed=stages_executed,
        ocr=ocr_result,
        ner=ner_result,
        summarization=summarization_result,
        metadata={
            "filename":  file.filename,
            "file_size": len(image_bytes),
            "language":  language,
        },
    )


@router.post(
    "/process/text",
    summary="Run NER + Summarization on raw text (no OCR)",
    description=(
        "If you already have extracted text, you can skip OCR and run "
        "NER and/or Summarization directly."
    ),
)
async def process_text(
    text: str = Form(..., min_length=1),
    run_ner_stage: bool = Form(True),
    run_summarization_stage: bool = Form(True),
    ner_model: NERModel = Form(NERModel.bert_base),
    ner_threshold: float = Form(0.85),
    summarization_model: SummarizationModel = Form(SummarizationModel.bart),
    summary_max_length: int = Form(150),
    summary_min_length: int = Form(40),
):
    pipeline_start = time.perf_counter()
    document_id = str(uuid.uuid4())
    stages_executed: list[str] = []

    ner_result = None
    summarization_result = None

    if run_ner_stage:
        stages_executed.append("ner")
        t = time.perf_counter()
        try:
            ner_data = await run_ner(
                text=text,
                model_name=ner_model.value if hasattr(ner_model, "value") else ner_model,
                threshold=ner_threshold,
            )
            ner_result = PipelineStageResult(
                success=True,
                data=ner_data,
                processing_time_ms=round((time.perf_counter() - t) * 1000, 2),
            )
        except Exception as exc:
            ner_result = PipelineStageResult(
                success=False,
                error=str(exc),
                processing_time_ms=round((time.perf_counter() - t) * 1000, 2),
            )

    if run_summarization_stage:
        stages_executed.append("summarization")
        t = time.perf_counter()
        try:
            sum_data = await run_summarization(
                text=text,
                model_name=(
                    summarization_model.value
                    if hasattr(summarization_model, "value")
                    else summarization_model
                ),
                max_length=summary_max_length,
                min_length=summary_min_length,
            )
            summarization_result = PipelineStageResult(
                success=True,
                data=sum_data,
                processing_time_ms=round((time.perf_counter() - t) * 1000, 2),
            )
        except Exception as exc:
            summarization_result = PipelineStageResult(
                success=False,
                error=str(exc),
                processing_time_ms=round((time.perf_counter() - t) * 1000, 2),
            )

    return PipelineResponse(
        document_id=document_id,
        total_processing_time_ms=round((time.perf_counter() - pipeline_start) * 1000, 2),
        stages_executed=stages_executed,
        ocr=None,
        ner=ner_result,
        summarization=summarization_result,
        metadata={"input_length": len(text)},
    )
