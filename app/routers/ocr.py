"""
OCR Router — /api/v1/ocr
Accepts image uploads and returns extracted text.
"""

from __future__ import annotations

import io

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.models.schemas import OCREngine, OCRResponse
from app.services.ocr_service import extract_text
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
    "image/bmp",
    "image/webp",
    "application/octet-stream",  # allow generic binary
}

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


def _validate_image(file: UploadFile) -> None:
    content_type = file.content_type or ""
    # Be lenient — some clients send wrong content-type
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{content_type}'. "
                   f"Accepted: JPEG, PNG, TIFF, BMP, WebP.",
        )


@router.post(
    "/extract",
    response_model=OCRResponse,
    summary="Extract text from an image",
    description=(
        "Upload an image (JPEG, PNG, TIFF, BMP, WebP) and receive extracted text "
        "with confidence scores and optional bounding boxes. "
        "Supports PaddleOCR (accurate) and Tesseract (lightweight)."
    ),
)
async def ocr_extract(
    file: UploadFile = File(..., description="Image file to process"),
    engine: OCREngine = Query(OCREngine.auto, description="OCR engine to use"),
    language: str = Query("en", description="Document language (ISO 639-1 code)"),
    enhance_image: bool = Query(True, description="Apply image preprocessing"),
    include_bboxes: bool = Query(False, description="Include per-word bounding boxes in response"),
):
    _validate_image(file)

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)} MB.",
        )
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    logger.info(f"OCR request: engine={engine}, lang={language}, size={len(image_bytes)} bytes")

    try:
        result = await extract_text(
            image_bytes=image_bytes,
            engine=engine.value if hasattr(engine, "value") else engine,
            language=language,
            enhance_image=enhance_image,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"OCR failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="OCR processing failed.") from exc

    if not include_bboxes:
        result.pop("bounding_boxes", None)

    return OCRResponse(**result)


@router.get(
    "/engines",
    summary="List available OCR engines",
)
async def list_engines():
    """Return which OCR engines are installed and available."""
    availability = {}

    try:
        from paddleocr import PaddleOCR  # type: ignore  # noqa: F401
        availability["paddle"] = True
    except ImportError:
        availability["paddle"] = False

    try:
        import pytesseract  # type: ignore
        pytesseract.get_tesseract_version()
        availability["tesseract"] = True
    except Exception:
        availability["tesseract"] = False

    return {
        "engines": availability,
        "default": "auto",
        "auto_strategy": "paddle → tesseract fallback",
    }
