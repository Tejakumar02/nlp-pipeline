"""
OCR Service — wraps PaddleOCR (primary) and Tesseract (fallback/alternative).

Design:
  - Models are loaded lazily (singleton per engine) to avoid cold-start penalties.
  - Image preprocessing (denoise, deskew, adaptive threshold) improves accuracy.
  - Both engines are normalised to a common output schema.
"""

from __future__ import annotations

import io
import time
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy singletons
_paddle_ocr = None
_tesseract_available: Optional[bool] = None


# ── Engine initialisation ─────────────────────────────────────────────────

def _get_paddle_ocr(lang: str = "en"):
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
            _paddle_ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
            logger.info("PaddleOCR initialised.")
        except ImportError:
            logger.warning("PaddleOCR not installed — will use Tesseract.")
            _paddle_ocr = False   # mark as unavailable
    return _paddle_ocr if _paddle_ocr is not False else None


def _is_tesseract_available() -> bool:
    global _tesseract_available
    if _tesseract_available is None:
        try:
            import pytesseract  # type: ignore
            pytesseract.get_tesseract_version()
            _tesseract_available = True
            logger.info("Tesseract available.")
        except Exception:
            _tesseract_available = False
            logger.warning("Tesseract not available.")
    return _tesseract_available


# ── Image preprocessing ───────────────────────────────────────────────────

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Apply standard preprocessing to improve OCR accuracy:
      1. Convert to RGB
      2. Upscale very small images
      3. Increase contrast
      4. Sharpen
      5. Convert to grayscale
      6. Adaptive threshold → binary
    """
    img = image.convert("RGB")

    # Upscale if too small
    w, h = img.size
    if w < 800 or h < 600:
        scale = max(800 / w, 600 / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Contrast + sharpness
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    # Grayscale
    gray = img.convert("L")

    # Denoise
    gray = gray.filter(ImageFilter.MedianFilter(size=3))

    return gray


# ── PaddleOCR extraction ──────────────────────────────────────────────────

def _paddle_extract(image: Image.Image, lang: str = "en") -> dict:
    ocr = _get_paddle_ocr(lang)
    if ocr is None:
        raise RuntimeError("PaddleOCR is not available.")

    img_array = np.array(image)
    result = ocr.ocr(img_array, cls=True)

    lines, confidences, boxes = [], [], []
    if result and result[0]:
        for line in result[0]:
            bbox, (text, conf) = line
            if text.strip():
                lines.append(text.strip())
                confidences.append(float(conf))
                boxes.append({
                    "text": text.strip(),
                    "confidence": round(float(conf), 4),
                    "bbox": [[round(p[0]), round(p[1])] for p in bbox],
                })

    full_text = "\n".join(lines)
    avg_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0
    return {"text": full_text, "confidence": avg_conf, "bounding_boxes": boxes}


# ── Tesseract extraction ──────────────────────────────────────────────────

def _tesseract_extract(image: Image.Image, lang: str = "eng") -> dict:
    if not _is_tesseract_available():
        raise RuntimeError("Tesseract is not installed or not in PATH.")

    import pytesseract  # type: ignore

    # Tesseract uses 3-char lang codes
    lang_map = {"en": "eng", "fr": "fra", "de": "deu", "es": "spa", "zh": "chi_sim"}
    tess_lang = lang_map.get(lang, "eng")

    data = pytesseract.image_to_data(
        image,
        lang=tess_lang,
        output_type=pytesseract.Output.DICT,
    )

    words, confidences, boxes = [], [], []
    for i, conf in enumerate(data["conf"]):
        try:
            conf_val = int(conf)
        except (ValueError, TypeError):
            continue
        if conf_val > 0 and data["text"][i].strip():
            words.append(data["text"][i].strip())
            confidences.append(conf_val / 100.0)
            boxes.append({
                "text": data["text"][i].strip(),
                "confidence": round(conf_val / 100.0, 4),
                "bbox": {
                    "x": data["left"][i],
                    "y": data["top"][i],
                    "w": data["width"][i],
                    "h": data["height"][i],
                },
            })

    full_text = pytesseract.image_to_string(image, lang=tess_lang).strip()
    avg_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0
    return {"text": full_text, "confidence": avg_conf, "bounding_boxes": boxes}


# ── Detect language hint from Tesseract ──────────────────────────────────

def _detect_language(image: Image.Image) -> Optional[str]:
    """Best-effort language detection using Tesseract's OSD."""
    try:
        import pytesseract  # type: ignore
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        return osd.get("script_confidence") and osd.get("script")
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────

async def extract_text(
    image_bytes: bytes,
    engine: str = "auto",
    language: str = "en",
    enhance_image: bool = True,
) -> dict:
    """
    Extract text from image bytes.

    Args:
        image_bytes: Raw image data (JPEG, PNG, TIFF, BMP, …)
        engine: "paddle" | "tesseract" | "auto"
        language: ISO 639-1 code
        enhance_image: Whether to apply preprocessing

    Returns:
        dict with keys: text, confidence, engine_used, bounding_boxes,
                        word_count, char_count, language_detected, processing_time_ms
    """
    t0 = time.perf_counter()

    image = Image.open(io.BytesIO(image_bytes))
    if enhance_image:
        image = preprocess_image(image)

    result = None
    engine_used = engine

    if engine in ("paddle", "auto"):
        try:
            result = _paddle_extract(image, lang=language)
            engine_used = "paddle"
        except Exception as exc:
            logger.warning(f"PaddleOCR failed ({exc}); falling back to Tesseract.")
            if engine == "paddle":
                raise

    if result is None and engine in ("tesseract", "auto"):
        result = _tesseract_extract(image, lang=language)
        engine_used = "tesseract"

    text = result["text"]
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "text": text,
        "confidence": result["confidence"],
        "engine_used": engine_used,
        "word_count": len(text.split()),
        "char_count": len(text),
        "language_detected": _detect_language(image),
        "processing_time_ms": elapsed_ms,
        "bounding_boxes": result.get("bounding_boxes"),
    }
