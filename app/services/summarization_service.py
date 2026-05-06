"""
Summarization Service

Uses HuggingFace Transformers pipeline with:
  - BART-large-CNN (default, best quality)
  - T5-small (lightweight, fast)
  - PEGASUS-XSum (abstractive, concise)

Long documents are split into chunks and hierarchically summarised.
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)

_loaded_models: dict[str, Any] = {}


# ── Model loading ─────────────────────────────────────────────────────────

def _get_summarizer(model_name: str):
    if model_name not in _loaded_models:
        try:
            from transformers import pipeline  # type: ignore
            logger.info(f"Loading summarization model: {model_name}")
            _loaded_models[model_name] = pipeline(
                "summarization",
                model=model_name,
                device=-1,
            )
            logger.info(f"Summarization model loaded: {model_name}")
        except Exception as exc:
            logger.error(f"Failed to load model {model_name}: {exc}")
            raise RuntimeError(f"Could not load model '{model_name}': {exc}") from exc
    return _loaded_models[model_name]


# ── Text utilities ────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_into_chunks(
    text: str,
    max_words: int = 800,
    overlap_words: int = 50,
) -> list[str]:
    """
    Split text into overlapping word-level chunks suitable for BART/T5/PEGASUS.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    stride = max_words - overlap_words
    idx = 0
    while idx < len(words):
        chunk = " ".join(words[idx : idx + max_words])
        chunks.append(chunk)
        idx += stride

    return chunks


# ── T5 prompt prefix ──────────────────────────────────────────────────────

def _add_model_prefix(text: str, model_name: str) -> str:
    """T5 requires a task prefix; BART/PEGASUS do not."""
    if "t5" in model_name.lower():
        return f"summarize: {text}"
    return text


# ── Hierarchical summarisation ────────────────────────────────────────────

def _summarise_chunks(
    chunks: list[str],
    summarizer,
    model_name: str,
    max_length: int,
    min_length: int,
    num_beams: int,
    do_sample: bool,
    length_penalty: float,
) -> str:
    """
    Summarise each chunk, then summarise the combined intermediate summaries
    if there are multiple chunks (hierarchical approach).
    """
    intermediate: list[str] = []

    chunk_max = min(max_length, 150)
    chunk_min = min(min_length, 40)

    for chunk in chunks:
        prompt = _add_model_prefix(chunk, model_name)
        result = summarizer(
            prompt,
            max_length=chunk_max,
            min_length=chunk_min,
            num_beams=num_beams,
            do_sample=do_sample,
            length_penalty=length_penalty,
            truncation=True,
        )
        intermediate.append(result[0]["summary_text"].strip())

    if len(intermediate) == 1:
        return intermediate[0]

    # Second pass: summarise the intermediate summaries
    combined = " ".join(intermediate)
    prompt = _add_model_prefix(combined, model_name)
    result = summarizer(
        prompt,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        do_sample=do_sample,
        length_penalty=length_penalty,
        truncation=True,
    )
    return result[0]["summary_text"].strip()


# ── Public API ────────────────────────────────────────────────────────────

async def run_summarization(
    text: str,
    model_name: str = "facebook/bart-large-cnn",
    max_length: int = 150,
    min_length: int = 40,
    do_sample: bool = False,
    num_beams: int = 4,
    length_penalty: float = 2.0,
) -> dict:
    """
    Summarise input text.

    Args:
        text:           Input document text
        model_name:     HuggingFace model identifier
        max_length:     Max tokens in summary
        min_length:     Min tokens in summary
        do_sample:      Enable sampling (adds variety, reduces determinism)
        num_beams:      Beam search width
        length_penalty: >1 encourages longer outputs; <1 shorter

    Returns:
        dict with summary, lengths, compression ratio, model info, timing
    """
    t0 = time.perf_counter()

    text = _clean_text(text)
    summarizer = _get_summarizer(model_name)

    chunks = _split_into_chunks(text)
    logger.info(f"Summarising {len(text)} chars in {len(chunks)} chunk(s) with {model_name}")

    summary = _summarise_chunks(
        chunks,
        summarizer,
        model_name=model_name,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        do_sample=do_sample,
        length_penalty=length_penalty,
    )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    original_len = len(text)
    summary_len = len(summary)
    compression = round(1.0 - (summary_len / original_len), 4) if original_len else 0.0

    return {
        "summary":             summary,
        "original_length":     original_len,
        "summary_length":      summary_len,
        "compression_ratio":   compression,
        "model_used":          model_name,
        "processing_time_ms":  elapsed_ms,
    }
