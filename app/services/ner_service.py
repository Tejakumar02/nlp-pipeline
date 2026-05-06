"""
Named Entity Recognition Service

Uses HuggingFace Transformers pipeline with lazy model loading.
Supports multiple model choices (BERT-base NER, BERT-large NER, RoBERTa NER).

Entity types (CoNLL-2003 standard):
  PER   — Person names
  ORG   — Organisations
  LOC   — Locations
  MISC  — Miscellaneous (events, nationalities, products, etc.)
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Model registry: model_name → loaded pipeline (or None if not yet loaded)
_loaded_models: dict[str, Any] = {}

# Human-readable label mapping
LABEL_DESCRIPTIONS = {
    "PER":  "Person",
    "ORG":  "Organisation",
    "LOC":  "Location",
    "MISC": "Miscellaneous",
    "GPE":  "Geopolitical Entity",
    "FAC":  "Facility",
    "PRODUCT": "Product",
    "EVENT": "Event",
    "NORP": "Nationality / Religion / Political group",
    "LAW":  "Law",
    "LANGUAGE": "Language",
    "MONEY": "Money",
    "PERCENT": "Percentage",
    "DATE": "Date",
    "TIME": "Time",
    "QUANTITY": "Quantity",
}


# ── Model loading ─────────────────────────────────────────────────────────

def _get_ner_pipeline(model_name: str):
    if model_name not in _loaded_models:
        try:
            from transformers import pipeline  # type: ignore
            logger.info(f"Loading NER model: {model_name}")
            _loaded_models[model_name] = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple",
                device=-1,  # CPU; set to 0 for GPU
            )
            logger.info(f"NER model loaded: {model_name}")
        except Exception as exc:
            logger.error(f"Failed to load NER model {model_name}: {exc}")
            raise RuntimeError(f"Could not load NER model '{model_name}': {exc}") from exc
    return _loaded_models[model_name]


# ── Text chunking ─────────────────────────────────────────────────────────

def _chunk_text(text: str, max_tokens: int = 400) -> list[tuple[str, int]]:
    """
    Split text into overlapping chunks so BERT's 512-token limit is respected.
    Returns list of (chunk_text, char_offset) tuples.
    """
    words = text.split()
    chunks: list[tuple[str, int]] = []
    chunk_size = max_tokens
    stride = max_tokens - 50  # 50-token overlap

    char_offset = 0
    word_idx = 0

    while word_idx < len(words):
        chunk_words = words[word_idx : word_idx + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append((chunk_text, char_offset))
        char_offset += len(chunk_text) + 1
        word_idx += stride

    return chunks


# ── Entity normalisation ──────────────────────────────────────────────────

def _normalize_label(label: str) -> str:
    """Strip B-/I- prefixes and return clean entity type."""
    for prefix in ("B-", "I-", "S-", "E-"):
        if label.startswith(prefix):
            return label[len(prefix):]
    return label


def _normalize_entity_text(text: str, label: str) -> str | None:
    """Light normalization: strip surrounding punctuation."""
    clean = text.strip(".,!?;:\"'()[]{}").strip()
    return clean if clean else None


# ── Deduplication ─────────────────────────────────────────────────────────

def _deduplicate_entities(entities: list[dict]) -> list[dict]:
    """Remove duplicate entities (same text + label, keep highest score)."""
    seen: dict[tuple[str, str], dict] = {}
    for ent in entities:
        key = (ent["text"].lower(), ent["label"])
        if key not in seen or ent["score"] > seen[key]["score"]:
            seen[key] = ent
    return sorted(seen.values(), key=lambda e: e["start"])


# ── Public API ────────────────────────────────────────────────────────────

async def run_ner(
    text: str,
    model_name: str = "dslim/bert-base-NER",
    aggregation_strategy: str = "simple",
    threshold: float = 0.85,
) -> dict:
    """
    Run NER on input text.

    Args:
        text: Input text (up to ~100k chars; chunked automatically)
        model_name: HuggingFace model identifier
        aggregation_strategy: Token aggregation strategy
        threshold: Minimum confidence score to include an entity

    Returns:
        dict with entities, entity_groups, counts, model info, timing
    """
    t0 = time.perf_counter()

    ner_pipeline = _get_ner_pipeline(model_name)

    # Chunk long texts
    chunks = _chunk_text(text) if len(text.split()) > 450 else [(text, 0)]

    raw_entities: list[dict] = []

    for chunk_text, char_offset in chunks:
        try:
            preds = ner_pipeline(
                chunk_text,
                aggregation_strategy=aggregation_strategy,
            )
        except Exception as exc:
            logger.warning(f"NER chunk failed: {exc}")
            continue

        for pred in preds:
            score = float(pred.get("score", 0.0))
            if score < threshold:
                continue

            raw_label = pred.get("entity_group") or pred.get("entity", "")
            label = _normalize_label(raw_label)
            entity_text = pred.get("word", "").strip()

            if not entity_text:
                continue

            raw_entities.append({
                "text":       entity_text,
                "label":      label,
                "score":      round(score, 4),
                "start":      pred.get("start", 0) + char_offset,
                "end":        pred.get("end", 0) + char_offset,
                "normalized": _normalize_entity_text(entity_text, label),
            })

    # Deduplicate
    entities = _deduplicate_entities(raw_entities)

    # Build entity groups
    groups: dict[str, dict] = defaultdict(lambda: {"label": "", "entities": [], "count": 0})
    for ent in entities:
        lbl = ent["label"]
        groups[lbl]["label"] = lbl
        groups[lbl]["entities"].append(ent)
        groups[lbl]["count"] += 1

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "entities":            entities,
        "entity_groups":       dict(groups),
        "entity_count":        len(entities),
        "unique_entity_types": sorted(groups.keys()),
        "model_used":          model_name,
        "processing_time_ms":  elapsed_ms,
    }
