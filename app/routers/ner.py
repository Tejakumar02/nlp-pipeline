"""
NER Router — /api/v1/ner
Named Entity Recognition endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query

from app.models.schemas import NERModel, NERRequest, NERResponse
from app.services.ner_service import LABEL_DESCRIPTIONS, run_ner
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/extract",
    response_model=NERResponse,
    summary="Extract named entities from text",
    description=(
        "Identify and classify named entities (persons, organisations, locations, etc.) "
        "in the provided text using a HuggingFace Transformers NER model. "
        "Long texts are automatically chunked."
    ),
)
async def ner_extract(request: NERRequest = Body(...)):
    logger.info(
        f"NER request: model={request.model}, "
        f"text_len={len(request.text)}, threshold={request.threshold}"
    )

    try:
        result = await run_ner(
            text=request.text,
            model_name=request.model,
            aggregation_strategy=request.aggregation_strategy,
            threshold=request.threshold,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"NER failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="NER processing failed.") from exc

    return NERResponse(**result)


@router.get(
    "/models",
    summary="List available NER models",
)
async def list_models():
    return {
        "models": [m.value for m in NERModel],
        "default": NERModel.bert_base.value,
        "descriptions": {
            NERModel.bert_base.value:  "Fast, good accuracy. Best for most use-cases.",
            NERModel.bert_large.value: "Higher accuracy, slower. For production quality.",
            NERModel.roberta.value:    "State-of-the-art RoBERTa NER model.",
        },
    }


@router.get(
    "/entity-types",
    summary="List entity type descriptions",
)
async def entity_types():
    return {"entity_types": LABEL_DESCRIPTIONS}


@router.post(
    "/extract/batch",
    summary="Batch NER extraction",
    description="Run NER on a list of texts. Returns results in the same order.",
)
async def ner_batch(
    texts: list[str] = Body(..., description="List of text strings to process"),
    model: NERModel = Query(NERModel.bert_base),
    threshold: float = Query(0.85, ge=0.0, le=1.0),
):
    if len(texts) > 50:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 50 texts per request.",
        )

    results = []
    for text in texts:
        if not text.strip():
            results.append({"error": "empty_text", "entities": []})
            continue
        try:
            r = await run_ner(text=text, model_name=model, threshold=threshold)
            results.append(r)
        except Exception as exc:
            results.append({"error": str(exc), "entities": []})

    return {"results": results, "count": len(results)}
