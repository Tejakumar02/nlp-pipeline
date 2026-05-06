"""
Summarization Router — /api/v1/summarization
"""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query

from app.models.schemas import SummarizationModel, SummarizationRequest, SummarizationResponse
from app.services.summarization_service import run_summarization
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/summarize",
    response_model=SummarizationResponse,
    summary="Summarize text",
    description=(
        "Generate an abstractive summary of the input text using a HuggingFace "
        "Transformers model. Supports BART, T5, and PEGASUS. "
        "Long documents are hierarchically summarised."
    ),
)
async def summarize(request: SummarizationRequest = Body(...)):
    logger.info(
        f"Summarization request: model={request.model}, "
        f"text_len={len(request.text)}, "
        f"max_len={request.max_length}"
    )

    try:
        result = await run_summarization(
            text=request.text,
            model_name=request.model,
            max_length=request.max_length,
            min_length=request.min_length,
            do_sample=request.do_sample,
            num_beams=request.num_beams,
            length_penalty=request.length_penalty,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Summarization failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Summarization failed.") from exc

    return SummarizationResponse(**result)


@router.get(
    "/models",
    summary="List available summarization models",
)
async def list_models():
    return {
        "models": [m.value for m in SummarizationModel],
        "default": SummarizationModel.bart.value,
        "descriptions": {
            SummarizationModel.bart.value:    "BART-large-CNN. Best quality, balanced speed.",
            SummarizationModel.t5_small.value: "T5-small. Fastest, lower quality.",
            SummarizationModel.pegasus.value: "PEGASUS-XSum. Concise abstractive summaries.",
        },
        "notes": {
            "chunking":     "Texts >800 words are automatically split and hierarchically summarised.",
            "determinism":  "Set do_sample=false (default) for deterministic output.",
        },
    }


@router.post(
    "/summarize/batch",
    summary="Batch text summarization",
)
async def summarize_batch(
    texts: list[str] = Body(..., description="List of texts to summarise"),
    model: SummarizationModel = Query(SummarizationModel.bart),
    max_length: int = Query(150, ge=30, le=1024),
    min_length: int = Query(40, ge=10),
):
    if len(texts) > 20:
        raise HTTPException(
            status_code=400,
            detail="Batch limited to 20 texts per request.",
        )

    results = []
    for text in texts:
        if not text.strip():
            results.append({"error": "empty_text"})
            continue
        try:
            r = await run_summarization(
                text=text,
                model_name=model,
                max_length=max_length,
                min_length=min_length,
            )
            results.append(r)
        except Exception as exc:
            results.append({"error": str(exc)})

    return {"results": results, "count": len(results)}
