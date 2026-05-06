"""
Multi-Model NLP Document Processing Pipeline
FastAPI application entry point
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.routers import ocr, ner, summarization, pipeline
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: warm up models on startup."""
    logger.info("Starting NLP Document Pipeline...")
    logger.info("Models will be loaded lazily on first request.")
    yield
    logger.info("Shutting down NLP Document Pipeline.")


app = FastAPI(
    title="Multi-Model NLP Document Pipeline",
    description=(
        "A production-grade document processing pipeline integrating "
        "OCR (PaddleOCR / Tesseract), Named Entity Recognition (HuggingFace), "
        "and Summarization (HuggingFace) via FastAPI."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ──────────────────────────────────────────────────────────────

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(elapsed)
    return response


# ── Global exception handler ────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": str(exc),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# ── Routers ─────────────────────────────────────────────────────────────────

app.include_router(ocr.router,           prefix="/api/v1/ocr",           tags=["OCR"])
app.include_router(ner.router,           prefix="/api/v1/ner",           tags=["NER"])
app.include_router(summarization.router, prefix="/api/v1/summarization", tags=["Summarization"])
app.include_router(pipeline.router,      prefix="/api/v1/pipeline",      tags=["Full Pipeline"])


# ── Health / Root ────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Multi-Model NLP Document Pipeline",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "ocr":           "/api/v1/ocr",
            "ner":           "/api/v1/ner",
            "summarization": "/api/v1/summarization",
            "pipeline":      "/api/v1/pipeline",
            "docs":          "/docs",
        },
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}
