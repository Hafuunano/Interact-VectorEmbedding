from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import CLASSIFY_TIMEOUT_SECONDS, EXECUTOR_WORKERS
from emotion_classifier import classify, is_ready as emotion_ready
from embedding import get_encoder
from keywords import extract_keywords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Emotion Classification API", description="Zero-shot emotion + keywords via M3E embedding")

_executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to classify")
    top_keywords: int = Field(default=5, ge=1, le=20, description="Number of keywords to return")


class ClassifyResponse(BaseModel):
    emotion: str
    score: float
    keywords: list[str]


def _classify_and_keywords(text: str, top_keywords: int) -> dict[str, Any]:
    result = classify(text)
    kw = extract_keywords(text, top_k=top_keywords)
    return {"emotion": result.emotion, "score": result.score, "keywords": kw}


@app.post("/classify", response_model=ClassifyResponse)
async def classify_endpoint(req: ClassifyRequest) -> ClassifyResponse:
    """Classify emotion and extract keywords. Runs in thread pool."""
    loop = asyncio.get_event_loop()
    text_len = len(req.text)
    try:
        out = await asyncio.wait_for(
            loop.run_in_executor(
                _executor,
                _classify_and_keywords,
                req.text.strip(),
                req.top_keywords,
            ),
            timeout=CLASSIFY_TIMEOUT_SECONDS,
        )
        logger.info("classify ok len=%s emotion=%s", text_len, out["emotion"])
        return ClassifyResponse(**out)
    except asyncio.TimeoutError:
        logger.warning("classify timeout len=%s", text_len)
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        logger.exception("classify error len=%s: %s", text_len, e)
        raise HTTPException(status_code=500, detail="Classification failed")


@app.get("/health")
async def health() -> dict[str, Any]:
    """Check model and category cache are ready."""
    encoder_loaded = get_encoder() is not None
    categories_ready = emotion_ready()
    ok = encoder_loaded and categories_ready
    return {
        "status": "ok" if ok else "degraded",
        "encoder_loaded": encoder_loaded,
        "categories_ready": categories_ready,
    }


@app.on_event("startup")
async def startup() -> None:
    """Warm encoder and emotion label cache on startup."""
    logger.info("Warming encoder and emotion cache...")
    get_encoder()
    # One dummy classify to populate _label_embeddings so /health reports ready
    classify("你好呀！今天怎么样呢")
    logger.info("Startup warm-up done.")
