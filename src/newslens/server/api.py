from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from newslens.core.config import settings
from newslens.core.engine import InferenceEngine
from newslens.core.model import NewsLens
from newslens.server.schemas import (
    BatchSummarizeRequest,
    BatchSummarizeResponse,
    SummarizeRequest,
    SummarizeResponse,
)

# Global holder for the model to ensure it's a singleton
lens_storage = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting A2A Server. Loading model: {settings.base_model_name}")
    engine = InferenceEngine(adapter_path=settings.adapter_path)
    lens_storage["model"] = NewsLens(engine=engine, max_workers=settings.max_workers)
    yield
    print("Shutting down server...")
    lens_storage.clear()


app = FastAPI(
    title="NewsLens A2A Server",
    description="REST API for Agent-to-Agent news restructuring and summarization.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_lens():
    """Dependency to inject the NewsLens instance."""
    instance = lens_storage.get("model")
    if not instance:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return instance


@app.get("/health")
async def health_check():
    return {"status": "ready", "model": settings.base_model_name}


# @app.post("/summarize", response_model=SummarizeResponse)
# async def summarize(request: SummarizeRequest, lens: NewsLens = Depends(get_lens)):
#     start_time = time.time()

#     try:
#         # We use the ASYNC version from our core/model.py
#         # to prevent blocking the FastAPI event loop
#         summary = await lens.asummarize(
#             text=request.text,
#             temperature=request.temperature,
#             max_tokens=request.max_tokens,
#             top_p=request.top_p,
#         )

#         return SummarizeResponse(
#             summary=summary,
#             tokens_used=len(summary.split()),  # Simple heuristic
#             processing_time=round(time.time() - start_time, 3),
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest, lens: NewsLens = Depends(get_lens)):
    summary = await lens.asummarize(
        request.article, max_tokens=request.max_tokens, temperature=request.temperature
    )
    return SummarizeResponse(
        summary=summary, input_length=len(request.article), output_length=len(summary)
    )


@app.post("/summarize/batch", response_model=BatchSummarizeResponse)
async def batch_summarize(
    request: BatchSummarizeRequest, lens: NewsLens = Depends(get_lens)
):
    summaries = await lens.asummarize_batch(
        request.articles, max_tokens=request.max_tokens, temperature=request.temperature
    )
    return BatchSummarizeResponse(summaries=summaries, total_articles=len(summaries))
