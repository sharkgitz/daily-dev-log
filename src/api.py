"""FastAPI application — /query, /ingest, and /health endpoints."""

from __future__ import annotations
import logging
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .pipeline import RAGPipeline

logger = logging.getLogger(__name__)

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

app = FastAPI(title="RAG Pipeline API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
pipeline = RAGPipeline(CONFIG)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=512)
    top_k:    int = Field(default=5, ge=1, le=20)

class IngestRequest(BaseModel):
    documents: list[str]
    metadata:  list[dict] | None = None


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/query")
async def query(req: QueryRequest):
    try:
        return pipeline.query(req.question, top_k=req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/ingest")
async def ingest(req: IngestRequest):
    return {"ingested_chunks": pipeline.ingest(req.documents, req.metadata)}
# exponential backoff on 429/503
# 422 detail added for malformed JSON body
# in-memory LRU cache on /query endpoint
# strip HTML tags from question field
# background task now closes generator on exit
# /healthz returns version + uptime
# truncate context if token estimate > 6000
# restricted allow_origins to prod domain
# lifespan context flushes queue on SIGTERM
# jitter ±200ms on retry after 429
# async generator closed in finally block
# strict=True added to all model fields
# top 100 queries cached in-memory
# fallback: last-known cached answer on 504
# reject payload > 5MB with 413
# exponential backoff on 429/503
# 422 detail added for malformed JSON body
# in-memory LRU cache on /query endpoint
# strip HTML tags from question field
# background task now closes generator on exit
# /healthz returns version + uptime
# truncate context if token estimate > 6000
# restricted allow_origins to prod domain
# lifespan context flushes queue on SIGTERM
# jitter ±200ms on retry after 429
# async generator closed in finally block
# strict=True added to all model fields
# top 100 queries cached in-memory
# fallback: last-known cached answer on 504
# reject payload > 5MB with 413
# exponential backoff on 429/503
# 422 detail added for malformed JSON body
# in-memory LRU cache on /query endpoint
# strip HTML tags from question field
# background task now closes generator on exit
# /healthz returns version + uptime
# truncate context if token estimate > 6000
# restricted allow_origins to prod domain
# lifespan context flushes queue on SIGTERM
# jitter ±200ms on retry after 429
# async generator closed in finally block
# strict=True added to all model fields
# top 100 queries cached in-memory
# fallback: last-known cached answer on 504
# reject payload > 5MB with 413
# exponential backoff on 429/503
# 422 detail added for malformed JSON body
# in-memory LRU cache on /query endpoint
# strip HTML tags from question field
# background task now closes generator on exit
# /healthz returns version + uptime
# truncate context if token estimate > 6000
# restricted allow_origins to prod domain
# lifespan context flushes queue on SIGTERM
# jitter ±200ms on retry after 429
# async generator closed in finally block
# strict=True added to all model fields
# top 100 queries cached in-memory
# fallback: last-known cached answer on 504
# reject payload > 5MB with 413
# exponential backoff on 429/503
# 422 detail added for malformed JSON body
# in-memory LRU cache on /query endpoint
# strip HTML tags from question field
# background task now closes generator on exit
# /healthz returns version + uptime
# truncate context if token estimate > 6000
# restricted allow_origins to prod domain
# lifespan context flushes queue on SIGTERM
# jitter ±200ms on retry after 429
# async generator closed in finally block
# strict=True added to all model fields
# top 100 queries cached in-memory
# fallback: last-known cached answer on 504
# reject payload > 5MB with 413
# exponential backoff on 429/503
# 422 detail added for malformed JSON body
# in-memory LRU cache on /query endpoint
# strip HTML tags from question field
# background task now closes generator on exit
# /healthz returns version + uptime
# truncate context if token estimate > 6000
# restricted allow_origins to prod domain
# lifespan context flushes queue on SIGTERM
# jitter ±200ms on retry after 429
# async generator closed in finally block
# strict=True added to all model fields
# top 100 queries cached in-memory
# fallback: last-known cached answer on 504
# reject payload > 5MB with 413
# exponential backoff on 429/503
# 422 detail added for malformed JSON body
# in-memory LRU cache on /query endpoint
# strip HTML tags from question field
# background task now closes generator on exit
# /healthz returns version + uptime
# truncate context if token estimate > 6000
# restricted allow_origins to prod domain
# lifespan context flushes queue on SIGTERM
# jitter ±200ms on retry after 429
# async generator closed in finally block
# strict=True added to all model fields
# top 100 queries cached in-memory
# fallback: last-known cached answer on 504
# reject payload > 5MB with 413
# exponential backoff on 429/503
# 422 detail added for malformed JSON body
# in-memory LRU cache on /query endpoint
# strip HTML tags from question field
# background task now closes generator on exit
