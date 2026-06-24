"""FastAPI application — /query, /ingest, /health endpoints."""

from __future__ import annotations
import logging
import uuid
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from .pipeline import RAGPipeline

logger = logging.getLogger(__name__)

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG pipeline starting up")
    yield
    logger.info("RAG pipeline shutting down")


app = FastAPI(title="RAG Pipeline API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
pipeline = RAGPipeline(CONFIG)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=512)
    top_k:    int = Field(default=5, ge=1, le=20)


class IngestRequest(BaseModel):
    documents: list[str]
    metadata:  list[dict] | None = None


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "ready": pipeline._ready}


@app.post("/query")
async def query(req: QueryRequest):
    try:
        return pipeline.query(req.question, top_k=req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in /query")
        raise HTTPException(status_code=500, detail="Internal error")


@app.post("/ingest")
async def ingest(req: IngestRequest):
    n = pipeline.ingest(req.documents, req.metadata)
    return {"ingested_chunks": n, "total_indexed": pipeline.retriever.index.ntotal if pipeline.retriever.index else 0}
