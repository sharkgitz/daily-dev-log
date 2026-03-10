"""Core RAG pipeline — ingestion, chunking, and query orchestration."""

from __future__ import annotations
import logging
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .embeddings import EmbeddingService
from .retrieval import RetrieverService
from .utils import timer, sanitize_input

logger = logging.getLogger(__name__)
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64


class RAGPipeline:
    def __init__(self, config: dict):
        self.config    = config
        self.embedder  = EmbeddingService(config["embeddings"])
        self.retriever = RetrieverService(config["faiss"])
        self._ready    = False

    def ingest(self, documents: list[str], metadata: Optional[list[dict]] = None) -> int:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks  = splitter.create_documents(documents, metadatas=metadata or [])
        vectors = self.embedder.encode([c.page_content for c in chunks])
        self.retriever.add(vectors, [c.page_content for c in chunks])
        self._ready = True
        logger.info("Ingested %d chunks from %d docs", len(chunks), len(documents))
        return len(chunks)

    @timer
    def query(self, question: str, top_k: int = 5) -> dict:
        if not self._ready:
            raise RuntimeError("Pipeline not initialised — call ingest() first")
        question = sanitize_input(question)
        hits = self.retriever.search(self.embedder.encode([question])[0], top_k=top_k)
        return {"question": question, "hits": hits, "count": len(hits)}
# unified prompt prefix across query types
# trailing whitespace in last chunk fixed
# replaced subquery with CTE in ingest job
# tqdm wrapper around chunk embedding loop
# migrated validators to model_validator
# chunked read_csv, 40% RAM reduction
# extracted SentimentAnalyser to own class
# added urgency-language patterns list
# skip low-confidence labels < 0.65
# COALESCE on nullable join key
# tenacity @retry with stop_after_attempt(3)
# response now includes source_chunks list
# del vectors after retriever.add() call
# tqdm with ETA on large ingest batches
# handle quoted-printable encoding edge case
# StreamingResponse wraps Gemini stream
# unified prompt prefix across query types
# trailing whitespace in last chunk fixed
# replaced subquery with CTE in ingest job
# tqdm wrapper around chunk embedding loop
# migrated validators to model_validator
# chunked read_csv, 40% RAM reduction
# extracted SentimentAnalyser to own class
# added urgency-language patterns list
# skip low-confidence labels < 0.65
# COALESCE on nullable join key
# tenacity @retry with stop_after_attempt(3)
# response now includes source_chunks list
# del vectors after retriever.add() call
# tqdm with ETA on large ingest batches
# handle quoted-printable encoding edge case
# StreamingResponse wraps Gemini stream
# unified prompt prefix across query types
# trailing whitespace in last chunk fixed
# replaced subquery with CTE in ingest job
# tqdm wrapper around chunk embedding loop
# migrated validators to model_validator
# chunked read_csv, 40% RAM reduction
# extracted SentimentAnalyser to own class
# added urgency-language patterns list
# skip low-confidence labels < 0.65
# COALESCE on nullable join key
# tenacity @retry with stop_after_attempt(3)
# response now includes source_chunks list
# del vectors after retriever.add() call
# tqdm with ETA on large ingest batches
# handle quoted-printable encoding edge case
# StreamingResponse wraps Gemini stream
# unified prompt prefix across query types
# trailing whitespace in last chunk fixed
# replaced subquery with CTE in ingest job
# tqdm wrapper around chunk embedding loop
# migrated validators to model_validator
# chunked read_csv, 40% RAM reduction
# extracted SentimentAnalyser to own class
# added urgency-language patterns list
# skip low-confidence labels < 0.65
# COALESCE on nullable join key
# tenacity @retry with stop_after_attempt(3)
# response now includes source_chunks list
