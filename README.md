# rag-pipeline

A retrieval-augmented generation (RAG) system I built to understand how semantic search and document Q&A work end to end — from chunking and embedding to FAISS indexing and LLM answer synthesis — rather than just calling a black-box library.

## What it does

Takes a set of documents, embeds them with `sentence-transformers`, indexes them in FAISS, then answers natural language questions by retrieving the most relevant chunks and synthesising a response through the Gemini API.

```
[Documents] → Chunker → Embedder → FAISS Index
                                        |
[Query] → Embedder ──────────────> Top-k Retrieval -> Gemini -> Answer
```

## Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | async, auto-docs, Pydantic v2 validation |
| Embeddings | all-MiniLM-L6-v2 | fast, solid quality at 384-dim |
| Vector store | FAISS (IndexFlatL2) | simple, no infra needed for this scale |
| LLM | Gemini 1.5 Flash | fast + cheap, 1M token context window |
| Data | BigQuery + Pandas | loading large document corpora |

## Setup

```bash
pip install -r requirements.txt
# add your GOOGLE_API_KEY to the environment
uvicorn src.api:app --reload
```

Config lives in `config.yaml` — adjust chunk size, score threshold, and model there.

## API

```
POST /ingest   { "documents": ["..."], "metadata": [...] }
POST /query    { "question": "...", "top_k": 5 }
GET  /health
```

## Notes

- `IndexFlatL2` is fine up to ~50k chunks; beyond that switch to `IndexIVFFlat` with nprobe tuning
- `score_threshold: 0.72` is conservative — lower it if you're getting empty result sets on valid queries
- Response includes `source_chunks` so you can verify the retrieval before trusting the answer
- numpy is pinned to `<2.0` in requirements because faiss-cpu doesn't build cleanly against numpy 2.x yet

## Tests

```bash
pytest tests/ -v --tb=short
```
