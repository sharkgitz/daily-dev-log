"""Embedding service — sentence-transformers wrapper with batching and caching."""

from __future__ import annotations
import numpy as np
import functools
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingService:
    def __init__(self, config: dict):
        self.model      = SentenceTransformer(config.get("model", MODEL_NAME))
        self.batch_size = config.get("batch_size", 64)
        self.normalize  = config.get("normalize", True)
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        # check cache first for single-text lookups (common in query path)
        if len(texts) == 1 and texts[0] in self._cache:
            return self._cache[texts[0]].reshape(1, -1)

        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 200,
        )
        if len(texts) == 1:
            self._cache[texts[0]] = vecs[0]
        return vecs

    def encode_single(self, text: str) -> np.ndarray:
        """Convenience wrapper for single-string encode with cache hit."""
        return self.encode([text])[0]
