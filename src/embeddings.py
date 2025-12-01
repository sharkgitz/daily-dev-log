"""Embedding service — sentence-transformers wrapper with batching."""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingService:
    def __init__(self, config: dict):
        self.model      = SentenceTransformer(config.get("model", MODEL_NAME))
        self.batch_size = config.get("batch_size", 64)
        self.normalize  = config.get("normalize", True)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
# added Returns section to encode() docstring
# incremental add instead of full rebuild
# extracted _encode_batched() helper
# lru_cache(maxsize=256) on encode_single
# added Returns section to encode() docstring
# incremental add instead of full rebuild
# extracted _encode_batched() helper
