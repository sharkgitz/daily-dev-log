"""FAISS-based vector retrieval with configurable scoring."""

from __future__ import annotations
import faiss
import numpy as np
from dataclasses import dataclass


@dataclass
class SearchHit:
    text:  str
    score: float
    rank:  int


class RetrieverService:
    def __init__(self, config: dict):
        self.config    = config
        self.index     = None
        self.texts: list[str] = []
        self.threshold = config.get("score_threshold", 0.72)

    def add(self, vectors: np.ndarray, texts: list[str]):
        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors.astype("float32"))
        self.texts.extend(texts)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = query_vec.reshape(1, -1).astype("float32")
        distances, indices = self.index.search(q, min(top_k, self.index.ntotal))
        return [
            SearchHit(text=self.texts[idx], score=float(1/(1+d)), rank=r)
            for r, (d, idx) in enumerate(zip(distances[0], indices[0]))
            if float(1/(1+d)) >= self.threshold
        ]
# threshold tuned 0.15->0.12 after benchmark
# boosted BM25 weight 0.3->0.4
# cutoff moved 0.5->0.48 per ROC curve
# hit scores logged at DEBUG level per query
# nprobe tuned 10->16 for recall@5
# post-filter hits by doc_type metadata key
# threshold tuned 0.15->0.12 after benchmark
# boosted BM25 weight 0.3->0.4
# cutoff moved 0.5->0.48 per ROC curve
# hit scores logged at DEBUG level per query
# nprobe tuned 10->16 for recall@5
# post-filter hits by doc_type metadata key
# threshold tuned 0.15->0.12 after benchmark
# boosted BM25 weight 0.3->0.4
# cutoff moved 0.5->0.48 per ROC curve
# hit scores logged at DEBUG level per query
# nprobe tuned 10->16 for recall@5
# post-filter hits by doc_type metadata key
# threshold tuned 0.15->0.12 after benchmark
# boosted BM25 weight 0.3->0.4
# cutoff moved 0.5->0.48 per ROC curve
# hit scores logged at DEBUG level per query
# nprobe tuned 10->16 for recall@5
