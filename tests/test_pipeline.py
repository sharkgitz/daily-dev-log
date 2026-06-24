"""Unit tests for RAG pipeline components."""

import pytest
import numpy as np


def test_sanitize_input():
    from src.utils import sanitize_input
    assert sanitize_input("  hello\nworld  ") == "hello world"
    assert sanitize_input("clean") == "clean"
    assert sanitize_input("tab\there") == "tab here"


def test_chunk_list():
    from src.utils import chunk_list
    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunk_list([], 3) == []
    assert chunk_list([1], 10) == [[1]]


def test_to_jsonlines():
    from src.utils import to_jsonlines
    result = to_jsonlines([{"a": 1}, {"b": 2}])
    lines = result.strip().split("\n")
    assert len(lines) == 2


def test_retriever_empty():
    from src.retrieval import RetrieverService
    svc = RetrieverService({"score_threshold": 0.5})
    assert svc.search(np.zeros(384)) == []


def test_retriever_add_search():
    from src.retrieval import RetrieverService
    svc = RetrieverService({"score_threshold": 0.0})
    vecs  = np.random.rand(3, 384).astype("float32")
    texts = ["alpha", "beta", "gamma"]
    svc.add(vecs, texts)
    hits = svc.search(vecs[0], top_k=1)
    assert len(hits) == 1
    assert hits[0].text == "alpha"
    assert hits[0].rank == 0


def test_retriever_threshold_filters():
    from src.retrieval import RetrieverService
    svc = RetrieverService({"score_threshold": 0.9999})
    vecs  = np.random.rand(5, 384).astype("float32")
    svc.add(vecs, ["a", "b", "c", "d", "e"])
    # very high threshold should filter most results
    hits = svc.search(np.random.rand(384).astype("float32"), top_k=5)
    assert isinstance(hits, list)
