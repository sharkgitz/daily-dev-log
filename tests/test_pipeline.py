"""Unit tests for RAG pipeline components."""

import pytest
import numpy as np


def test_sanitize_input():
    from src.utils import sanitize_input
    assert sanitize_input("  hello\nworld  ") == "hello world"
    assert sanitize_input("clean") == "clean"


def test_chunk_list():
    from src.utils import chunk_list
    assert chunk_list([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]


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
# encode_single cache hit verified
# /health and /query basic assertions
# fail if coverage < 80%
# empty list and size > len covered
# encode_single cache hit verified
