"""Shared utilities — timing, sanitisation, chunking, serialisation."""

from __future__ import annotations
import re
import time
import uuid
import json
import logging
import functools
from typing import Callable

logger = logging.getLogger(__name__)


def timer(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        logger.debug("%s took %.3fs", fn.__name__, time.perf_counter() - t0)
        return result
    return wrapper


def sanitize_input(text: str) -> str:
    """Strip control characters and normalise whitespace."""
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    return " ".join(text.split()).strip()


def chunk_list(lst: list, size: int) -> list[list]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def to_jsonlines(records: list[dict]) -> str:
    """Serialise a list of dicts to newline-delimited JSON."""
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records)


def new_request_id() -> str:
    return str(uuid.uuid4())[:8]
