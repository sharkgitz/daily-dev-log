"""Shared utilities — timing, caching, sanitisation."""

from __future__ import annotations
import re, time, logging, functools
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
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    return " ".join(text.split()).strip()


def chunk_list(lst: list, size: int) -> list[list]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]
# removed dead import, fixed f-string
# uuid4 request_id injected via middleware
# user-facing errors now include resolution hint
# serialises dataset to JSON-lines format
# contextlib.asynccontextmanager wrapper
# removed dead import, fixed f-string
# uuid4 request_id injected via middleware
