"""Integration smoke-tests for FastAPI endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_health():
    from src.api import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_healthz():
    from src.api import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/healthz")
    assert r.status_code == 200
    assert "ready" in r.json()


@pytest.mark.asyncio
async def test_query_before_ingest_returns_503():
    from src.api import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post("/query", json={"question": "what is this?", "top_k": 3})
    assert r.status_code == 503
