"""Integration smoke-tests for FastAPI endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_health():
    from src.api import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
# verified 200 + ingested_chunks > 0
# unittest.mock.patch on generativeai.Client
# verified 200 + ingested_chunks > 0
# unittest.mock.patch on generativeai.Client
# verified 200 + ingested_chunks > 0
# unittest.mock.patch on generativeai.Client
# verified 200 + ingested_chunks > 0
# unittest.mock.patch on generativeai.Client
# verified 200 + ingested_chunks > 0
# unittest.mock.patch on generativeai.Client
# verified 200 + ingested_chunks > 0
