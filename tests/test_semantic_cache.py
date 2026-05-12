import numpy as np
import pytest

from semantic_cache import SemanticCache


@pytest.mark.asyncio
async def test_exact_cache_hit_returns_cached_chunks():
    cache = SemanticCache(dimension=3)
    embedding = np.array([1.0, 0.0, 0.0])
    chunks = [{"text": "alpha"}]

    await cache.set("What is Alpha?", embedding, chunks)

    result = await cache.get("what is alpha?", embedding)

    assert result == chunks


@pytest.mark.asyncio
async def test_semantic_cache_hit_returns_similar_query():
    cache = SemanticCache(dimension=3)
    cached_chunks = [{"text": "budget"}]

    await cache.set("budget", np.array([1.0, 0.0, 0.0]), cached_chunks)

    result = await cache.get("cost", np.array([0.99, 0.01, 0.0]))

    assert result == cached_chunks


@pytest.mark.asyncio
async def test_cache_miss_returns_none():
    cache = SemanticCache(dimension=3)

    result = await cache.get("unknown", np.array([0.0, 1.0, 0.0]))

    assert result is None
