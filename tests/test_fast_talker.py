from types import SimpleNamespace

import pytest

from fast_talker import FastTalker


class FakeEmbeddings:
    def embed_query(self, query: str) -> list[float]:
        if "alpha" in query.lower():
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]


class FakeCache:
    def __init__(self, cached=None):
        self.cached = cached
        self.set_calls = []

    async def get(self, query, query_embedding):
        return self.cached

    async def set(self, query, query_embedding, chunks):
        self.set_calls.append((query, chunks))


class FakeQdrant:
    def __init__(self):
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    payload={"text": "Alpha details", "project_name": "Alpha", "page": 49},
                    score=0.98,
                )
            ]
        )


@pytest.mark.asyncio
async def test_retrieve_uses_cache_when_available():
    cache = FakeCache(cached=[{"text": "cached", "project": "Alpha"}])
    talker = FastTalker(FakeEmbeddings(), FakeQdrant(), cache)

    chunks, _ = await talker.retrieve("Tell me about alpha")

    assert chunks == [{"text": "cached", "project": "Alpha"}]
    assert talker.get_stats()["cache_hits"] == 1


@pytest.mark.asyncio
async def test_retrieve_queries_qdrant_on_cache_miss():
    qdrant = FakeQdrant()
    cache = FakeCache(cached=None)
    talker = FastTalker(FakeEmbeddings(), qdrant, cache)

    chunks, _ = await talker.retrieve("Tell me about alpha", project_filter="Alpha")

    assert chunks[0]["project"] == "Alpha"
    assert chunks[0]["page"] == 49
    assert qdrant.calls[0]["query_filter"] is not None
    assert cache.set_calls
    assert talker.get_stats()["cache_misses"] == 1
