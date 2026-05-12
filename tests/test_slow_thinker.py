from types import SimpleNamespace

import pytest

from slow_thinker import SlowThinker


class FakeChatContext:
    def __init__(self, messages):
        self._messages = messages

    def messages(self):
        return self._messages


class FakeEmbeddings:
    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeCache:
    def __init__(self, hit=False):
        self.hit = hit
        self.set_calls = []

    async def get(self, query, query_embedding):
        return [{"text": "cached"}] if self.hit else None

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
                    payload={"text": "Budget info", "project_name": "Alpha", "page": 12}
                )
            ]
        )


@pytest.mark.asyncio
async def test_predict_next_queries_uses_chat_context_and_current_query():
    thinker = SlowThinker(FakeEmbeddings(), FakeQdrant(), FakeCache())
    thinker.chat_ctx = FakeChatContext(
        [SimpleNamespace(role="user", text_content="What is the schedule for alpha?")]
    )

    predicted = await thinker.predict_next_queries("How long will it take?")

    assert "What is the budget for this project?" in predicted


@pytest.mark.asyncio
async def test_prefetch_populates_cache_on_miss():
    cache = FakeCache(hit=False)
    qdrant = FakeQdrant()
    thinker = SlowThinker(FakeEmbeddings(), qdrant, cache)
    thinker.chat_ctx = FakeChatContext(
        [SimpleNamespace(role="user", text_content="What is the project schedule?")]
    )

    await thinker.pre_fetch("When does it finish?")

    assert qdrant.calls
    assert cache.set_calls
