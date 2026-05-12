import asyncio
import os
import time
from typing import Optional

import numpy as np
from livekit.agents.telemetry import tracer
from qdrant_client.models import FieldCondition, Filter, MatchValue


class FastTalker:
    def __init__(self, embeddings, qdrant_client, semantic_cache) -> None:
        self.embeddings = embeddings
        self.qdrant = qdrant_client
        self.cache = semantic_cache
        self.cache_hits = 0
        self.cache_misses = 0
        self.collection_name = os.getenv("QDRANT_COLLECTION", "projects_rag")
        self.result_limit = int(os.getenv("RAG_RETRIEVAL_LIMIT", "4"))

    async def retrieve(
        self,
        query: str,
        project_filter: Optional[str] = None,
    ) -> tuple[list[dict], float]:
        with tracer.start_as_current_span("rag_retrieval") as span:
            span.set_attribute("query", query)
            span.set_attribute("project_filter", project_filter or "none")

            start = time.perf_counter()
            query_embedding = await asyncio.to_thread(self.embeddings.embed_query, query)
            chunks = await self.cache.get(query, np.array(query_embedding))

            if chunks:
                self.cache_hits += 1
                total_ms = (time.perf_counter() - start) * 1000
                span.set_attribute("cache_hit", True)
                span.set_attribute("retrieval_time_ms", total_ms)
                print(f"Cache hit: {total_ms:.1f}ms")
                return chunks, total_ms

            self.cache_misses += 1
            span.set_attribute("cache_hit", False)

            filter_param = (
                Filter(
                    must=[
                        FieldCondition(
                            key="project_name",
                            match=MatchValue(value=project_filter),
                        )
                    ]
                )
                if project_filter
                else None
            )

            response = await asyncio.to_thread(
                self.qdrant.query_points,
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=filter_param,
                limit=self.result_limit,
                with_payload=True,
            )
            results = response.points
            chunks = [
                {
                    "text": result.payload["text"],
                    "project": result.payload["project_name"],
                    "score": result.score,
                    "page": result.payload.get("page"),
                }
                for result in results
            ]
            await self.cache.set(query, np.array(query_embedding), chunks)

            total_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("retrieval_time_ms", total_ms)
            print(f"Qdrant search: {total_ms:.1f}ms")
            return chunks, total_ms

    def get_stats(self) -> dict[str, float]:
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 1),
            "total_queries": total,
        }
