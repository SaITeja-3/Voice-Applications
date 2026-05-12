import asyncio
import hashlib
from collections import OrderedDict
from typing import Optional

import faiss
import numpy as np


class SemanticCache:
    def __init__(self, dimension: int = 384) -> None:
        self.exact_cache: OrderedDict[str, list[dict]] = OrderedDict()
        self.max_exact_size = 100
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.cached_chunks: list[list[dict]] = []
        self.semantic_threshold = 0.92

    def _hash_query(self, query: str) -> str:
        normalized = query.lower().strip().encode()
        return hashlib.md5(normalized).hexdigest()

    async def get(self, query: str, query_embedding: np.ndarray) -> Optional[list[dict]]:
        query_hash = self._hash_query(query)
        if query_hash in self.exact_cache:
            self.exact_cache.move_to_end(query_hash)
            print("EXACT CACHE HIT")
            return self.exact_cache[query_hash]

        if self.faiss_index.ntotal > 0:
            query_vec = query_embedding.reshape(1, -1).astype("float32")
            faiss.normalize_L2(query_vec)
            distances, indexes = await asyncio.to_thread(self.faiss_index.search, query_vec, 1)
            if distances[0][0] >= self.semantic_threshold:
                print(f"SEMANTIC CACHE HIT ({distances[0][0]:.3f})")
                return self.cached_chunks[indexes[0][0]]

        print("Cache miss")
        return None

    async def set(self, query: str, query_embedding: np.ndarray, chunks: list[dict]) -> None:
        query_hash = self._hash_query(query)
        self.exact_cache[query_hash] = chunks
        if len(self.exact_cache) > self.max_exact_size:
            self.exact_cache.popitem(last=False)

        query_vec = query_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_vec)
        await asyncio.to_thread(self.faiss_index.add, query_vec)
        self.cached_chunks.append(chunks)

    def get_stats(self) -> dict[str, int]:
        return {
            "exact_cache_size": len(self.exact_cache),
            "semantic_cache_size": self.faiss_index.ntotal,
        }
