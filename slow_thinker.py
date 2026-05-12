import asyncio
import os

import numpy as np
from livekit.agents.llm import ChatContext


class SlowThinker:
    def __init__(self, embeddings, qdrant_client, semantic_cache) -> None:
        self.embeddings = embeddings
        self.qdrant = qdrant_client
        self.cache = semantic_cache
        self.chat_ctx = ChatContext()
        self.collection_name = os.getenv("QDRANT_COLLECTION", "projects_rag")
        self.result_limit = int(os.getenv("RAG_RETRIEVAL_LIMIT", "4"))

        self.follow_up_patterns = {
            "timeline": ["schedule", "deadline", "when does", "how long"],
            "budget": ["cost", "price", "funding", "expenses"],
            "team": ["who", "stakeholders", "members", "leads"],
            "technical": ["technology", "stack", "tools", "architecture"],
        }

    async def predict_next_queries(self, current_query: str) -> list[str]:
        predicted: list[str] = []

        recent_messages = self.chat_ctx.messages()[-3:]
        for message in recent_messages:
            if message.role == "user":
                content = message.text_content or ""
                for category, keywords in self.follow_up_patterns.items():
                    if any(keyword in content.lower() for keyword in keywords):
                        if category == "timeline":
                            predicted.append("What is the budget for this project?")
                        elif category == "budget":
                            predicted.append("Who are the team members?")
                        elif category == "team":
                            predicted.append("What is the project timeline?")
                        elif category == "technical":
                            predicted.append("What is the project status?")

        for category, keywords in self.follow_up_patterns.items():
            if any(keyword in current_query.lower() for keyword in keywords):
                if category == "timeline":
                    predicted.append("What is the budget for this project?")
                elif category == "budget":
                    predicted.append("Who are the team members?")

        return list(dict.fromkeys(predicted))[:3]

    async def pre_fetch(self, partial_transcript: str) -> None:
        predicted_queries = await self.predict_next_queries(partial_transcript)
        for query in predicted_queries:
            query_embedding = await asyncio.to_thread(self.embeddings.embed_query, query)
            query_np = np.array(query_embedding)

            if await self.cache.get(query, query_np):
                continue

            response = await asyncio.to_thread(
                self.qdrant.query_points,
                collection_name=self.collection_name,
                query=query_embedding,
                limit=self.result_limit,
                with_payload=True,
            )
            results = response.points
            chunks = [
                {
                    "text": result.payload["text"],
                    "project": result.payload["project_name"],
                    "page": result.payload.get("page"),
                }
                for result in results
            ]
            await self.cache.set(query, query_np, chunks)
            print(f"Pre-fetched '{query}' ({len(chunks)} chunks)")
