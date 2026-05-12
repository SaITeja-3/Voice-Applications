import asyncio
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, function_tool
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice.events import CloseEvent, UserInputTranscribedEvent
from livekit.plugins import deepgram, openai, silero
from qdrant_client import QdrantClient

from fast_talker import FastTalker
from groq_tts import GroqTTS
from semantic_cache import SemanticCache
from slow_thinker import SlowThinker

load_dotenv()


class RAGVoiceAgent(Agent):
    def __init__(self, fast_talker: FastTalker, slow_thinker: SlowThinker) -> None:
        super().__init__(
            instructions=(
                "You are a helpful assistant answering questions about the indexed document and its contents. "
                "For any question about the textbook, examples, chapters, pages, definitions, or explanations, "
                "use the retrieve_context tool before answering. "
                "Keep replies concise for voice, ideally under 180 characters and no more than 2 short sentences. "
                "Be conversational and natural for voice. "
                "If the context does not contain the answer, say so briefly."
            ),
        )
        self.fast_talker = fast_talker
        self.slow_thinker = slow_thinker

    @function_tool
    async def retrieve_context(self, query: str) -> str:
        """Retrieve relevant document context for the query."""
        chunks, retrieval_ms = await self.fast_talker.retrieve(
            query,
            project_filter=self._extract_project(query),
        )
        print(f"Retrieval: {retrieval_ms:.1f}ms | Chunks: {len(chunks)}")
        if not chunks:
            return "No relevant information found."
        lines = []
        for chunk in chunks:
            page_suffix = f" p.{chunk['page']}" if chunk.get("page") else ""
            lines.append(f"[{chunk['project']}{page_suffix}]: {chunk['text'][:220]}")
        return "\n".join(lines)

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        _ = new_message
        self.slow_thinker.chat_ctx = turn_ctx.copy()

    def _extract_project(self, query: str) -> Optional[str]:
        for keyword in ["alpha", "beta", "gamma"]:
            if keyword in query.lower():
                return f"project_{keyword}"
        return None


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        model_kwargs={
            "device": os.getenv("EMBED_DEVICE", "cpu"),
            "trust_remote_code": False,
        },
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=os.getenv("HF_HOME"),
    )
    cache = SemanticCache(dimension=384)
    fast_talker = FastTalker(embeddings, qdrant_client, cache)
    slow_thinker = SlowThinker(embeddings, qdrant_client, cache)

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(
            model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.7,
        ),
        tts=GroqTTS(
            model=os.getenv("TTS_MODEL", "canopylabs/orpheus-v1-english"),
            voice=os.getenv("TTS_VOICE", "hannah"),
            base_url=os.getenv("TTS_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=os.getenv("GROQ_API_KEY"),
            response_format=os.getenv("TTS_RESPONSE_FORMAT", "wav"),
        ),
    )

    agent = RAGVoiceAgent(fast_talker, slow_thinker)
    session_closed = asyncio.get_running_loop().create_future()

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: UserInputTranscribedEvent) -> None:
        if not event.is_final:
            return
        slow_thinker.chat_ctx = session.history.copy()
        asyncio.create_task(slow_thinker.pre_fetch(event.transcript))

    @session.on("close")
    def on_session_close(event: CloseEvent) -> None:
        if not session_closed.done():
            if event.error:
                session_closed.set_exception(RuntimeError(str(event.error)))
            else:
                session_closed.set_result(None)

    await session.start(room=ctx.room, agent=agent)
    await session_closed

    print("Session Stats:")
    print(fast_talker.get_stats())
    print(cache.get_stats())


if __name__ == "__main__":
    cli.run_app(server)
