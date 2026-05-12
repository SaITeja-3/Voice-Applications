import os

import httpx
import openai
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
)
from livekit.agents.tts.tts import ChunkedStream
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS


class GroqTTS(tts.TTS):
    def __init__(
        self,
        *,
        model: str = "canopylabs/orpheus-v1-english",
        voice: str = "hannah",
        base_url: str = "https://api.groq.com/openai/v1",
        api_key: str | None = None,
        response_format: str = "wav",
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self._model = model
        self._voice = voice
        self._response_format = response_format
        self._client = openai.AsyncClient(
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            base_url=base_url,
            max_retries=0,
            timeout=httpx.Timeout(connect=15.0, read=30.0, write=30.0, pool=30.0),
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "groq"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return _GroqChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        await self._client.close()


class _GroqChunkedStream(ChunkedStream):
    def __init__(self, *, tts: GroqTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            response = await self._tts._client.audio.speech.create(
                model=self._tts._model,
                voice=self._tts._voice,
                input=self.input_text,
                response_format=self._tts._response_format,
                timeout=httpx.Timeout(60, connect=self._conn_options.timeout),
            )
            audio_bytes = await response.aread()
            output_emitter.initialize(
                request_id="groq-tts",
                sample_rate=self._tts.sample_rate,
                num_channels=self._tts.num_channels,
                mime_type=f"audio/{self._tts._response_format}",
            )
            output_emitter.push(audio_bytes)
            output_emitter.flush()
        except openai.APITimeoutError:
            raise APITimeoutError() from None
        except openai.APIStatusError as exc:
            raise APIStatusError(
                exc.message,
                status_code=exc.status_code,
                request_id=exc.request_id,
                body=exc.body,
            ) from None
        except Exception as exc:
            raise APIConnectionError() from exc
