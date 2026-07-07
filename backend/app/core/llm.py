"""LLM client abstraction.

Single interface over the NVIDIA NIM endpoint (OpenAI-compatible).
Everything downstream (RAG answerer, personas, debate mode) talks to
`LLMClient` only — never to a provider SDK directly — so swapping or
adding providers later is a one-file change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import AsyncOpenAI

from app.core.config import get_settings


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict = field(default_factory=dict)


class LLMClient:
    """Async chat-completion client backed by NVIDIA NIM (Nemotron)."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.nvidia_api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Add it to backend/.env "
                "(see .env.example)."
            )
        self._client = AsyncOpenAI(
            base_url=settings.nvidia_base_url,
            api_key=settings.nvidia_api_key,
        )
        self._model = settings.llm_model

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """One-shot chat completion. Low temperature by default —
        RAG answers should stick to the retrieved sources."""
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = resp.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            model=resp.model,
            usage=resp.usage.model_dump() if resp.usage else {},
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ):
        """Streaming variant — yields text deltas (for the chat UI later)."""
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Lazy singleton — created on first use so the app can boot without a key."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
