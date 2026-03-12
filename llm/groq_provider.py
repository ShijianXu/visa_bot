"""Groq LLM provider."""

from typing import Iterator

from groq import Groq

import config
from .base import LLMProvider


class GroqProvider(LLMProvider):
    """Groq Cloud back-end (llama-3.x, mixtral, …)."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._client = Groq(api_key=api_key or config.GROQ_API_KEY)
        self._model = model or config.GROQ_MODEL

    # ── LLMProvider interface ────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "groq"
