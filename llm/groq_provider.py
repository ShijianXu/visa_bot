"""Groq LLM provider."""

import time
from typing import Iterator

from groq import Groq, APIError, RateLimitError

import config
from .base import LLMProvider

_MAX_RETRIES = 3


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
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except RateLimitError as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
            except APIError as exc:
                # Retry on 5xx server errors only
                if getattr(exc, "status_code", 0) >= 500:
                    last_exc = exc
                    time.sleep(1)
                else:
                    raise
        raise last_exc  # type: ignore[misc]

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
