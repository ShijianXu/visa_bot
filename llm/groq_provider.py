"""Groq LLM provider."""

import time
from typing import Iterator

from groq import Groq, APIError, RateLimitError

import config
from .base import LLMProvider

_MAX_RETRIES = 5
_MAX_BACKOFF = 65  # Groq resets per-minute limits after 60 s


def _rate_limit_wait(exc: RateLimitError, attempt: int) -> None:
    """Sleep the right amount on a 429: use Retry-After header when present,
    otherwise exponential backoff capped at _MAX_BACKOFF seconds."""
    wait = 0.0
    try:
        # Groq sets 'retry-after' (seconds) or 'x-ratelimit-reset-requests'
        headers = exc.response.headers  # type: ignore[union-attr]
        for key in ("retry-after", "x-ratelimit-reset-requests"):
            val = headers.get(key, "")
            if val:
                wait = float(val) + 1.0
                break
    except Exception:
        pass
    if not wait:
        wait = min(2 ** attempt * 3, _MAX_BACKOFF)
    time.sleep(wait)


class GroqProvider(LLMProvider):
    """Groq Cloud back-end (llama-3.x, mixtral, …)."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._client = Groq(api_key=api_key or config.GROQ_API_KEY)
        self._model = model or config.GROQ_MODEL
        self._fallback_model = config.GROQ_FALLBACK_MODEL

    # ── LLMProvider interface ────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        _model_override: str | None = None,
    ) -> str:
        chain = [_model_override] if _model_override else self._model_chain()
        for model in chain:
            last_exc: Exception | None = None
            for attempt in range(_MAX_RETRIES):
                try:
                    response = self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content
                except RateLimitError as exc:
                    last_exc = exc
                    _rate_limit_wait(exc, attempt)
                except APIError as exc:
                    if getattr(exc, "status_code", 0) >= 500:
                        last_exc = exc
                        time.sleep(1)
                    else:
                        raise
            # All retries on this model exhausted — try fallback if rate limited
            if isinstance(last_exc, RateLimitError) and model != self._fallback_model:
                continue
            raise last_exc  # type: ignore[misc]
        raise RuntimeError("All models exhausted")

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
    ) -> Iterator[str]:
        for model in self._model_chain():
            last_exc: Exception | None = None
            for attempt in range(_MAX_RETRIES):
                try:
                    stream = self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        stream=True,
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield delta
                    return
                except RateLimitError as exc:
                    last_exc = exc
                    _rate_limit_wait(exc, attempt)
            if isinstance(last_exc, RateLimitError) and model != self._fallback_model:
                continue
            if last_exc:
                raise last_exc

    def _model_chain(self) -> list[str]:
        """Primary model first, fallback second (deduplicated)."""
        seen = []
        for m in [self._model, self._fallback_model]:
            if m and m not in seen:
                seen.append(m)
        return seen

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "groq"
