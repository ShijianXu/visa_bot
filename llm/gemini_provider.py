"""Google Gemini LLM provider (google-genai SDK).

Free tier (Google AI Studio):
  gemini-2.0-flash      — 15 RPM, 1 000 000 tokens/day
  gemini-2.0-flash-lite — 30 RPM, 1 500 000 tokens/day (used for preprocessing)

Get a free API key at https://aistudio.google.com/apikey
"""

import time
from typing import Iterator

from google import genai
from google.genai import types
from google.genai.errors import ClientError

import config
from .base import LLMProvider

_MAX_RETRIES = 5
_MAX_BACKOFF = 65  # Gemini resets per-minute limits after ~60 s


def _is_rate_limit(exc: ClientError) -> bool:
    return getattr(exc, "status_code", 0) == 429 or "429" in str(exc)


def _rate_limit_wait(exc: ClientError, attempt: int) -> None:
    """Sleep based on retry metadata or exponential backoff."""
    wait = 0.0
    try:
        # The new SDK surfaces retry delay via exception details
        details = getattr(exc, "details", None) or {}
        retry_delay = details.get("retryDelay", "")
        if retry_delay:
            # format is e.g. "30s"
            wait = float(retry_delay.rstrip("s")) + 1.0
    except Exception:
        pass
    if not wait:
        wait = min(2 ** attempt * 3, _MAX_BACKOFF)
    time.sleep(wait)


def _to_gemini_contents(
    messages: list[dict],
) -> tuple[str, list[types.Content]]:
    """Convert OpenAI-style messages to Gemini format.

    Returns (system_instruction, contents_list).
    Gemini uses role "model" instead of "assistant" and takes the system
    prompt separately.
    """
    system_parts: list[str] = []
    contents: list[types.Content] = []
    for msg in messages:
        role = msg["role"]
        text = msg.get("content", "")
        if role == "system":
            system_parts.append(text)
        elif role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=text)]))
    return "\n\n".join(system_parts), contents


class GeminiProvider(LLMProvider):
    """Google Gemini back-end (gemini-2.0-flash, gemini-2.0-flash-lite, …)."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._client = genai.Client(api_key=api_key or config.GEMINI_API_KEY)
        self._model = model or config.GEMINI_MODEL
        self._fallback_model = config.GEMINI_FALLBACK_MODEL

    # ── LLMProvider interface ────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        _model_override: str | None = None,
    ) -> str:
        chain = [_model_override] if _model_override else self._model_chain()
        system_instruction, contents = _to_gemini_contents(messages)
        cfg = types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        for model_name in chain:
            last_exc: Exception | None = None
            for attempt in range(_MAX_RETRIES):
                try:
                    response = self._client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=cfg,
                    )
                    return response.text
                except ClientError as exc:
                    if _is_rate_limit(exc):
                        last_exc = exc
                        _rate_limit_wait(exc, attempt)
                    else:
                        raise
                except Exception:
                    raise
            # Retries exhausted — try fallback if rate-limited
            if isinstance(last_exc, ClientError) and _is_rate_limit(last_exc) and model_name != self._fallback_model:
                continue
            raise last_exc  # type: ignore[misc]
        raise RuntimeError("All Gemini models exhausted")

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
    ) -> Iterator[str]:
        system_instruction, contents = _to_gemini_contents(messages)
        cfg = types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            temperature=temperature,
        )

        for model_name in self._model_chain():
            last_exc: Exception | None = None
            for attempt in range(_MAX_RETRIES):
                try:
                    for chunk in self._client.models.generate_content_stream(
                        model=model_name,
                        contents=contents,
                        config=cfg,
                    ):
                        if chunk.text:
                            yield chunk.text
                    return
                except ClientError as exc:
                    if _is_rate_limit(exc):
                        last_exc = exc
                        _rate_limit_wait(exc, attempt)
                    else:
                        raise
            if isinstance(last_exc, ClientError) and _is_rate_limit(last_exc) and model_name != self._fallback_model:
                continue
            if last_exc:
                raise last_exc

    def _model_chain(self) -> list[str]:
        seen: list[str] = []
        for m in [self._model, self._fallback_model]:
            if m and m not in seen:
                seen.append(m)
        return seen

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "gemini"
