"""LLM provider registry and factory.

To add a new provider, import its class here and add it to _REGISTRY.
"""

import config
from .base import LLMProvider
from .groq_provider import GroqProvider
from .gemini_provider import GeminiProvider

# ── Registry ─────────────────────────────────────────────────────────────────
_REGISTRY: dict[str, type[LLMProvider]] = {
    "groq":   GroqProvider,
    "gemini": GeminiProvider,
}


def get_provider(name: str | None = None, **kwargs) -> LLMProvider:
    """Instantiate and return the requested LLM provider.

    Args:
        name:    Provider slug. Defaults to config.LLM_PROVIDER.
        **kwargs: Passed through to the provider constructor.
    """
    provider_name = (name or config.LLM_PROVIDER).lower()
    if provider_name not in _REGISTRY:
        available = ", ".join(_REGISTRY)
        raise ValueError(
            f"Unknown LLM provider '{provider_name}'. Available: {available}"
        )
    return _REGISTRY[provider_name](**kwargs)


def list_providers() -> list[str]:
    """Return all registered provider names."""
    return list(_REGISTRY)
