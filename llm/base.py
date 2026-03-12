"""Abstract base class for LLM providers.

Adding a new provider:
  1. Create a file in llm/ (e.g. llm/openai_provider.py)
  2. Subclass LLMProvider and implement chat() + chat_stream()
  3. Register it in llm/factory.py
"""

from abc import ABC, abstractmethod
from typing import Iterator


class LLMProvider(ABC):
    """Uniform interface for all LLM back-ends."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Send a list of messages and return the complete response text."""

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
    ) -> Iterator[str]:
        """Yield response tokens as they arrive."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider slug (e.g. 'groq', 'openai')."""
