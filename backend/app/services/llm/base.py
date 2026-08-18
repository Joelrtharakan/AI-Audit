"""Provider-neutral LLM interface and normalized response model.

Every agent node and pipeline service type-hints against `LLMProvider` and
consumes `LLMResponse` or `chat_completion` output, remaining completely decoupled
from provider-specific implementations (Ollama, GitHub Copilot SDK, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Normalized response container produced by any LLM provider."""

    content: str
    provider: str
    model: str
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract interface for pluggable LLM intelligence providers."""

    @abstractmethod
    async def generate(
        self,
        *,
        node: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        response_format: str | None = None,
        num_ctx: int | None = None,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response for a specific agent node execution.

        Args:
            node: The identifying name of the requesting agent node (e.g. 'core_synthesis', 'extraction', 'critic').
            prompt: The user-level prompt text.
            system_prompt: Optional system instruction prompt text.
            temperature: Sampling temperature (0.0 for deterministic tasks).
            max_output_tokens: Optional limit on the number of generated tokens.
            response_format: Optional format constraint (e.g. 'json' or None).
            num_ctx: Optional context window token limit (Ollama-specific/ignored by others).
            timeout_seconds: Optional per-call override timeout in seconds.
        """

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format_json: bool = False,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        node: str = "unknown",
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Provider-neutral chat completion bridge for message-based callers.

        Extracts system and user prompts from the message list and delegates
        to `generate()`.
        """
        system_parts: list[str] = []
        user_parts: list[str] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        prompt = "\n\n".join(user_parts) if user_parts else ""

        response = await self.generate(
            node=node,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_format="json" if response_format_json else None,
            num_ctx=num_ctx,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        return response.content

    async def check_health(self) -> dict[str, Any]:
        """Check provider connectivity, authentication, and model readiness."""
        return {"available": True, "provider": self.__class__.__name__}
