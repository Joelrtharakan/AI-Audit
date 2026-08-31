"""Provider-agnostic LLM client contract and legacy compatibility module.

Delegates directly to the unified `app.services.llm` provider architecture.
All agent nodes and pipeline services can type-hint against `LLMClient` or `LLMProvider`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.services.llm.base import LLMProvider, LLMResponse
from app.services.llm.exceptions import (
    AllLLMProvidersUnavailableError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMError,
    LLMInvalidResponseError,
    LLMNetworkError,
    LLMProviderError,
    LLMRateLimitedError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
    NoLLMProviderConfiguredError,
    UnsupportedLLMProviderError,
)
from app.services.llm.factory import get_llm_provider


class LLMServerError(LLMProviderError):
    """5xx from the provider."""


@runtime_checkable
class LLMClient(Protocol):
    """Protocol matching provider chat_completion interface."""

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format_json: bool = False,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        **kwargs,
    ) -> str: ...


def get_llm_client(
    timeout_seconds: float | None = None, model: str | None = None
) -> LLMProvider:
    """Returns the authoritative LLM provider (Ollama, Copilot, or Cloud Router).

    `model` (spec Pass 49 §18): optional per-call model override on the SAME
    configured provider -- lets the two semantic stages use a stronger model
    without touching code or switching providers. `None`/"" -> the global
    configured model. This is configuration authority, never an automatic
    silent switch.
    """
    _m = (model or "").strip() or None
    return get_llm_provider(timeout_seconds=timeout_seconds, model=_m)


__all__ = [
    "LLMClient",
    "LLMProvider",
    "LLMResponse",
    "get_llm_client",
    "get_llm_provider",
    "LLMError",
    "LLMProviderError",
    "LLMAuthenticationError",
    "LLMConnectionError",
    "LLMNetworkError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMRateLimitedError",
    "LLMInvalidResponseError",
    "LLMUnavailableError",
    "AllLLMProvidersUnavailableError",
    "LLMConfigurationError",
    "NoLLMProviderConfiguredError",
    "UnsupportedLLMProviderError",
    "LLMServerError",
]
