"""Authoritative LLM Provider Factory.

Single entry-point for selecting and instantiating the configured LLMProvider
(Ollama for development/testing and degraded fallback, Microsoft 365 Copilot for
production -- reached through LiteLLM's custom-provider mechanism).
"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.services.llm.base import LLMProvider
from app.services.llm.exceptions import UnsupportedLLMProviderError
from app.services.llm.providers.microsoft_copilot_provider import MicrosoftCopilotProvider
from app.services.llm.providers.ollama_provider import OllamaProvider


def get_llm_provider(
    provider_name: str | None = None,
    timeout_seconds: float | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """Return the authoritative LLMProvider instance based on runtime configuration.

    Args:
        provider_name: Optional explicit provider override ("ollama", "copilot"). Defaults to settings.llm_provider.
        timeout_seconds: Optional operation-specific timeout in seconds.
    """
    settings = get_settings()
    name = (provider_name or settings.llm_provider).strip().lower()

    if name == "ollama":
        model = kwargs.pop("model", None) or settings.ollama_model
        return OllamaProvider(model=model, timeout_seconds=timeout_seconds, **kwargs)

    if name in ("microsoft_copilot", "microsoft-copilot", "m365_copilot", "m365-copilot", "copilot"):
        model = kwargs.pop("model", None)
        return MicrosoftCopilotProvider(model=model, timeout_seconds=timeout_seconds, **kwargs)

    # Legacy router fallback for cloud APIs (Groq, OpenRouter, Gemini)
    if name in ("groq", "openrouter", "gemini", "router"):
        from app.services.llm_router import get_llm_router
        return get_llm_router()

    raise UnsupportedLLMProviderError(
        f"Unsupported LLM provider: '{name}'. Valid providers are 'ollama', 'microsoft_copilot', "
        "'groq', 'openrouter', 'gemini'."
    )
