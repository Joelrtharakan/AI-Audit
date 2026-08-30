"""Authoritative LLM provider factory.

Single entry-point. Returns the ONE application-owned inference boundary --
`LiteLLMProvider` -- bound to the immutable `LLMExecutionConfig` that was frozen
for this request (`app.services.llm.execution.begin_request`), or, for direct /
non-request callers (unit tests), resolved from settings.

There is deliberately NO provider fan-out / failover / multi-provider router in
the inference path. `groq` / `gemini` / `openrouter` are single native LiteLLM
routes; a failure surfaces as the app's normal degraded / fail-closed behavior,
never a silent switch to another provider.
"""

from __future__ import annotations

from typing import Any

from app.services.llm.base import LLMProvider
from app.services.llm.execution import (
    UnknownProviderError,
    effective_execution_config,
    resolve_execution_config,
)
from app.services.llm.exceptions import UnsupportedLLMProviderError
from app.services.llm.providers.litellm_provider import LiteLLMProvider


def get_llm_provider(
    provider_name: str | None = None,
    timeout_seconds: float | None = None,
    model: str | None = None,
    **_ignored: Any,
) -> LLMProvider:
    """Return the LiteLLM-backed provider for the active execution config.

    Args:
        provider_name: optional explicit provider override (unit tests / health
            endpoints). When omitted, the request-frozen config is used, falling
            back to settings.
        timeout_seconds: optional per-operation timeout override.
        model: optional explicit model override (only meaningful with
            provider_name).
    """
    if provider_name is None and model is None:
        return LiteLLMProvider(effective_execution_config(), timeout_seconds=timeout_seconds)

    try:
        config = resolve_execution_config(provider=provider_name, model=model)
    except UnknownProviderError as exc:
        raise UnsupportedLLMProviderError(str(exc)) from exc
    return LiteLLMProvider(config, timeout_seconds=timeout_seconds)
