"""Provider-agnostic LLM client contract.

Every step of the pipeline (observation quality, extraction, classification,
generation) type-hints against `LLMClient` and catches `LLMError`, never a
concrete provider class. `get_llm_client()` is the ONE place that decides
which concrete client(s) actually handle a request -- as of the
multi-provider failover work, that means returning the shared LLMRouter
singleton (see app/services/llm_router.py), which internally tries
Groq -> OpenRouter -> Gemini in that order and only raises LLMError once
every configured provider has failed. Callers never see provider identity;
they only see LLMClient's contract and LLMError/LLMRateLimitedError.
"""

from typing import Protocol


class LLMError(RuntimeError):
    """Base class for every LLM-provider failure. Node-level code already
    catches this broadly (often as `except (LLMError, ValueError, KeyError)`
    or a bare `except Exception`) -- that catching code needs zero changes
    for the router to work, since the router only raises LLMError (after
    exhausting all providers) or returns a successful string, exactly like
    a single-provider client always did."""


class LLMRateLimitedError(LLMError):
    """Raised on HTTP 429. Carries `retry_after` (seconds, from the
    provider's Retry-After header) when available, so the router can use it
    as circuit-breaker cooldown information -- never as a blocking sleep."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMTimeoutError(LLMError):
    """Request exceeded the client's own timeout."""


class LLMNetworkError(LLMError):
    """Connection-level failure (DNS, connection refused, reset, etc.)."""


class LLMAuthenticationError(LLMError):
    """401/403 or equivalent -- retrying the same provider will not help."""


class LLMServerError(LLMError):
    """5xx from the provider."""


class LLMInvalidResponseError(LLMError):
    """2xx response but the body didn't have the expected shape."""


class LLMConfigurationError(LLMError):
    """Provider is not usable in this deployment (e.g. missing API key)."""


class NoLLMProviderConfiguredError(LLMConfigurationError):
    """Raised ONLY when NONE of the providers has an API key configured."""


class AllLLMProvidersUnavailableError(LLMError):
    """Raised when configured providers were attempted but all failed (e.g. 429 rate limited, timeout, 5xx)."""

    def __init__(self, message: str, provider_statuses: dict[str, dict] | None = None) -> None:
        super().__init__(message)
        self.provider_statuses = provider_statuses or {}


class LLMClient(Protocol):
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format_json: bool = False,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
    ) -> str: ...


def get_llm_client(timeout_seconds: float | None = None) -> LLMClient:
    """Returns OllamaClient for local inference, or LLMRouter when explicitly configured.

    `timeout_seconds` lets a caller request an operation-specific timeout
    (e.g. a shorter one for an optional/secondary call, a dedicated one for
    a compact recovery retry) instead of the single global
    `ollama_timeout_seconds` default. Only honored for the direct-Ollama
    path -- the router manages its own per-provider timeouts internally.
    """
    from app.config import get_settings

    settings = get_settings()
    if settings.llm_provider != "ollama":
        from app.services.llm_router import get_llm_router
        return get_llm_router()

    from app.services.ollama_client import OllamaClient
    return OllamaClient(timeout_seconds=timeout_seconds)
