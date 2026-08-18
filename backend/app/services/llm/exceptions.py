"""Normalized LLM exception taxonomy.

All provider implementations (Ollama, GitHub Copilot SDK, Cloud Router) normalize
provider-specific network, authentication, timeout, rate-limiting, and decoding
errors into these common exception classes so that the orchestration and
deterministic validation layers never depend on provider-specific errors.
"""

from __future__ import annotations


class LLMProviderError(RuntimeError):
    """Base class for all LLM provider failures."""


# Backward compatibility alias
LLMError = LLMProviderError


class LLMAuthenticationError(LLMProviderError):
    """Raised when provider credentials/tokens are missing, invalid, or expired (HTTP 401/403)."""


class LLMConnectionError(LLMProviderError):
    """Raised on connection-level failures (DNS resolution, refused connection, reset, unreachable host)."""


# Backward compatibility alias
LLMNetworkError = LLMConnectionError


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM generation or session request exceeds its configured timeout deadline."""


class LLMRateLimitError(LLMProviderError):
    """Raised when a provider rejects a request due to rate-limiting (HTTP 429 or quota limit)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# Backward compatibility alias
LLMRateLimitedError = LLMRateLimitError


class LLMInvalidResponseError(LLMProviderError):
    """Raised when a 200 OK was received but the response payload was empty, malformed, or missing expected content."""


class LLMUnavailableError(LLMProviderError):
    """Raised when a model or service is currently unavailable (e.g. 503, model not loaded, or all fallback providers exhausted)."""

    def __init__(self, message: str, provider_statuses: dict[str, dict] | None = None) -> None:
        super().__init__(message)
        self.provider_statuses = provider_statuses or {}


# Backward compatibility alias
AllLLMProvidersUnavailableError = LLMUnavailableError


class LLMConfigurationError(LLMProviderError):
    """Raised when provider configuration is invalid or missing required runtime prerequisites."""


# Backward compatibility alias
NoLLMProviderConfiguredError = LLMConfigurationError


class UnsupportedLLMProviderError(LLMConfigurationError):
    """Raised when an unknown or unsupported provider name is specified in configuration."""
