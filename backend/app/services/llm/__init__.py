"""LLM Service Package.

Exposes the provider-neutral interface, factory, exceptions, and JSON parser utilities.
"""

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
from app.services.llm.json_parser import (
    extract_json_str,
    normalize_llm_output,
    parse_llm_json,
    strip_html,
    validate_llm_schema,
)

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "get_llm_provider",
    "LLMProviderError",
    "LLMError",
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
    "parse_llm_json",
    "extract_json_str",
    "validate_llm_schema",
    "normalize_llm_output",
    "strip_html",
]
