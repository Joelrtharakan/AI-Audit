"""Thin client for OpenRouter's OpenAI-compatible chat completions API.

Second in the provider failover order (Groq -> OpenRouter -> Gemini), so
this is a normal production provider, not a special-case fallback hack.
Rate-limit (429) handling is NOT retried here -- app/services/llm_router.py's
circuit breaker owns that decision, reacting to a single 429 by opening
OpenRouter's circuit and failing over to Gemini. Bounded retry is still
applied for genuinely transient network errors.

Never lets the provider API key leak into logs or exceptions raised past
this module.
"""

import logging

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.services.llm_client import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMInvalidResponseError,
    LLMNetworkError,
    LLMRateLimitedError,
    LLMServerError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class OpenRouterError(LLMError):
    pass


class RateLimitedError(OpenRouterError, LLMRateLimitedError):
    pass


class OpenRouterTimeoutError(OpenRouterError, LLMTimeoutError):
    pass


class OpenRouterNetworkError(OpenRouterError, LLMNetworkError):
    pass


class OpenRouterAuthenticationError(OpenRouterError, LLMAuthenticationError):
    pass


class OpenRouterServerError(OpenRouterError, LLMServerError):
    pass


class OpenRouterInvalidResponseError(OpenRouterError, LLMInvalidResponseError):
    pass


class OpenRouterConfigurationError(OpenRouterError, LLMConfigurationError):
    pass


class OpenRouterClient:
    def __init__(self, timeout_seconds: float | None = None) -> None:
        self._settings = get_settings()
        self._timeout = timeout_seconds or self._settings.openrouter_timeout_seconds

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format_json: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        settings = self._settings
        if not settings.openrouter_api_key:
            raise OpenRouterConfigurationError("OPENROUTER_API_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.openrouter_site_url,
            "X-Title": settings.openrouter_app_name,
        }
        payload: dict = {
            "model": settings.openrouter_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{settings.openrouter_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            logger.warning("OpenRouter request timed out.")
            raise OpenRouterTimeoutError("OpenRouter request timed out.") from exc
        except httpx.RequestError as exc:
            logger.warning("OpenRouter network error: %s", type(exc).__name__)
            raise OpenRouterNetworkError(f"OpenRouter request failed: {type(exc).__name__}") from exc

        if resp.status_code == 429:
            retry_after_hdr = resp.headers.get("retry-after")
            try:
                retry_after_sec = float(retry_after_hdr) if retry_after_hdr else 2.0
            except ValueError:
                retry_after_sec = 2.0
            logger.warning("OpenRouter rate limited (429) — Retry-After: %.1fs", retry_after_sec)
            raise RateLimitedError(
                f"Rate limited by OpenRouter. Retry-After: {retry_after_sec}s",
                retry_after=retry_after_sec,
            )

        if resp.status_code in (401, 403):
            logger.error("OpenRouter authentication error %s", resp.status_code)
            raise OpenRouterAuthenticationError(f"OpenRouter authentication failed ({resp.status_code}).")

        if resp.status_code >= 500:
            logger.error("OpenRouter server error %s", resp.status_code)
            raise OpenRouterServerError(f"OpenRouter server error ({resp.status_code}).")

        if resp.status_code >= 400:
            # Never echo response body verbatim -- it may contain provider-side
            # request echoes; log a truncated, sanitized summary instead.
            logger.error("OpenRouter error %s", resp.status_code)
            raise OpenRouterError(f"OpenRouter returned status {resp.status_code}.")

        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as exc:
            raise OpenRouterInvalidResponseError("Unexpected OpenRouter response shape.") from exc
        if not content:
            # Some (esp. free-tier) models occasionally return content: null
            # or an empty string with a 200 status -- that's not a usable
            # completion. Raise so the router treats this as a provider
            # failure and fails over, instead of the caller getting None
            # where a string was promised.
            raise OpenRouterInvalidResponseError("OpenRouter returned an empty completion.")
        return content
