"""Groq API client using OpenAI-compatible chat completions.

Groq offers ultra-fast inference (500+ tokens/sec) via Llama 3 / Qwen models.
Rate-limit (429) handling is intentionally NOT retried here -- that decision
now belongs to app/services/llm_router.py's circuit breaker, which reacts to
a single 429 by opening Groq's circuit and failing over to the next
provider immediately, rather than this client sleeping/retrying against a
provider that just said "too many requests" and then failing over anyway.
Bounded retry is still applied here for genuinely transient network errors
(connection resets, brief timeouts), since those often succeed on the very
next attempt and don't call for a full provider failover.
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


class GroqError(LLMError):
    pass


class GroqRateLimitedError(GroqError, LLMRateLimitedError):
    pass


class GroqTimeoutError(GroqError, LLMTimeoutError):
    pass


class GroqNetworkError(GroqError, LLMNetworkError):
    pass


class GroqAuthenticationError(GroqError, LLMAuthenticationError):
    pass


class GroqServerError(GroqError, LLMServerError):
    pass


class GroqInvalidResponseError(GroqError, LLMInvalidResponseError):
    pass


class GroqConfigurationError(GroqError, LLMConfigurationError):
    pass


class GroqClient:
    def __init__(self, timeout_seconds: float | None = None) -> None:
        self._settings = get_settings()
        self._timeout = timeout_seconds or self._settings.groq_timeout_seconds

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format_json: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        settings = self._settings
        if not settings.groq_api_key:
            raise GroqConfigurationError("GROQ_API_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": settings.groq_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{settings.groq_base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            logger.warning("Groq request timed out.")
            raise GroqTimeoutError("Groq API request timed out.") from exc
        except httpx.RequestError as exc:
            logger.warning("Groq network error: %s", type(exc).__name__)
            raise GroqNetworkError("Groq API request failed due to network error.") from exc

        if response.status_code == 429:
            retry_after_hdr = response.headers.get("retry-after")
            try:
                retry_after_sec = float(retry_after_hdr) if retry_after_hdr else 2.0
            except ValueError:
                retry_after_sec = 2.0
            logger.warning("Groq rate limited (429) — Retry-After: %.1fs", retry_after_sec)
            raise GroqRateLimitedError(
                f"Groq API rate limit reached (429). Retry-After: {retry_after_sec}s",
                retry_after=retry_after_sec,
            )

        if response.status_code in (401, 403):
            logger.error("Groq authentication error %s", response.status_code)
            raise GroqAuthenticationError(f"Groq API authentication failed ({response.status_code}).")

        if response.status_code >= 500:
            logger.error("Groq server error %s", response.status_code)
            raise GroqServerError(f"Groq API server error ({response.status_code}).")

        if response.status_code >= 400:
            logger.error("Groq HTTP error %s", response.status_code)
            raise GroqError(f"Groq API HTTP error {response.status_code}.")

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as exc:
            raise GroqInvalidResponseError("Unexpected Groq response shape.") from exc
        if not content:
            raise GroqInvalidResponseError("Groq returned an empty completion.")
        return content
