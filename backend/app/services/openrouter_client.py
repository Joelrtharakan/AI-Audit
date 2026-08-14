"""Thin client for OpenRouter's OpenAI-compatible chat completions API.

Handles timeouts, retries with exponential backoff (including explicit 429
handling), and never lets the provider API key leak into logs or exceptions
raised past this module.
"""

import logging

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.services.llm_client import LLMError, LLMRateLimitedError

logger = logging.getLogger(__name__)


class OpenRouterError(LLMError):
    pass


class RateLimitedError(OpenRouterError, LLMRateLimitedError):
    pass


class OpenRouterClient:
    def __init__(self, timeout_seconds: float = 120.0) -> None:
        self._settings = get_settings()
        self._timeout = timeout_seconds

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(RateLimitedError),
    )
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format_json: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        settings = self._settings
        if not settings.openrouter_api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not configured.")

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
            raise OpenRouterError("OpenRouter request timed out.") from exc
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

        if resp.status_code == 429:
            logger.warning("OpenRouter rate limited (429); will retry with backoff.")
            raise RateLimitedError("Rate limited by OpenRouter.")

        if resp.status_code >= 400:
            # Never echo response body verbatim -- it may contain provider-side
            # request echoes; log a truncated, sanitized summary instead.
            logger.error("OpenRouter error %s", resp.status_code)
            raise OpenRouterError(f"OpenRouter returned status {resp.status_code}.")

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise OpenRouterError("Unexpected OpenRouter response shape.") from exc
