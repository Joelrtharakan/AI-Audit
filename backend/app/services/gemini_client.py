"""Client for Google's Gemini generateContent REST API.

Third in the provider failover order (Groq -> OpenRouter -> Gemini).
Implemented with plain httpx against the public REST endpoint, matching the
project's existing pattern for groq_client.py/openrouter_client.py, rather
than adding the google-generativeai SDK as a new dependency -- Gemini's
REST surface is small and stable enough not to need it here.

Gemini's request/response shape differs from the OpenAI-style chat
completions used by Groq/OpenRouter: messages become `contents` with
role "user"/"model" (no "assistant"), and any "system" message is passed
separately via `systemInstruction`. This client translates between the
two shapes so callers keep using the same OpenAI-style `messages` list
as every other provider.
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


class GeminiError(LLMError):
    pass


class GeminiRateLimitedError(GeminiError, LLMRateLimitedError):
    pass


class GeminiTimeoutError(GeminiError, LLMTimeoutError):
    pass


class GeminiNetworkError(GeminiError, LLMNetworkError):
    pass


class GeminiAuthenticationError(GeminiError, LLMAuthenticationError):
    pass


class GeminiServerError(GeminiError, LLMServerError):
    pass


class GeminiInvalidResponseError(GeminiError, LLMInvalidResponseError):
    pass


class GeminiConfigurationError(GeminiError, LLMConfigurationError):
    pass


def _to_gemini_contents(messages: list[dict[str, str]]) -> tuple[list[dict], dict | None]:
    """Translates OpenAI-style messages into Gemini's `contents` array plus
    an optional `systemInstruction`. Gemini has no "system" role in
    `contents` and uses "model" (not "assistant") for prior model turns."""
    system_parts: list[str] = []
    contents: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")
        text = msg.get("content", "")
        if role == "system":
            system_parts.append(text)
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": text}]})

    system_instruction = {"parts": [{"text": "\n\n".join(system_parts)}]} if system_parts else None
    return contents, system_instruction


class GeminiClient:
    def __init__(self, timeout_seconds: float | None = None) -> None:
        self._settings = get_settings()
        self._timeout = timeout_seconds or self._settings.gemini_timeout_seconds

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format_json: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        settings = self._settings
        if not settings.google_api_key:
            raise GeminiConfigurationError("GOOGLE_API_KEY is not configured.")

        contents, system_instruction = _to_gemini_contents(messages)

        generation_config: dict = {
            "temperature": temperature,
            # Newer Gemini models spend part of maxOutputTokens on hidden
            # "thinking" tokens before the visible answer -- for this
            # pipeline's structured-extraction/synthesis calls that's pure
            # overhead, not a benefit, and can silently exhaust a tight
            # max_tokens budget before any visible text is produced (the
            # response comes back 200 OK with finishReason=MAX_TOKENS and
            # empty content, which the empty-completion check below would
            # otherwise misreport as a provider failure). Disabling it
            # keeps the full token budget for the actual answer.
            "thinkingConfig": {"thinkingBudget": 0},
        }
        if response_format_json:
            generation_config["responseMimeType"] = "application/json"
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        payload: dict = {"contents": contents, "generationConfig": generation_config}
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        url = (
            f"{settings.gemini_base_url.rstrip('/')}/models/{settings.gemini_model}:generateContent"
        )

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    url,
                    headers={"Content-Type": "application/json", "x-goog-api-key": settings.google_api_key},
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            logger.warning("Gemini request timed out.")
            raise GeminiTimeoutError("Gemini API request timed out.") from exc
        except httpx.RequestError as exc:
            logger.warning("Gemini network error: %s", type(exc).__name__)
            raise GeminiNetworkError("Gemini API request failed due to network error.") from exc

        if response.status_code == 429:
            retry_after_hdr = response.headers.get("retry-after")
            try:
                retry_after_sec = float(retry_after_hdr) if retry_after_hdr else 2.0
            except ValueError:
                retry_after_sec = 2.0
            logger.warning("Gemini rate limited (429) — Retry-After: %.1fs", retry_after_sec)
            raise GeminiRateLimitedError(
                f"Gemini API rate limit reached (429). Retry-After: {retry_after_sec}s",
                retry_after=retry_after_sec,
            )

        if response.status_code in (401, 403):
            logger.error("Gemini authentication error %s", response.status_code)
            raise GeminiAuthenticationError(f"Gemini API authentication failed ({response.status_code}).")

        if response.status_code >= 500:
            logger.error("Gemini server error %s", response.status_code)
            raise GeminiServerError(f"Gemini API server error ({response.status_code}).")

        if response.status_code >= 400:
            logger.error("Gemini HTTP error %s", response.status_code)
            raise GeminiError(f"Gemini API HTTP error {response.status_code}.")

        try:
            data = response.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"]
        except (ValueError, KeyError, IndexError) as exc:
            raise GeminiInvalidResponseError("Unexpected Gemini response shape.") from exc
        if not content:
            raise GeminiInvalidResponseError("Gemini returned an empty completion.")
        return content
