"""Thin client for a local Ollama server's OpenAI-compatible chat completions
endpoint. Mirrors OpenRouterClient's interface exactly (same method
signature, same retry/backoff pattern) so finding_analysis_service.py's
get_llm_client() factory can hand either one to the rest of the pipeline
without any provider-specific branching downstream.

No API key: Ollama is a local, unauthenticated server. This is intentionally
for fast local dev iteration, not production -- see README section 6b.
"""

import logging

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.services.llm_client import LLMError, LLMRateLimitedError

logger = logging.getLogger(__name__)


class OllamaError(LLMError):
    pass


class OllamaRateLimitedError(OllamaError, LLMRateLimitedError):
    pass


class OllamaClient:
    def __init__(self, timeout_seconds: float = 120.0) -> None:
        self._settings = get_settings()
        self._timeout = timeout_seconds

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(OllamaRateLimitedError),
    )
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format_json: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        settings = self._settings

        payload: dict = {
            "model": settings.ollama_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            # Ollama's OpenAI-compat layer also honors "num_predict" natively; send
            # both so the token budget applies regardless of which one it reads.
            payload["max_tokens"] = max_tokens
            payload["num_predict"] = max_tokens

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{settings.ollama_base_url}/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise OllamaError("Ollama request timed out.") from exc
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Ollama request failed: {exc}. Is `ollama serve` running at "
                f"{settings.ollama_base_url}?"
            ) from exc

        if resp.status_code == 429:
            # Ollama itself doesn't rate-limit, but a proxy in front of it might --
            # handled the same way as OpenRouter for a uniform retry story.
            logger.warning("Ollama rate limited (429); will retry with backoff.")
            raise OllamaRateLimitedError("Rate limited by Ollama.")

        if resp.status_code >= 400:
            logger.error("Ollama error %s", resp.status_code)
            raise OllamaError(f"Ollama returned status {resp.status_code}.")

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise OllamaError("Unexpected Ollama response shape.") from exc
