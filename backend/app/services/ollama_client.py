"""Thin client for a local Ollama server's NATIVE /api/chat endpoint (not the
OpenAI-compat /v1/chat/completions layer). Mirrors OpenRouterClient's
interface exactly (same method signature, same retry/backoff pattern) so
finding_analysis_service.py's get_llm_client() factory can hand either one to
the rest of the pipeline without any provider-specific branching downstream.

Uses the native endpoint specifically for "think": false -- hybrid-reasoning
models (e.g. qwen3) emit a large hidden reasoning trace by default that (a)
only the OpenAI-compat layer's "think" field silently ignores, and (b) counts
against the same token budget as the actual JSON content, risking truncated/
empty output on the long prompts this pipeline sends. The native endpoint
honors "think": false and drops the reasoning trace entirely -- verified via
direct API test: OpenAI-compat endpoint burned ~300 completion tokens on a
trivial prompt even with "think": false in the payload (ignored), while the
native endpoint with the same flag used 6. Harmless no-op for non-thinking
models (e.g. qwen2.5) -- verified directly, not assumed.

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
        # settings.ollama_base_url is the OpenAI-compat base (".../v1"); the
        # native endpoint lives one level up at ".../api/chat".
        base = settings.ollama_base_url
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        native_url = f"{base}/api/chat"

        options: dict = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict = {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": options,
        }
        if response_format_json:
            payload["format"] = "json"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    native_url,
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
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise OllamaError("Unexpected Ollama response shape.") from exc
