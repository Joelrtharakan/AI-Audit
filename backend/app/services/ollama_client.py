"""Production Ollama Client for LQMS Audit Investigation Engine.

Handles native Ollama API communication with configurable model selection,
thinking mode controls, context size, keep-alive, metrics logging, and fast failure.
"""

from __future__ import annotations

import logging
import time
import httpx

from app.config import get_settings
from app.services.llm_client import LLMClient, LLMError, LLMNetworkError, LLMTimeoutError

logger = logging.getLogger(__name__)


class OllamaError(LLMError):
    pass


class OllamaClient(LLMClient):
    def __init__(self, timeout_seconds: float | None = None) -> None:
        settings = get_settings()
        self._settings = settings
        self._timeout = timeout_seconds or settings.ollama_timeout_seconds

    def _get_native_url(self, endpoint: str) -> str:
        base = self._settings.ollama_base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return f"{base}/{endpoint.lstrip('/')}"

    async def check_health(self) -> dict[str, bool | str]:
        """Check server reachability and model availability."""
        url = self._get_native_url("/api/tags")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    models = [m.get("name") for m in resp.json().get("models", [])]
                    target_model = self._settings.ollama_model
                    is_available = target_model in models or any(m.startswith(target_model) for m in models)
                    return {
                        "available": True,
                        "model": target_model,
                        "model_installed": is_available,
                        "installed_models": models,
                    }
        except Exception as exc:
            logger.warning("Ollama health check failed: %s", exc)
        return {
            "available": False,
            "model": self._settings.ollama_model,
            "model_installed": False,
            "installed_models": [],
        }

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format_json: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        settings = self._settings
        url = self._get_native_url("/api/chat")

        options: dict = {
            "temperature": temperature if temperature is not None else settings.ollama_temperature,
            "num_ctx": settings.ollama_num_ctx,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict = {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
            "think": settings.ollama_thinking,
            "keep_alive": settings.ollama_keep_alive,
            "options": options,
        }
        if response_format_json:
            payload["format"] = "json"

        t_start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            logger.error("provider=ollama model=%s event=timeout elapsed_ms=%d", settings.ollama_model, elapsed_ms)
            raise LLMTimeoutError(f"Ollama request timed out after {self._timeout}s.") from exc
        except httpx.HTTPError as exc:
            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            logger.error("provider=ollama model=%s event=network_error elapsed_ms=%d err=%s", settings.ollama_model, elapsed_ms, exc)
            raise LLMNetworkError(f"Ollama connection failed: {exc}. Is Ollama running at {settings.ollama_base_url}?") from exc

        elapsed_ms = int((time.monotonic() - t_start) * 1000)

        if resp.status_code >= 400:
            logger.error("provider=ollama model=%s event=http_error status=%d elapsed_ms=%d", settings.ollama_model, resp.status_code, elapsed_ms)
            raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 0)
        tokens_per_sec = round((eval_count / (eval_duration / 1e9)), 2) if eval_duration > 0 else 0.0

        logger.info(
            "provider=ollama model=%s elapsed_ms=%d tokens_generated=%d tokens_per_second=%.2f success=true",
            settings.ollama_model,
            elapsed_ms,
            eval_count,
            tokens_per_sec,
        )

        try:
            content = data["message"]["content"]
            if not content or not content.strip():
                raise OllamaError("Ollama returned empty completion content.")
            return content
        except (KeyError, TypeError) as exc:
            raise OllamaError(f"Unexpected Ollama response structure: {data}") from exc
