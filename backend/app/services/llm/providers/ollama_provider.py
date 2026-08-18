"""Production Ollama Provider for LQMS Audit Investigation Engine.

Handles native Ollama API communication with configurable model selection,
thinking mode controls, context size, keep-alive, metrics logging, and fast failure.
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from typing import Any
import httpx

from app.config import get_settings
from app.services.llm.base import LLMProvider, LLMResponse
from app.services.llm.exceptions import (
    LLMConnectionError,
    LLMInvalidResponseError,
    LLMProviderError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

# One persistent, connection-pooling AsyncClient per running event loop,
# reused across every node/call instead of opening a fresh TCP connection per request.
_client_lock = asyncio.Lock()
_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = weakref.WeakKeyDictionary()


async def _get_shared_client() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is not None and not client.is_closed:
        return client
    async with _client_lock:
        client = _clients.get(loop)
        if client is not None and not client.is_closed:
            return client
        client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )
        _clients[loop] = client
        return client


class OllamaProvider(LLMProvider):
    """Ollama implementation of LLMProvider for local inference."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        settings = get_settings()
        self._settings = settings
        self._base_url = base_url or settings.ollama_base_url
        self._model = model or settings.ollama_model
        self._timeout = timeout_seconds or settings.ollama_timeout_seconds

    def _get_native_url(self, endpoint: str) -> str:
        base = self._base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return f"{base}/{endpoint.lstrip('/')}"

    async def check_health(self) -> dict[str, Any]:
        """Check server reachability and model availability."""
        url = self._get_native_url("/api/tags")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    models = [m.get("name") for m in resp.json().get("models", [])]
                    target_model = self._model
                    is_available = target_model in models or any(m.startswith(target_model) for m in models)
                    return {
                        "available": True,
                        "provider": "ollama",
                        "model": target_model,
                        "model_installed": is_available,
                        "installed_models": models,
                    }
        except Exception as exc:
            logger.warning("Ollama health check failed: %s", exc)
        return {
            "available": False,
            "provider": "ollama",
            "model": self._model,
            "model_installed": False,
            "installed_models": [],
        }

    async def generate(
        self,
        *,
        node: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        response_format: str | None = None,
        num_ctx: int | None = None,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        settings = self._settings
        url = self._get_native_url("/api/chat")
        effective_num_ctx = num_ctx or settings.ollama_num_ctx
        effective_timeout = timeout_seconds or self._timeout
        effective_max_tokens = max_output_tokens or 1024

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "think": settings.ollama_thinking,
            "keep_alive": settings.ollama_keep_alive,
            "options": {
                "temperature": temperature if temperature is not None else settings.ollama_temperature,
                "num_ctx": effective_num_ctx,
                "num_predict": effective_max_tokens,
            },
        }
        if response_format == "json":
            payload["format"] = "json"

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_input_tokens = prompt_chars // 4
        logger.info(
            "LLM REQUEST provider=ollama model=%s node=%s prompt_chars=%d estimated_input_tokens=%d "
            "max_output_tokens=%d temperature=%.2f think=%s num_ctx=%d keep_alive=%s",
            self._model,
            node,
            prompt_chars,
            estimated_input_tokens,
            effective_max_tokens,
            payload["options"]["temperature"],
            settings.ollama_thinking,
            effective_num_ctx,
            settings.ollama_keep_alive,
        )

        if estimated_input_tokens + effective_max_tokens > effective_num_ctx:
            logger.warning(
                "LLM REQUEST provider=ollama node=%s estimated_input_tokens(%d) + max_output_tokens(%d) "
                "exceeds num_ctx(%d) -- context truncation/shifting is likely",
                node,
                estimated_input_tokens,
                effective_max_tokens,
                effective_num_ctx,
            )

        t_start = time.monotonic()
        try:
            client = await _get_shared_client()
            resp = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=effective_timeout,
            )
        except httpx.TimeoutException as exc:
            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            logger.error(
                "LLM REQUEST FAILED provider=ollama node=%s model=%s failure_type=TIMEOUT elapsed_ms=%d "
                "timeout_setting_s=%.1f requested_max_output=%d estimated_input_tokens=%d num_ctx=%d",
                node,
                self._model,
                elapsed_ms,
                effective_timeout,
                effective_max_tokens,
                estimated_input_tokens,
                effective_num_ctx,
            )
            raise LLMTimeoutError(f"Ollama request timed out after {effective_timeout}s.") from exc
        except httpx.HTTPError as exc:
            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            logger.error(
                "LLM REQUEST FAILED provider=ollama node=%s model=%s failure_type=CONNECTION_ERROR elapsed_ms=%d "
                "err=%s requested_max_output=%d estimated_input_tokens=%d num_ctx=%d",
                node,
                self._model,
                elapsed_ms,
                exc,
                effective_max_tokens,
                estimated_input_tokens,
                effective_num_ctx,
            )
            raise LLMConnectionError(
                f"Ollama connection failed: {exc}. Is Ollama running at {self._base_url}?"
            ) from exc

        elapsed_ms = int((time.monotonic() - t_start) * 1000)

        if resp.status_code >= 400:
            logger.error(
                "LLM REQUEST FAILED provider=ollama node=%s model=%s failure_type=HTTP_ERROR status=%d elapsed_ms=%d",
                node,
                self._model,
                resp.status_code,
                elapsed_ms,
            )
            raise LLMProviderError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 0)
        tokens_per_sec = round((eval_count / (eval_duration / 1e9)), 2) if eval_duration > 0 else 0.0
        prompt_eval_count = data.get("prompt_eval_count", 0)
        prompt_eval_ms = round(data.get("prompt_eval_duration", 0) / 1e6)
        load_ms = round(data.get("load_duration", 0) / 1e6)
        done_reason = data.get("done_reason", "unknown")

        logger.info(
            "LLM RESPONSE provider=ollama model=%s node=%s elapsed_ms=%d prompt_eval_count=%d prompt_eval_ms=%d "
            "load_ms=%d tokens_generated=%d tokens_per_second=%.2f done_reason=%s max_output_tokens=%d success=true",
            self._model,
            node,
            elapsed_ms,
            prompt_eval_count,
            prompt_eval_ms,
            load_ms,
            eval_count,
            tokens_per_sec,
            done_reason,
            effective_max_tokens,
        )

        try:
            content = data["message"]["content"]
            if not content or not content.strip():
                raise LLMInvalidResponseError("Ollama returned empty completion content.")
        except (KeyError, TypeError) as exc:
            raise LLMInvalidResponseError(f"Unexpected Ollama response structure: {data}") from exc

        raw_metadata = {
            "node": node,
            "done_reason": done_reason,
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "max_output_tokens": effective_max_tokens,
            "elapsed_ms": elapsed_ms,
            "load_ms": load_ms,
            "hit_output_limit": done_reason == "length",
        }

        return LLMResponse(
            content=content,
            provider="ollama",
            model=self._model,
            latency_ms=elapsed_ms,
            input_tokens=prompt_eval_count or estimated_input_tokens,
            output_tokens=eval_count,
            finish_reason=done_reason,
            raw_metadata=raw_metadata,
        )
