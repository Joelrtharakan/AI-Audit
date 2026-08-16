"""Production Ollama Client for LQMS Audit Investigation Engine.

Handles native Ollama API communication with configurable model selection,
thinking mode controls, context size, keep-alive, metrics logging, and fast failure.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
import weakref
import httpx

from app.config import get_settings
from app.services.llm_client import LLMClient, LLMError, LLMNetworkError, LLMTimeoutError

logger = logging.getLogger(__name__)

# Set by a node right before it calls chat_completion, purely so the request/
# response log lines below can be attributed to a node (e.g. "core_synthesis"
# vs "critic") without threading a `node` parameter through the shared
# provider-agnostic LLMClient protocol used by every provider client and the
# router. Same ContextVar-per-asyncio-task pattern as llm_router's
# _last_call_metadata. Only meaningful when llm_provider="ollama" (the
# default and only path that constructs OllamaClient directly).
_current_node: contextvars.ContextVar[str] = contextvars.ContextVar(
    "ollama_current_node", default="unknown"
)


def set_current_node(node: str) -> None:
    _current_node.set(node)


# Ground-truth metadata from the most recent chat_completion call on this
# asyncio task, so a caller can inspect WHY generation stopped (Ollama's own
# `done_reason`: "stop" = the model finished naturally, "length" = it hit
# num_predict) instead of inferring truncation from
# generated_tokens == max_output_tokens, which is unreliable -- a model can
# legitimately fill the entire budget with valid, complete JSON.
_last_call_metadata: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "ollama_last_call_metadata", default={}
)


def get_last_call_metadata() -> dict:
    return _last_call_metadata.get()


class OllamaError(LLMError):
    pass


# One persistent, connection-pooling AsyncClient per running event loop,
# reused across every node/call instead of opening a fresh TCP connection
# per request (Section 6): avoids paying handshake/connection-setup cost on
# every single one of extraction/core_synthesis/critic's calls.
#
# Keyed by the event LOOP OBJECT itself (via a WeakKeyDictionary), never by
# id(loop): a plain dict keyed by id() is unsafe here because a garbage-
# collected loop's memory address can be reused by a brand-new loop (this
# happens routinely under pytest-asyncio, which creates a fresh loop per
# test), which would hand back an httpx.AsyncClient whose connection pool is
# still bound to the OLD, now-dead loop's transport -- using it from the new
# loop doesn't raise, it just hangs forever waiting on I/O that can never
# complete. WeakKeyDictionary ties the cache entry to the loop's actual
# lifetime, so a dead loop's entry is dropped automatically instead of
# silently colliding with a new one.
_client_lock = asyncio.Lock()
_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = weakref.WeakKeyDictionary()


async def _get_shared_client() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is not None and not client.is_closed:
        # timeout is set per-request below (httpx supports a per-call
        # `timeout=` override), so a shared client is safe to reuse even
        # though individual calls specify different timeouts.
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
        num_ctx: int | None = None,
    ) -> str:
        settings = self._settings
        url = self._get_native_url("/api/chat")
        effective_num_ctx = num_ctx or settings.ollama_num_ctx

        payload: dict = {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
            "think": settings.ollama_thinking,
            "keep_alive": settings.ollama_keep_alive,
            "options": {
                "temperature": temperature if temperature is not None else settings.ollama_temperature,
                "num_ctx": effective_num_ctx,
                "num_predict": max_tokens or 1024,
            },
        }
        if response_format_json:
            payload["format"] = "json"

        node = _current_node.get()
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        # Rough heuristic only (~4 chars/token for English prose); the
        # post-call log below reports Ollama's own prompt_eval_count, which
        # is the ground truth -- this estimate exists so a caller can be
        # warned about an oversized prompt BEFORE paying for the request.
        estimated_input_tokens = prompt_chars // 4
        logger.info(
            "OLLAMA REQUEST node=%s model=%s prompt_chars=%d estimated_input_tokens=%d "
            "max_output_tokens=%d temperature=%.2f think=%s num_ctx=%d keep_alive=%s",
            node,
            settings.ollama_model,
            prompt_chars,
            estimated_input_tokens,
            payload["options"]["num_predict"],
            payload["options"]["temperature"],
            settings.ollama_thinking,
            effective_num_ctx,
            settings.ollama_keep_alive,
        )
        if estimated_input_tokens + payload["options"]["num_predict"] > effective_num_ctx:
            logger.warning(
                "OLLAMA REQUEST node=%s estimated_input_tokens(%d) + max_output_tokens(%d) "
                "exceeds num_ctx(%d) -- context truncation/shifting is likely, which can make "
                "this call far slower than its token count would suggest",
                node, estimated_input_tokens, payload["options"]["num_predict"], effective_num_ctx,
            )

        t_start = time.monotonic()
        try:
            client = await _get_shared_client()
            resp = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            logger.error(
                "OLLAMA REQUEST FAILED node=%s model=%s failure_type=TIMEOUT elapsed_ms=%d "
                "timeout_setting_s=%.1f requested_max_output=%d estimated_input_tokens=%d "
                "num_ctx=%d think=%s",
                node, settings.ollama_model, elapsed_ms, self._timeout,
                payload["options"]["num_predict"], estimated_input_tokens,
                effective_num_ctx, settings.ollama_thinking,
            )
            _last_call_metadata.set({"failure_type": "TIMEOUT"})
            raise LLMTimeoutError(f"Ollama request timed out after {self._timeout}s.") from exc
        except httpx.HTTPError as exc:
            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            logger.error(
                "OLLAMA REQUEST FAILED node=%s model=%s failure_type=PROVIDER_FAILURE elapsed_ms=%d "
                "err=%s requested_max_output=%d estimated_input_tokens=%d num_ctx=%d think=%s",
                node, settings.ollama_model, elapsed_ms, exc,
                payload["options"]["num_predict"], estimated_input_tokens,
                effective_num_ctx, settings.ollama_thinking,
            )
            _last_call_metadata.set({"failure_type": "PROVIDER_FAILURE"})
            raise LLMNetworkError(f"Ollama connection failed: {exc}. Is Ollama running at {settings.ollama_base_url}?") from exc

        elapsed_ms = int((time.monotonic() - t_start) * 1000)

        if resp.status_code >= 400:
            logger.error(
                "OLLAMA REQUEST FAILED node=%s model=%s failure_type=HTTP_ERROR status=%d elapsed_ms=%d",
                node, settings.ollama_model, resp.status_code, elapsed_ms,
            )
            _last_call_metadata.set({"failure_type": "PROVIDER_FAILURE"})
            raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 0)
        tokens_per_sec = round((eval_count / (eval_duration / 1e9)), 2) if eval_duration > 0 else 0.0
        # Ground-truth split of where the elapsed time actually went (Ollama
        # reports these in nanoseconds): prompt_eval = prefill of the input
        # context, eval = generation of the output tokens, load = time spent
        # loading/swapping the model in (should be ~0 while it stays warm).
        prompt_eval_count = data.get("prompt_eval_count", 0)
        prompt_eval_ms = round(data.get("prompt_eval_duration", 0) / 1e6)
        load_ms = round(data.get("load_duration", 0) / 1e6)
        # Ollama's own signal for WHY generation stopped: "stop" = the model
        # emitted its own end-of-turn/end-of-JSON naturally; "length" = it
        # was cut off at num_predict. This is the ground truth a caller
        # should use to decide whether a response might be truncated --
        # generated_tokens == max_output_tokens alone is NOT reliable proof,
        # since a model can legitimately use the full budget and still
        # finish (done_reason=="stop") right at the boundary.
        done_reason = data.get("done_reason", "unknown")

        logger.info(
            "provider=ollama model=%s node=%s elapsed_ms=%d prompt_eval_count=%d prompt_eval_ms=%d "
            "load_ms=%d tokens_generated=%d tokens_per_second=%.2f done_reason=%s max_output_tokens=%d "
            "success=true",
            settings.ollama_model,
            node,
            elapsed_ms,
            prompt_eval_count,
            prompt_eval_ms,
            load_ms,
            eval_count,
            tokens_per_sec,
            done_reason,
            payload["options"]["num_predict"],
        )

        _last_call_metadata.set({
            "node": node,
            "done_reason": done_reason,
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "max_output_tokens": payload["options"]["num_predict"],
            "elapsed_ms": elapsed_ms,
            "load_ms": load_ms,
            "hit_output_limit": done_reason == "length",
        })

        try:
            content = data["message"]["content"]
            if not content or not content.strip():
                raise OllamaError("Ollama returned empty completion content.")
            return content
        except (KeyError, TypeError) as exc:
            raise OllamaError(f"Unexpected Ollama response structure: {data}") from exc
