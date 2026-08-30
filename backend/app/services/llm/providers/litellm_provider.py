"""The single, application-owned LLM inference boundary.

Every intentional LLM stage in the application reaches inference through exactly
one path:

    node -> get_llm_client() -> LiteLLMProvider.generate() -> litellm.acompletion(<one model>)

`LiteLLMProvider` is bound to ONE immutable `LLMExecutionConfig` (provider +
model, resolved once per request in `app.services.llm.execution`). It:

  * makes exactly ONE `litellm.acompletion` call per `generate()` -- no probing,
    no health check, no provider comparison, no fan-out;
  * NEVER falls back to another provider or model. `num_retries` is 0 unless
    `LLM_FALLBACK_ENABLED` is set, and even then LiteLLM retries the SAME model;
  * maps every LiteLLM/provider error into the existing
    `app.services.llm.exceptions` taxonomy so the app's degraded / fail-closed
    handling is unchanged and no raw LiteLLM error reaches the auditor;
  * writes `request_id / provider / model / node / elapsed_ms / success` to the
    logs and to the shared last-call metadata ContextVar.

Provider-specific transport lives behind LiteLLM:
  * Ollama            -> LiteLLM native "ollama_chat/<model>"
  * Microsoft Copilot -> registered CustomLLM handler (_m365_copilot_litellm_handler)
  * GitHub Copilot    -> registered CustomLLM handler (_github_copilot_litellm_handler),
                         wrapping our GitHub-OAuth/session auth (NOT LiteLLM's
                         native device-auth github_copilot route)
  * groq / gemini / openrouter -> LiteLLM native routes
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from app.config import get_settings
from app.services.llm.base import LLMProvider, LLMResponse
from app.services.llm.call_metadata import set_last_call_metadata
from app.services.llm.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMInvalidResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.services.llm.execution import LLMExecutionConfig, effective_execution_config

# Keep LiteLLM fully offline / non-surprising: never fetch its remote model-cost
# map, never raise on an unknown provider param (drop it instead of a hard fail).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

logger = logging.getLogger(__name__)

_handlers_registered = False


def register_all_custom_handlers() -> None:
    """Idempotently register the Microsoft + GitHub CustomLLM handlers in
    `litellm.custom_provider_map`. Called once on first provider construction."""
    global _handlers_registered
    if _handlers_registered:
        return
    settings = get_settings()

    from app.services.llm.providers._m365_copilot_litellm_handler import register as _register_m365
    _register_m365(
        graph_base_url=settings.microsoft_graph_base_url,
        timezone=settings.microsoft_copilot_timezone,
        web_grounding=settings.microsoft_copilot_web_grounding,
    )

    from app.services.llm.providers._github_copilot_litellm_handler import register as _register_gh
    _register_gh()

    import litellm
    litellm.drop_params = True          # unknown provider params are dropped, never a hard error
    litellm.suppress_debug_info = True
    _handlers_registered = True


class LiteLLMProvider(LLMProvider):
    """One provider, one model, one call. No fallback, no fan-out."""

    def __init__(
        self,
        config: LLMExecutionConfig | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        register_all_custom_handlers()
        self._config = config or effective_execution_config()
        self._settings = get_settings()
        self._timeout_override = timeout_seconds

    @property
    def config(self) -> LLMExecutionConfig:
        return self._config

    # -- non-inference health / metadata -----------------------------------

    async def check_health(self) -> dict[str, Any]:
        """Readiness probe. NEVER performs inference."""
        cfg = self._config
        base = {"provider": cfg.provider, "model": cfg.model, "litellm_model": cfg.litellm_model}

        if cfg.provider == "ollama":
            import httpx
            tags_url = (cfg.api_base or "http://localhost:11434").rstrip("/").removesuffix("/v1") + "/api/tags"
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(tags_url)
                    if resp.status_code == 200:
                        names = [m.get("name", "") for m in resp.json().get("models", [])]
                        installed = any(n == cfg.model or n.startswith(cfg.model) for n in names)
                        return {**base, "available": True, "model_installed": installed, "installed_models": names}
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ollama health probe failed: %s", exc)
            return {**base, "available": False, "model_installed": False, "installed_models": []}

        # Token/key presence only -- reachability is confirmed on the first real call.
        has_credential = bool(cfg.api_key)
        return {
            **base,
            "available": has_credential,
            "error": None if has_credential else f"No credential configured for provider '{cfg.provider}'.",
        }

    # -- the single inference path ----------------------------------------

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
        import litellm
        from app.services.llm.execution import current_request_id

        cfg = self._config
        request_id = current_request_id() or "-"
        effective_timeout = timeout_seconds or self._timeout_override or self._settings.ollama_timeout_seconds

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # cfg.extra_params carries the FROZEN provider-specific transport params
        # resolved once at request start (e.g. for ollama: keep_alive, num_ctx,
        # and reasoning_effort="disable" for reasoning models). Forwarded through
        # the single LiteLLM call unchanged.
        params: dict[str, Any] = dict(cfg.extra_params)
        params["temperature"] = temperature if temperature is not None else 0.0
        if max_output_tokens:
            params["max_tokens"] = max_output_tokens
        if response_format == "json":
            params["response_format"] = {"type": "json_object"}
        if cfg.provider == "ollama" and num_ctx:
            params["num_ctx"] = num_ctx  # per-call override wins over the frozen default
        if cfg.api_base:
            params["api_base"] = cfg.api_base
        if cfg.api_key:
            params["api_key"] = cfg.api_key
        # NO cross-provider fallback. Same-model retries only, and only if the
        # operator explicitly enabled it.
        num_retries = self._settings.ollama_max_retries if cfg.fallback_enabled else 0
        params["num_retries"] = num_retries

        prompt_chars = len(prompt) + (len(system_prompt or ""))
        logger.info(
            "LLM REQUEST request_id=%s provider=%s model=%s litellm_route=%s node=%s prompt_chars=%d "
            "num_ctx=%s max_tokens=%s reasoning_effort=%s keep_alive=%s timeout_s=%.1f format=%s retries=%d",
            request_id, cfg.provider, cfg.model, cfg.litellm_model, node, prompt_chars,
            params.get("num_ctx"), params.get("max_tokens"), params.get("reasoning_effort", "n/a"),
            params.get("keep_alive", "n/a"), effective_timeout, response_format or "text", num_retries,
        )

        t0 = time.monotonic()
        try:
            completion = await litellm.acompletion(
                model=cfg.litellm_model,
                messages=messages,
                timeout=effective_timeout,
                **params,
            )
        except Exception as exc:  # noqa: BLE001 - normalized below
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            mapped = _map_litellm_error(exc, cfg, effective_timeout)
            # Raw provider/LiteLLM diagnostics stay in the LOG only.
            logger.warning(
                "LLM RESPONSE request_id=%s provider=%s model=%s node=%s elapsed_ms=%d success=false "
                "failure_type=%s raw=%r",
                request_id, cfg.provider, cfg.model, node, elapsed_ms, type(mapped).__name__, str(exc)[:400],
            )
            set_last_call_metadata({
                "request_id": request_id, "provider_used": cfg.provider, "model": cfg.model,
                "fallback_used": False, "provider_attempts": [cfg.provider],
                "elapsed_ms": elapsed_ms, "node": node, "success": False,
            })
            raise mapped from exc

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        try:
            content = completion.choices[0].message.content or ""
        except (AttributeError, IndexError):
            content = ""
        if not content.strip():
            set_last_call_metadata({
                "request_id": request_id, "provider_used": cfg.provider, "model": cfg.model,
                "fallback_used": False, "provider_attempts": [cfg.provider],
                "elapsed_ms": elapsed_ms, "node": node, "success": False,
            })
            raise LLMInvalidResponseError(f"{cfg.provider}/{cfg.model} returned empty completion content.")

        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) or prompt_chars // 4
        output_tokens = getattr(usage, "completion_tokens", None) or len(content) // 4
        # Reasoning tokens (present only if a provider surfaces them) -- useful
        # to confirm thinking is actually disabled.
        reasoning_tokens = None
        _details = getattr(usage, "completion_tokens_details", None)
        if _details is not None:
            reasoning_tokens = getattr(_details, "reasoning_tokens", None)
        finish_reason = getattr(completion.choices[0], "finish_reason", "stop") or "stop"

        logger.info(
            "LLM RESPONSE request_id=%s provider=%s model=%s litellm_route=%s node=%s elapsed_ms=%d "
            "prompt_tokens=%s output_tokens=%s reasoning_tokens=%s finish_reason=%s retries=%d success=true",
            request_id, cfg.provider, cfg.model, cfg.litellm_model, node, elapsed_ms,
            prompt_tokens, output_tokens, reasoning_tokens, finish_reason, num_retries,
        )

        set_last_call_metadata({
            "request_id": request_id,
            "provider_used": cfg.provider,
            "model": cfg.model,
            "fallback_used": False,
            "provider_attempts": [cfg.provider],
            "elapsed_ms": elapsed_ms,
            "prompt_eval_count": prompt_tokens,
            "eval_count": output_tokens,
            "hit_output_limit": finish_reason == "length",
            "done_reason": finish_reason,
            "node": node,
            "success": True,
        })

        return LLMResponse(
            content=content,
            provider=cfg.provider,
            model=cfg.model,
            latency_ms=elapsed_ms,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            raw_metadata={
                "request_id": request_id,
                "node": node,
                "elapsed_ms": elapsed_ms,
                "litellm_model": cfg.litellm_model,
                "hit_output_limit": finish_reason == "length",
            },
        )


def _map_litellm_error(exc: Exception, cfg: LLMExecutionConfig, timeout_s: float) -> LLMProviderError:
    """LiteLLM/provider exception -> the app's normalized LLM error taxonomy.
    Raw LiteLLM messages stay in logs; callers only ever see these types."""
    import litellm

    name = type(exc).__name__
    msg = f"{cfg.provider}/{cfg.model}: {exc}"

    if isinstance(exc, LLMProviderError):
        return exc
    if isinstance(exc, (litellm.Timeout, TimeoutError)):
        return LLMTimeoutError(f"{cfg.provider}/{cfg.model} timed out after {timeout_s}s.")
    if isinstance(exc, litellm.AuthenticationError):
        return LLMAuthenticationError(msg)
    if isinstance(exc, litellm.RateLimitError):
        return LLMRateLimitError(msg)
    if isinstance(exc, (litellm.ServiceUnavailableError, litellm.InternalServerError,
                        getattr(litellm, "BadGatewayError", ()))):  # type: ignore[arg-type]
        return LLMUnavailableError(msg)
    if isinstance(exc, litellm.APIConnectionError):
        return LLMConnectionError(msg)
    if isinstance(exc, getattr(litellm, "ContextWindowExceededError", ())):  # type: ignore[arg-type]
        return LLMInvalidResponseError(msg)
    if isinstance(exc, litellm.APIError):
        return LLMProviderError(msg)
    if "connect" in name.lower() or "connection" in str(exc).lower():
        return LLMConnectionError(msg)
    return LLMProviderError(msg)
