"""Single authoritative LLM provider router.

Owns the strict Groq -> OpenRouter -> Gemini -> (all failed) failover order.
Every LLM-calling node/service reaches this through get_llm_client() in
app/services/llm_client.py; nothing outside this module knows a provider's
identity, decides retry-vs-failover, or holds circuit-breaker state.

    Agent Node
        |
        v
    get_llm_client() -> LLMRouter (this module's singleton)
        |
        v
    Groq (circuit HEALTHY/HALF_OPEN?) --429/failure--> OpenRouter --429/failure--> Gemini --failure--> raise LLMError
        |                                    |                                        |
      success                              success                                  success
        |                                    |                                        |
        v                                    v                                        v
                          caller receives the completion string
                     (get_last_call_metadata() exposes which provider, and
                      whether/how many fallbacks were used, for the ONE
                      caller in this asyncio task -- see the ContextVar note
                      below)

This module is infrastructure only. It has no opinion about causal
reasoning, hypotheses, root causes, or anything else in the analytical
pipeline -- it returns a string (or raises LLMError) exactly like a single
provider client always did, so every existing `except LLMError` /
`except Exception` fallback in the node layer keeps working unmodified.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass
from enum import Enum

from app.services.llm.base import LLMProvider, LLMResponse
from app.services.llm_client import (
    AllLLMProvidersUnavailableError,
    LLMAuthenticationError,
    LLMClient,
    LLMConfigurationError,
    LLMError,
    LLMInvalidResponseError,
    LLMNetworkError,
    LLMRateLimitedError,
    LLMServerError,
    LLMTimeoutError,
    NoLLMProviderConfiguredError,
)

logger = logging.getLogger(__name__)

PROVIDER_ORDER: tuple[str, ...] = ("groq", "openrouter", "gemini")


class CircuitState(str, Enum):
    HEALTHY = "HEALTHY"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class ProviderCircuit:
    """Per-provider circuit-breaker state. One instance per provider name,
    shared process-wide (see _CIRCUITS below) so that once one node's LLM
    call opens a provider's circuit, every other node/request in this
    process skips that provider for the rest of the cooldown instead of
    independently re-discovering the same rate limit."""

    state: CircuitState = CircuitState.HEALTHY
    opened_at: float = 0.0
    cooldown_seconds: float = 0.0
    consecutive_failures: int = 0
    half_open_probe_in_flight: bool = False
    last_status: str = "NOT_CONFIGURED"
    retry_after: float | None = None

    def is_skippable(self, now: float) -> bool:
        """True if this provider should be skipped right now without even
        attempting a request."""
        if self.state != CircuitState.OPEN:
            return False
        return now < self.opened_at + self.cooldown_seconds

    def maybe_enter_half_open(self, now: float) -> bool:
        """If the cooldown has elapsed and no probe is already in flight,
        transition OPEN -> HALF_OPEN and claim the probe slot. Returns True
        if THIS call should perform the probe request."""
        if self.state == CircuitState.OPEN and now >= self.opened_at + self.cooldown_seconds:
            if self.half_open_probe_in_flight:
                return False
            self.state = CircuitState.HALF_OPEN
            self.half_open_probe_in_flight = True
            return True
        return self.state == CircuitState.HEALTHY

    def record_success(self) -> None:
        was_half_open = self.state == CircuitState.HALF_OPEN
        self.state = CircuitState.HEALTHY
        self.consecutive_failures = 0
        self.half_open_probe_in_flight = False
        self.cooldown_seconds = 0.0
        self.last_status = "SUCCESS"
        self.retry_after = None
        if was_half_open:
            logger.info("provider circuit recovered event=circuit_closed")

    def record_failure(self, cooldown_seconds: float, status: str = "UNAVAILABLE", retry_after: float | None = None) -> None:
        self.state = CircuitState.OPEN
        self.opened_at = time.monotonic()
        self.cooldown_seconds = cooldown_seconds
        self.consecutive_failures += 1
        self.half_open_probe_in_flight = False
        self.last_status = status
        self.retry_after = retry_after


_CIRCUITS: dict[str, ProviderCircuit] = {name: ProviderCircuit() for name in PROVIDER_ORDER}
_SEMAPHORES: dict[str, asyncio.Semaphore] = {}
from app.services.llm.call_metadata import (  # noqa: E402
    _last_call_metadata as _last_call_metadata,
    get_last_call_metadata as get_last_call_metadata,  # re-export -> shared ContextVar
)

# NOTE: this multi-provider circuit-breaker / failover router is NO LONGER in the
# investigation inference path (see app.services.llm.factory). It is retained
# only for its metadata helpers and any explicit out-of-band cloud-router use.
# `get_llm_provider` never returns an LLMRouter -- one investigation = one
# provider + one model, no fan-out, no fallback.


def _classify_cooldown(exc: Exception, default_cooldown: float, max_cooldown: float) -> tuple[float, str, float | None]:
    """Returns (cooldown_seconds, status_name, retry_after).

    When a provider returns HTTP 429 with Retry-After, respects the Retry-After header.
    Cooldown is max(configured_cooldown, retry_after) unless retry_after is explicitly specified,
    in which case retry_after is used directly (capped at max_cooldown).
    """
    if isinstance(exc, LLMRateLimitedError):
        retry_after = exc.retry_after
        if retry_after is not None and retry_after > 0:
            return min(retry_after, max_cooldown), "RATE_LIMITED", retry_after
        return min(default_cooldown, max_cooldown), "RATE_LIMITED", None
    if isinstance(exc, LLMTimeoutError):
        return min(default_cooldown, max_cooldown), "TIMEOUT", None
    if isinstance(exc, LLMNetworkError):
        return min(default_cooldown, max_cooldown), "CONNECTION_ERROR", None
    if isinstance(exc, LLMAuthenticationError):
        return min(default_cooldown, max_cooldown), "AUTHENTICATION_ERROR", None
    if isinstance(exc, LLMServerError):
        return min(default_cooldown, max_cooldown), "SERVER_ERROR", None
    if isinstance(exc, LLMInvalidResponseError):
        return min(default_cooldown, max_cooldown), "INVALID_RESPONSE", None
    return min(default_cooldown, max_cooldown), "UNAVAILABLE", None


def _provider_configured(name: str) -> bool:
    from app.config import get_settings

    settings = get_settings()
    if name == "groq":
        return bool(settings.groq_api_key)
    if name == "openrouter":
        return bool(settings.openrouter_api_key)
    if name == "gemini":
        return bool(settings.google_api_key)
    return False


def _build_provider_client(name: str) -> LLMClient:
    if name == "groq":
        from app.services.groq_client import GroqClient

        return GroqClient()
    if name == "openrouter":
        from app.services.openrouter_client import OpenRouterClient

        return OpenRouterClient()
    if name == "gemini":
        from app.services.gemini_client import GeminiClient

        return GeminiClient()
    raise ValueError(f"Unknown provider {name!r}")


def _get_semaphore(name: str) -> asyncio.Semaphore:
    sem = _SEMAPHORES.get(name)
    if sem is None:
        from app.config import get_settings

        limit = get_settings().llm_router_max_concurrency_per_provider
        sem = asyncio.Semaphore(max(1, limit))
        _SEMAPHORES[name] = sem
    return sem


class LLMRouter(LLMProvider):
    """Implements the LLMProvider interface. Internally tries each configured
    provider in PROVIDER_ORDER (Groq -> OpenRouter -> Gemini), opening that provider's circuit
    on failure and moving to the next, until one succeeds or all configured providers fail."""

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

        t_start = time.monotonic()
        content = await self.chat_completion(
            messages=messages,
            temperature=temperature,
            response_format_json=response_format == "json",
            max_tokens=max_output_tokens,
            num_ctx=num_ctx,
            node=node,
            **kwargs,
        )
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        meta = _last_call_metadata.get()
        return LLMResponse(
            content=content,
            provider=meta.get("provider_used") or "router",
            model="router",
            latency_ms=elapsed_ms,
            finish_reason="stop",
            raw_metadata=meta,
        )

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format_json: bool = False,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        node: str = "unknown",
        **kwargs,
    ) -> str:
        from app.config import get_settings

        settings = get_settings()

        # If llm_provider is set to "ollama", delegate directly to OllamaClient with 1 retry
        if settings.llm_provider == "ollama":
            from app.services.ollama_client import OllamaClient
            client = OllamaClient()
            try:
                result = await client.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    response_format_json=response_format_json,
                    max_tokens=max_tokens,
                    num_ctx=num_ctx,
                )
                # Preserve the provider's own timing/token metadata (Ollama
                # sets prompt_eval_count / eval_count / *_duration) -- merge,
                # don't overwrite (observability, spec Pass 35 §30).
                _prev = dict(_last_call_metadata.get() or {})
                _prev.update({
                    "provider_used": "ollama",
                    "fallback_used": False,
                    "provider_attempts": ["ollama"],
                })
                _last_call_metadata.set(_prev)
                return result
            except Exception as exc:
                logger.warning("Ollama call failed: %s; trying 1 fast retry...", exc)
                try:
                    result = await client.chat_completion(
                        messages=messages,
                        temperature=temperature,
                        response_format_json=response_format_json,
                        max_tokens=max_tokens,
                        num_ctx=num_ctx,
                    )
                    _last_call_metadata.set({
                        "provider_used": "ollama",
                        "fallback_used": True,
                        "provider_attempts": ["ollama", "ollama_retry"],
                    })
                    return result
                except Exception as retry_exc:
                    logger.error("Ollama fast retry failed: %s. Escalating to DEGRADED mode.", retry_exc)
                    _last_call_metadata.set({
                        "provider_used": None,
                        "fallback_used": False,
                        "provider_attempts": ["ollama", "ollama_retry"],
                    })
                    raise AllLLMProvidersUnavailableError(
                        "Ollama local inference server is unavailable.",
                        provider_statuses={"ollama": {"status": "UNAVAILABLE"}},
                    ) from retry_exc

        attempts: list[str] = []
        last_error: Exception | None = None
        provider_statuses: dict[str, dict] = {}
        configured_providers = [name for name in PROVIDER_ORDER if _provider_configured(name)]

        if not configured_providers:
            logger.error("event=no_provider_configured missing_api_keys=%s", list(PROVIDER_ORDER))
            _last_call_metadata.set({
                "provider_used": None,
                "fallback_used": False,
                "provider_attempts": [],
            })
            raise NoLLMProviderConfiguredError(
                f"No LLM provider is configured (missing API keys for {', '.join(PROVIDER_ORDER)})."
            )

        now = time.monotonic()

        for name in PROVIDER_ORDER:
            if not _provider_configured(name):
                logger.info("provider=%s event=unavailable reason=missing_api_key", name)
                provider_statuses[name] = {"status": "NOT_CONFIGURED"}
                continue

            circuit = _CIRCUITS[name]

            if circuit.state == CircuitState.OPEN:
                should_probe = circuit.maybe_enter_half_open(now)
                if not should_probe:
                    cooldown_rem = max(0.0, circuit.opened_at + circuit.cooldown_seconds - now)
                    logger.info(
                        "provider=%s event=skip circuit=%s cooldown_remaining=%.1fs",
                        name, circuit.state.value, cooldown_rem,
                    )
                    provider_statuses[name] = {
                        "status": circuit.last_status,
                        "cooldown_remaining": round(cooldown_rem, 1),
                        "retry_after": circuit.retry_after,
                    }
                    continue
                logger.info("provider=%s event=half_open_probe circuit=HALF_OPEN", name)

            attempts.append(name)
            sem = _get_semaphore(name)
            t_start = time.monotonic()
            next_p = PROVIDER_ORDER[PROVIDER_ORDER.index(name) + 1] if PROVIDER_ORDER.index(name) + 1 < len(PROVIDER_ORDER) else "DEGRADED"
            try:
                async with sem:
                    client = _build_provider_client(name)
                    result = await client.chat_completion(
                        messages,
                        temperature=temperature,
                        response_format_json=response_format_json,
                        max_tokens=max_tokens,
                    )
                if response_format_json:
                    from app.services.llm_json import parse_llm_json

                    parse_llm_json(result)
            except LLMError as exc:
                elapsed_ms = int((time.monotonic() - t_start) * 1000)
                cooldown, status_name, retry_after = _classify_cooldown(
                    exc, settings.llm_router_default_cooldown_seconds, settings.llm_router_max_cooldown_seconds
                )
                circuit.record_failure(cooldown, status=status_name, retry_after=retry_after)
                logger.warning(
                    "provider=%s event=failure failure_type=%s elapsed_ms=%d retry_after=%s next_provider=%s circuit=OPEN cooldown=%.1fs",
                    name, status_name, elapsed_ms, retry_after, next_p, cooldown,
                )
                provider_statuses[name] = {
                    "status": status_name,
                    "retry_after": retry_after,
                    "cooldown": cooldown,
                    "elapsed_ms": elapsed_ms,
                }
                last_error = exc
                continue
            except (ValueError, TypeError) as exc:
                elapsed_ms = int((time.monotonic() - t_start) * 1000)
                cooldown = settings.llm_router_default_cooldown_seconds
                circuit.record_failure(
                    min(cooldown, settings.llm_router_max_cooldown_seconds),
                    status="INVALID_RESPONSE",
                )
                logger.warning(
                    "provider=%s event=failure failure_type=INVALID_RESPONSE elapsed_ms=%d next_provider=%s circuit=OPEN cooldown=%.1fs",
                    name, elapsed_ms, next_p, cooldown,
                )
                provider_statuses[name] = {"status": "INVALID_RESPONSE", "elapsed_ms": elapsed_ms}
                last_error = LLMError(f"{name} returned malformed JSON")
                continue

            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            circuit.record_success()
            provider_statuses[name] = {"status": "SUCCESS", "elapsed_ms": elapsed_ms}
            fallback_used = name != PROVIDER_ORDER[0] or len(attempts) > 1
            logger.info(
                "provider=%s event=success fallback_used=%s elapsed_ms=%d attempts=%s",
                name, fallback_used, elapsed_ms, attempts,
            )
            _last_call_metadata.set({
                "provider_used": name,
                "fallback_used": fallback_used,
                "provider_attempts": list(attempts),
                "elapsed_ms": elapsed_ms,
            })
            return result

        logger.error(
            "event=all_providers_failed providers_attempted=%s provider_statuses=%s analysis_mode=DEGRADED",
            attempts, provider_statuses,
        )
        _last_call_metadata.set({
            "provider_used": None,
            "fallback_used": True,
            "provider_attempts": list(attempts),
        })
        raise AllLLMProvidersUnavailableError(
            f"All configured LLM providers are currently unavailable (attempted: {attempts or configured_providers}).",
            provider_statuses=provider_statuses,
        ) from last_error


_router_singleton: LLMRouter | None = None


def get_llm_router() -> LLMRouter:
    global _router_singleton
    if _router_singleton is None:
        _router_singleton = LLMRouter()
    return _router_singleton


def reset_circuits_for_testing() -> None:
    """Test-only helper: resets every provider's circuit to HEALTHY. Tests
    that exercise circuit-breaker behavior must call this in setup/teardown
    since _CIRCUITS is a process-wide singleton and state would otherwise
    leak between test cases."""
    for name in PROVIDER_ORDER:
        _CIRCUITS[name] = ProviderCircuit()
    _last_call_metadata.set({})
