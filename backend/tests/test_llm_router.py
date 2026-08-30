"""Unit tests for app/services/llm_router.py — the single authoritative
provider router (Groq -> OpenRouter -> Gemini -> raise LLMError).

Covers the 10 required scenarios from the multi-provider failover spec.
Providers are mocked at the `_build_provider_client` factory boundary so
these tests are deterministic and don't depend on live rate limits/quotas;
a couple of tests below (marked accordingly in comments) were also verified
against a real, unmocked provider during development -- see the final
report for what was actually observed live.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.skip(reason="Multi-provider fan-out/failover router removed from the inference path: one investigation = one provider + one model via the LiteLLM boundary. See app/services/llm/execution.py + providers/litellm_provider.py.")

from app.services.gemini_client import GeminiConfigurationError
from app.services.groq_client import GroqRateLimitedError, GroqServerError
from app.services.llm_client import LLMError
from app.services.llm_router import (
    CircuitState,
    _CIRCUITS,
    get_last_call_metadata,
    get_llm_router,
    reset_circuits_for_testing,
)
from app.services.openrouter_client import RateLimitedError as OpenRouterRateLimitedError


@pytest.fixture(autouse=True)
def _reset_router_state():
    reset_circuits_for_testing()
    from app.config import get_settings
    settings = get_settings()
    original_provider = settings.llm_provider
    settings.llm_provider = "groq"
    yield
    settings.llm_provider = original_provider
    reset_circuits_for_testing()


def _mock_provider(result=None, error=None):
    m = AsyncMock()
    if error is not None:
        m.chat_completion.side_effect = error
    else:
        m.chat_completion.return_value = result
    return m


def _patch_providers(**providers):
    """providers: {"groq": mock_client, "openrouter": mock_client, ...} --
    any provider not passed defaults to a healthy mock returning "OK"."""
    def build(name):
        if name in providers:
            return providers[name]
        return _mock_provider(result=f"{name.upper()}_DEFAULT_OK")
    return patch("app.services.llm_router._build_provider_client", side_effect=build)


def _configured(**keys):
    """Context manager patching _provider_configured so tests don't depend
    on real environment API keys being set."""
    def configured(name):
        return keys.get(name, False)
    return patch("app.services.llm_router._provider_configured", side_effect=configured)


@pytest.mark.asyncio
async def test_1_groq_succeeds():
    router = get_llm_router()
    with _configured(groq=True, openrouter=True, gemini=True), \
         _patch_providers(groq=_mock_provider(result="GROQ OK")):
        result = await router.chat_completion([{"role": "user", "content": "hi"}])
    assert result == "GROQ OK"
    meta = get_last_call_metadata()
    assert meta["provider_used"] == "groq"
    assert meta["fallback_used"] is False


@pytest.mark.asyncio
async def test_2_groq_429_opens_circuit_and_groq_not_called_again():
    router = get_llm_router()
    groq = _mock_provider(error=GroqRateLimitedError("429", retry_after=30.0))
    openrouter = _mock_provider(result="OPENROUTER OK")
    with _configured(groq=True, openrouter=True, gemini=True), \
         _patch_providers(groq=groq, openrouter=openrouter):
        result = await router.chat_completion([{"role": "user", "content": "hi"}])
        assert result == "OPENROUTER OK"
        assert _CIRCUITS["groq"].state == CircuitState.OPEN

        groq.chat_completion.reset_mock()
        result2 = await router.chat_completion([{"role": "user", "content": "hi again"}])
        assert result2 == "OPENROUTER OK"
        assert groq.chat_completion.called is False, "Groq must not be called again during cooldown"


@pytest.mark.asyncio
async def test_3_groq_429_openrouter_succeeds_final_provider_is_openrouter():
    router = get_llm_router()
    groq = _mock_provider(error=GroqRateLimitedError("429", retry_after=10.0))
    openrouter = _mock_provider(result="OPENROUTER OK")
    with _configured(groq=True, openrouter=True, gemini=True), \
         _patch_providers(groq=groq, openrouter=openrouter):
        result = await router.chat_completion([{"role": "user", "content": "hi"}])
    assert result == "OPENROUTER OK"
    meta = get_last_call_metadata()
    assert meta["provider_used"] == "openrouter"
    assert meta["fallback_used"] is True
    assert meta["provider_attempts"] == ["groq", "openrouter"]


@pytest.mark.asyncio
async def test_4_groq_429_openrouter_429_gemini_succeeds():
    router = get_llm_router()
    groq = _mock_provider(error=GroqRateLimitedError("429", retry_after=10.0))
    openrouter = _mock_provider(error=OpenRouterRateLimitedError("429", retry_after=10.0))
    gemini = _mock_provider(result="GEMINI OK")
    with _configured(groq=True, openrouter=True, gemini=True), \
         _patch_providers(groq=groq, openrouter=openrouter, gemini=gemini):
        result = await router.chat_completion([{"role": "user", "content": "hi"}])
    assert result == "GEMINI OK"
    meta = get_last_call_metadata()
    assert meta["provider_used"] == "gemini"
    assert meta["fallback_used"] is True
    assert meta["provider_attempts"] == ["groq", "openrouter", "gemini"]


@pytest.mark.asyncio
async def test_5_all_providers_fail_raises_llm_error_no_fabrication():
    """The router itself must raise LLMError (not fabricate a response) --
    the "no fabricated root cause/CAPA/impact" guarantee is enforced by
    core_synthesis's degraded-mode fallback, which is exercised separately
    in test_analysis_mode_degraded.py; here we verify the router's own
    contract: total failure is a clean, typed exception, nothing silently
    swallowed or invented."""
    router = get_llm_router()
    groq = _mock_provider(error=GroqServerError("500"))
    openrouter = _mock_provider(error=OpenRouterRateLimitedError("429", retry_after=5.0))
    gemini = _mock_provider(error=GeminiConfigurationError("no key"))
    with _configured(groq=True, openrouter=True, gemini=True), \
         _patch_providers(groq=groq, openrouter=openrouter, gemini=gemini):
        with pytest.raises(LLMError):
            await router.chat_completion([{"role": "user", "content": "hi"}])
    meta = get_last_call_metadata()
    assert meta["provider_used"] is None
    assert meta["provider_attempts"] == ["groq", "openrouter", "gemini"]


@pytest.mark.asyncio
async def test_6_huge_retry_after_does_not_block_the_request():
    router = get_llm_router()
    groq = _mock_provider(error=GroqRateLimitedError("429", retry_after=2380.0))
    openrouter = _mock_provider(result="OPENROUTER OK")
    with _configured(groq=True, openrouter=True, gemini=True), \
         _patch_providers(groq=groq, openrouter=openrouter):
        t0 = time.monotonic()
        result = await asyncio.wait_for(
            router.chat_completion([{"role": "user", "content": "hi"}]), timeout=5.0
        )
        elapsed = time.monotonic() - t0
    assert result == "OPENROUTER OK"
    assert elapsed < 2.0, f"request must not block for anywhere near 2380s (took {elapsed:.2f}s)"
    # Cooldown is capped, not the raw 2380s Retry-After.
    from app.config import get_settings
    assert _CIRCUITS["groq"].cooldown_seconds <= get_settings().llm_router_max_cooldown_seconds


@pytest.mark.asyncio
async def test_7_concurrent_requests_after_groq_open_skip_groq_no_thundering_herd():
    router = get_llm_router()
    groq = _mock_provider(error=GroqRateLimitedError("429", retry_after=30.0))
    openrouter = _mock_provider(result="OPENROUTER OK")
    with _configured(groq=True, openrouter=True, gemini=True), \
         _patch_providers(groq=groq, openrouter=openrouter):
        # First request opens Groq's circuit.
        await router.chat_completion([{"role": "user", "content": "1"}])
        groq.chat_completion.reset_mock()

        # Several "simultaneous" requests must all skip Groq entirely.
        results = await asyncio.gather(*[
            router.chat_completion([{"role": "user", "content": str(i)}]) for i in range(5)
        ])
    assert all(r == "OPENROUTER OK" for r in results)
    assert groq.chat_completion.call_count == 0, "no request should have reached Groq during its cooldown"


@pytest.mark.asyncio
async def test_8_groq_recovers_open_to_half_open_to_healthy():
    router = get_llm_router()
    groq = _mock_provider(error=GroqRateLimitedError("429", retry_after=0.05))
    openrouter = _mock_provider(result="OPENROUTER OK")
    with _configured(groq=True, openrouter=True, gemini=True), \
         _patch_providers(groq=groq, openrouter=openrouter):
        await router.chat_completion([{"role": "user", "content": "1"}])
        assert _CIRCUITS["groq"].state == CircuitState.OPEN

        # Cooldown has a deliberate 1.0s floor (prevents rapid circuit
        # flapping on a provider reporting a near-zero Retry-After).
        await asyncio.sleep(1.1)

        # Groq now succeeds on the recovery probe.
        groq.chat_completion.side_effect = None
        groq.chat_completion.return_value = "GROQ RECOVERED"
        result = await router.chat_completion([{"role": "user", "content": "2"}])

    assert result == "GROQ RECOVERED"
    assert _CIRCUITS["groq"].state == CircuitState.HEALTHY


@pytest.mark.asyncio
async def test_9_openrouter_fails_after_groq_unavailable_gemini_receives_request():
    router = get_llm_router()
    groq = _mock_provider(error=GroqRateLimitedError("429", retry_after=10.0))
    openrouter = _mock_provider(error=OpenRouterRateLimitedError("429", retry_after=10.0))
    gemini = _mock_provider(result="GEMINI OK")
    with _configured(groq=True, openrouter=True, gemini=True), \
         _patch_providers(groq=groq, openrouter=openrouter, gemini=gemini):
        result = await router.chat_completion([{"role": "user", "content": "hi"}])
    assert result == "GEMINI OK"
    assert gemini.chat_completion.called


@pytest.mark.asyncio
async def test_10_all_providers_recover_routing_returns_to_groq():
    router = get_llm_router()
    groq = _mock_provider(error=GroqRateLimitedError("429", retry_after=0.05))
    openrouter = _mock_provider(result="OPENROUTER OK")
    with _configured(groq=True, openrouter=True, gemini=True), \
         _patch_providers(groq=groq, openrouter=openrouter):
        await router.chat_completion([{"role": "user", "content": "1"}])
        assert get_last_call_metadata()["provider_used"] == "openrouter"

        await asyncio.sleep(1.1)
        groq.chat_completion.side_effect = None
        groq.chat_completion.return_value = "GROQ OK"

        result = await router.chat_completion([{"role": "user", "content": "2"}])
    assert result == "GROQ OK"
    assert get_last_call_metadata()["provider_used"] == "groq", "Groq is priority #1 and must be preferred once healthy again"


# ---------------------------------------------------------------------------
# Missing API key handling (Section 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_groq_key_skips_straight_to_openrouter():
    router = get_llm_router()
    openrouter = _mock_provider(result="OPENROUTER OK")
    with _configured(groq=False, openrouter=True, gemini=True), \
         _patch_providers(openrouter=openrouter):
        result = await router.chat_completion([{"role": "user", "content": "hi"}])
    assert result == "OPENROUTER OK"
    assert get_last_call_metadata()["provider_attempts"] == ["openrouter"]


@pytest.mark.asyncio
async def test_no_providers_configured_raises_without_crashing():
    router = get_llm_router()
    with _configured(groq=False, openrouter=False, gemini=False):
        with pytest.raises(LLMError):
            await router.chat_completion([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_authentication_error_does_not_retry_same_provider_within_a_call():
    from app.services.groq_client import GroqAuthenticationError

    router = get_llm_router()
    groq = _mock_provider(error=GroqAuthenticationError("bad key"))
    openrouter = _mock_provider(result="OPENROUTER OK")
    with _configured(groq=True, openrouter=True, gemini=True), \
         _patch_providers(groq=groq, openrouter=openrouter):
        result = await router.chat_completion([{"role": "user", "content": "hi"}])
    assert result == "OPENROUTER OK"
    assert groq.chat_completion.call_count == 1


@pytest.mark.asyncio
async def test_empty_completion_from_provider_triggers_failover():
    """An OpenRouter-style empty/None completion (2xx but useless content)
    must be treated as a provider failure by the client, not silently
    returned as a successful empty string -- observed live during
    development against a real free-tier OpenRouter model."""
    from app.services.openrouter_client import OpenRouterInvalidResponseError

    router = get_llm_router()
    groq = _mock_provider(error=GroqRateLimitedError("429", retry_after=5.0))
    openrouter = _mock_provider(error=OpenRouterInvalidResponseError("empty completion"))
    gemini = _mock_provider(result="GEMINI OK")
    with _configured(groq=True, openrouter=True, gemini=True), \
         _patch_providers(groq=groq, openrouter=openrouter, gemini=gemini):
        result = await router.chat_completion([{"role": "user", "content": "hi"}])
    assert result == "GEMINI OK"
