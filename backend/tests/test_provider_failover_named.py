"""Exact-named provider-failover tests requested for the deliverable
checklist. These are deterministic router-level tests (see
tests/test_llm_router.py for the full 18-test suite and
tests/test_llm_router_live_graph.py for the get_agent_graph()-level
coverage) — kept separate under these specific names for grep-ability.
"""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.skip(reason="Provider failover removed: one investigation = one provider + one model via the LiteLLM boundary (no fan-out, no fallback). See app/services/llm/execution.py.")
from unittest.mock import AsyncMock, patch

from app.services.gemini_client import GeminiServerError
from app.services.groq_client import GroqRateLimitedError, GroqServerError
from app.services.llm_client import LLMError
from app.services.llm_router import CircuitState, _CIRCUITS, get_last_call_metadata, get_llm_router, reset_circuits_for_testing
from app.services.openrouter_client import OpenRouterInvalidResponseError
from app.services.openrouter_client import RateLimitedError as OpenRouterRateLimitedError


@pytest.fixture(autouse=True)
def _reset_router_state():
    reset_circuits_for_testing()
    from app.config import get_settings
    settings = get_settings()
    orig = settings.llm_provider
    settings.llm_provider = "groq"
    yield
    settings.llm_provider = orig
    reset_circuits_for_testing()


def _mock_provider(result=None, error=None):
    m = AsyncMock()
    if error is not None:
        m.chat_completion.side_effect = error
    else:
        m.chat_completion.return_value = result
    return m


def _patch(**providers):
    def build(name):
        return providers.get(name) or _mock_provider(result=f"{name.upper()}_DEFAULT_OK")
    return patch("app.services.llm_router._build_provider_client", side_effect=build)


def _configured(**keys):
    return patch("app.services.llm_router._provider_configured", side_effect=lambda n: keys.get(n, False))


@pytest.mark.asyncio
async def test_groq_to_openrouter_failover():
    router = get_llm_router()
    groq = _mock_provider(error=GroqRateLimitedError("429", retry_after=45.0))
    openrouter = _mock_provider(result='{"ok": true}')
    with _configured(groq=True, openrouter=True, gemini=True), _patch(groq=groq, openrouter=openrouter):
        result = await router.chat_completion([{"role": "user", "content": "hi"}], response_format_json=True)
    meta = get_last_call_metadata()
    assert result == '{"ok": true}'
    assert meta["provider_used"] == "openrouter"
    assert meta["fallback_used"] is True
    assert meta["provider_attempts"] == ["groq", "openrouter"]
    assert _CIRCUITS["groq"].state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_openrouter_to_gemini_failover():
    """OpenRouter's 200-but-empty-content failure mode specifically."""
    router = get_llm_router()
    openrouter = _mock_provider(error=OpenRouterInvalidResponseError("empty completion"))
    gemini = _mock_provider(result='{"ok": true}')
    with _configured(groq=False, openrouter=True, gemini=True), _patch(openrouter=openrouter, gemini=gemini):
        result = await router.chat_completion([{"role": "user", "content": "hi"}], response_format_json=True)
    meta = get_last_call_metadata()
    assert result == '{"ok": true}'
    assert meta["provider_used"] == "gemini"
    assert meta["fallback_used"] is True


@pytest.mark.asyncio
async def test_groq_openrouter_gemini_chain():
    """The critical chain: Groq 429 -> OpenRouter empty content -> Gemini
    succeeds. This is the exact scenario named as most important to prove:
    provider_attempts must show all three, in order, and the final result
    must be analysis_mode-equivalent to a normal success (LLM, not
    DEGRADED) -- verified at the router level here; the equivalent outcome
    at the full-report level is verified in
    test_llm_router_live_graph.py::test_full_graph_surfaces_provider_metadata_on_report
    and via a real (unmocked) Gemini call in
    test_llm_router_live_graph.py::test_live_groq_openrouter_unavailable_gemini_real_success."""
    router = get_llm_router()
    groq = _mock_provider(error=GroqRateLimitedError("429", retry_after=1463.0))
    openrouter = _mock_provider(error=OpenRouterInvalidResponseError("OpenRouter returned an empty completion."))
    gemini = _mock_provider(result='{"status": "ok"}')
    with _configured(groq=True, openrouter=True, gemini=True), _patch(groq=groq, openrouter=openrouter, gemini=gemini):
        result = await router.chat_completion([{"role": "user", "content": "hi"}], response_format_json=True)
    meta = get_last_call_metadata()
    assert result == '{"status": "ok"}'
    assert meta["provider_attempts"] == ["groq", "openrouter", "gemini"]
    assert meta["provider_used"] == "gemini"
    assert meta["fallback_used"] is True
    # Cooldown must be capped, never the raw 1463s Retry-After.
    from app.config import get_settings
    assert _CIRCUITS["groq"].cooldown_seconds <= get_settings().llm_router_max_cooldown_seconds


@pytest.mark.asyncio
async def test_all_providers_degraded():
    router = get_llm_router()
    groq = _mock_provider(error=GroqServerError("500"))
    openrouter = _mock_provider(error=OpenRouterRateLimitedError("429", retry_after=10.0))
    gemini = _mock_provider(error=GeminiServerError("503"))
    with _configured(groq=True, openrouter=True, gemini=True), _patch(groq=groq, openrouter=openrouter, gemini=gemini):
        with pytest.raises(LLMError):
            await router.chat_completion([{"role": "user", "content": "hi"}], response_format_json=True)
    meta = get_last_call_metadata()
    assert meta["provider_used"] is None
    assert meta["fallback_used"] is True
    assert meta["provider_attempts"] == ["groq", "openrouter", "gemini"]
    # The router's own contract stops here (raises LLMError); the "no
    # fabricated root cause/CAPA/impact/contributing factors" guarantee
    # this failure must trigger downstream is verified in
    # test_analysis_mode_degraded.py, which exercises core_synthesis_node's
    # actual degraded-mode fallback that catches this exact exception.


@pytest.mark.asyncio
async def test_retry_after_does_not_block():
    router = get_llm_router()
    groq = _mock_provider(error=GroqRateLimitedError("429", retry_after=1463.0))
    openrouter = _mock_provider(result="OK")
    with _configured(groq=True, openrouter=True, gemini=True), _patch(groq=groq, openrouter=openrouter):
        t0 = time.monotonic()
        result = await router.chat_completion([{"role": "user", "content": "hi"}])
        elapsed = time.monotonic() - t0
    assert result == "OK"
    assert elapsed < 2.0, f"must not block anywhere near 1463s (took {elapsed:.2f}s)"


@pytest.mark.asyncio
async def test_circuit_breaker_recovery():
    router = get_llm_router()
    groq = _mock_provider(error=GroqRateLimitedError("429", retry_after=0.05))
    openrouter = _mock_provider(result="OPENROUTER OK")
    with _configured(groq=True, openrouter=True, gemini=True), _patch(groq=groq, openrouter=openrouter):
        await router.chat_completion([{"role": "user", "content": "1"}])
        assert _CIRCUITS["groq"].state == CircuitState.OPEN

        import asyncio
        await asyncio.sleep(1.1)  # clear the 1.0s cooldown floor

        groq.chat_completion.side_effect = None
        groq.chat_completion.return_value = "GROQ RECOVERED"
        result = await router.chat_completion([{"role": "user", "content": "2"}])
    assert result == "GROQ RECOVERED"
    assert _CIRCUITS["groq"].state == CircuitState.HEALTHY
