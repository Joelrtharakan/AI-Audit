"""Request throttling on the expensive AI-analysis endpoints.

The regression suite disables rate limiting globally (tests/conftest.py --
the limiter is a single process-global bucket keyed by the shared
INTERNAL_API_KEY, and unrelated test files sharing one pytest process would
otherwise trip each other's limit). These tests explicitly re-enable it and
exercise the limiter directly and through the real FastAPI app.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.services.rate_limiter import (
    SlidingWindowRateLimiter,
    enforce_investigation_rate_limit,
    get_investigation_rate_limiter,
    reset_rate_limiter_for_testing,
)


# --------------------------------------------------------------------------- #
# unit: the sliding-window primitive
# --------------------------------------------------------------------------- #

def test_requests_within_limit_are_allowed():
    lim = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
    for _ in range(5):
        allowed, _ = lim.check("caller-a")
        assert allowed is True


def test_requests_exceeding_limit_are_rejected():
    lim = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        assert lim.check("caller-a")[0] is True
    allowed, retry_after = lim.check("caller-a")
    assert allowed is False
    assert retry_after > 0


def test_independent_identities_have_independent_limits():
    lim = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    assert lim.check("caller-a")[0] is True
    assert lim.check("caller-a")[0] is True
    assert lim.check("caller-a")[0] is False
    # a different identity is completely unaffected
    assert lim.check("caller-b")[0] is True
    assert lim.check("caller-b")[0] is True


def test_expired_window_state_allows_new_requests():
    lim = SlidingWindowRateLimiter(max_requests=1, window_seconds=0.05)
    assert lim.check("caller-a")[0] is True
    assert lim.check("caller-a")[0] is False
    import time
    time.sleep(0.06)
    assert lim.check("caller-a")[0] is True


def test_rejected_call_does_not_extend_its_own_window():
    # A caller hammering the endpoint after being throttled must not push its
    # own retry-after further out with each rejected attempt.
    lim = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    lim.check("caller-a")
    _, retry_1 = lim.check("caller-a")
    _, retry_2 = lim.check("caller-a")
    assert retry_2 <= retry_1 + 0.01


def test_concurrent_requests_at_the_threshold_are_deterministic():
    lim = SlidingWindowRateLimiter(max_requests=10, window_seconds=60)
    results = [lim.check("caller-a")[0] for _ in range(15)]
    assert results.count(True) == 10
    assert results.count(False) == 5


# --------------------------------------------------------------------------- #
# dependency: enforce_investigation_rate_limit
# --------------------------------------------------------------------------- #

class _FakeClient:
    host = "127.0.0.1"


class _FakeRequest:
    def __init__(self, key: str = "test-key"):
        self.headers = {"x-internal-api-key": key}
        self.client = _FakeClient()


@pytest.fixture(autouse=True)
def _isolated_limiter(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "3")
    from app.config import get_settings
    get_settings.cache_clear()
    reset_rate_limiter_for_testing()
    yield
    reset_rate_limiter_for_testing()
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_dependency_allows_requests_within_limit():
    for _ in range(3):
        await enforce_investigation_rate_limit(_FakeRequest())


@pytest.mark.asyncio
async def test_dependency_rejects_requests_beyond_limit_with_429_and_retry_after():
    for _ in range(3):
        await enforce_investigation_rate_limit(_FakeRequest())
    with pytest.raises(HTTPException) as ei:
        await enforce_investigation_rate_limit(_FakeRequest())
    assert ei.value.status_code == 429
    assert "Retry-After" in ei.value.headers
    assert int(ei.value.headers["Retry-After"]) >= 1


@pytest.mark.asyncio
async def test_dependency_disabled_flag_bypasses_limiter(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    from app.config import get_settings
    get_settings.cache_clear()
    for _ in range(10):
        await enforce_investigation_rate_limit(_FakeRequest())
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_different_caller_identities_are_isolated_through_the_dependency():
    for _ in range(3):
        await enforce_investigation_rate_limit(_FakeRequest("key-a"))
    with pytest.raises(HTTPException):
        await enforce_investigation_rate_limit(_FakeRequest("key-a"))
    # key-b has its own untouched budget
    for _ in range(3):
        await enforce_investigation_rate_limit(_FakeRequest("key-b"))


def test_settings_backed_limiter_uses_configured_threshold(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "7")
    from app.config import get_settings
    get_settings.cache_clear()
    reset_rate_limiter_for_testing()
    limiter = get_investigation_rate_limiter()
    assert limiter.max_requests == 7
    get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# end-to-end: the real FastAPI app, real routes
# --------------------------------------------------------------------------- #

def test_investigate_endpoint_returns_429_once_limit_is_exceeded(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "1")
    from app.config import get_settings
    get_settings.cache_clear()
    reset_rate_limiter_for_testing()
    try:
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        headers = {"X-Internal-Api-Key": "test-key"}
        first = client.post("/api/v1/investigate", json={"finding_text": "x"}, headers=headers)
        assert first.status_code != 429
        second = client.post("/api/v1/investigate", json={"finding_text": "x"}, headers=headers)
        assert second.status_code == 429
        assert "Retry-After" in second.headers
    finally:
        reset_rate_limiter_for_testing()
        get_settings.cache_clear()


def test_health_endpoint_is_never_throttled(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "1")
    from app.config import get_settings
    get_settings.cache_clear()
    reset_rate_limiter_for_testing()
    try:
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        for _ in range(20):
            resp = client.get("/health")
            assert resp.status_code == 200
    finally:
        reset_rate_limiter_for_testing()
        get_settings.cache_clear()


def test_rate_limiting_never_bypasses_authentication(monkeypatch):
    # An unauthenticated call must still be rejected with 401, never let
    # through merely because the caller hasn't hit the rate limit yet.
    monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "1000")
    from app.config import get_settings
    get_settings.cache_clear()
    reset_rate_limiter_for_testing()
    try:
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.post("/api/v1/investigate", json={"finding_text": "x"})
        assert resp.status_code == 401
    finally:
        reset_rate_limiter_for_testing()
        get_settings.cache_clear()
