"""Request throttling for the expensive AI-analysis endpoints.

An in-process, per-caller SLIDING-WINDOW limiter. Deliberately dependency-free
(no new external package) and deliberately narrow in scope: it protects a
single worker process from being flooded, nothing more.

PRODUCTION CAVEAT (explicit, not hidden): this counter lives in one process's
memory. A multi-worker (uvicorn --workers N) or multi-instance deployment does
NOT share it -- each process/instance enforces its own independent limit, so
the effective aggregate limit is `rate_limit_requests_per_minute * instances`.
A deployment that needs one true shared limit across workers/instances MUST
back this with a shared store (e.g. Redis) instead. That is an infrastructure
decision for the deployment, not something this module can resolve on its
own -- documented in docs/RATE_LIMITING.md.

This module is not authentication or authorization -- it runs AFTER
`require_internal_api_key` on the routes that use it, and never bypasses that
gate. It is deliberately excluded from `/health` (load-balancer probes must
never be throttled) and from the provider-switch endpoint (a rare, already
API-key-gated operation with no need for throughput protection).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status


class SlidingWindowRateLimiter:
    """Per-key sliding-window request counter. Thread-safety note: FastAPI's
    default sync-dependency execution and the GIL make simple deque
    append/popleft safe enough for this best-effort, single-process
    protection; it is not a substitute for a distributed limiter under real
    concurrency guarantees."""

    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, key: str, now: float) -> deque[float]:
        window = self._hits[key]
        cutoff = now - self.window_seconds
        while window and window[0] < cutoff:
            window.popleft()
        return window

    def check(self, key: str) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds). Records the hit only when
        allowed, so a rejected caller retrying immediately does not itself
        extend its own penalty window."""
        now = time.monotonic()
        window = self._prune(key, now)
        if len(window) >= self.max_requests:
            retry_after = max(0.0, self.window_seconds - (now - window[0]))
            return False, retry_after
        window.append(now)
        return True, 0.0

    def reset(self) -> None:
        """Test-only: clear all counters."""
        self._hits.clear()


_investigation_limiter: SlidingWindowRateLimiter | None = None


def get_investigation_rate_limiter() -> SlidingWindowRateLimiter:
    global _investigation_limiter
    if _investigation_limiter is None:
        from app.config import get_settings
        s = get_settings()
        _investigation_limiter = SlidingWindowRateLimiter(max(1, s.rate_limit_requests_per_minute))
    return _investigation_limiter


def reset_rate_limiter_for_testing() -> None:
    global _investigation_limiter
    _investigation_limiter = None


def _caller_key(request: Request) -> str:
    """Best-effort caller identity: the presented internal API key (a real
    per-caller identity would require ASP.NET to mint distinguishable service
    credentials, which this repository does not define -- see
    docs/RATE_LIMITING.md) falling back to client host."""
    key = request.headers.get("x-internal-api-key") or (request.client.host if request.client else "unknown")
    return key


async def enforce_investigation_rate_limit(request: Request) -> None:
    from app.config import get_settings
    if not get_settings().rate_limit_enabled:
        return
    limiter = get_investigation_rate_limiter()
    allowed, retry_after = limiter.check(_caller_key(request))
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many analysis requests. Please retry after the indicated interval.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
