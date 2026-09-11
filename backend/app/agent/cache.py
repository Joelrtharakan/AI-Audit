"""Bounded in-memory analysis cache keyed by a deterministic hash of finding context.

The key includes model + prompt_version so that swapping models or editing a
prompt template can never silently serve a stale result generated under a
different configuration.

Two safety properties are enforced *here*, at the cache boundary, so a caller
cannot bypass them:

  1. Bounded size + TTL. At most ``analysis_cache_max_entries`` live entries;
     each expires after ``analysis_cache_ttl_seconds``. Eviction is LRU. This
     prevents unbounded memory growth in a long-running process.

  2. Failure/degraded results are never stored. A DEGRADED analysis, a
     deterministic-fallback that only ran because the semantic LLM was
     unavailable, or a transient provider/timeout failure in a sub-analysis
     must not be replayed for every subsequent request for the same finding
     once the provider recovers. ``set_cached_analysis`` inspects the payload
     and silently declines to store a non-cacheable result.

This cache is per-process and is NOT shared across uvicorn workers.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

from app.config import get_settings

# Sub-analysis semantic-status values that indicate a transient LLM failure
# (a healthy retry could produce a materially better result) rather than an
# honest "cannot be assessed from the evidence" conclusion.
_TRANSIENT_LLM_STATUSES = frozenset({"LLM_UNAVAILABLE", "LLM_TIMEOUT"})

_LOCK = threading.Lock()
# key -> (stored_at_monotonic, data)
_CACHE: "OrderedDict[str, tuple[float, dict[str, Any]]]" = OrderedDict()


def compute_cache_key(
    finding_text: str,
    department: str = "",
    standard: str = "",
    model: str = "",
    prompt_version: str = "",
) -> str:
    payload = (
        f"{finding_text.strip().lower()}|{department.strip().lower()}|{standard.strip().lower()}"
        f"|{model.strip().lower()}|{prompt_version.strip().lower()}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_cacheable(data: dict[str, Any]) -> tuple[bool, str]:
    """Return (cacheable, reason). A False result is never stored."""
    report = data.get("report")
    if not isinstance(report, dict):
        return False, "NO_REPORT"

    if report.get("analysis_mode") == "DEGRADED":
        return False, "DEGRADED"

    # The canonical semantic LLM was attempted and failed; the deterministic
    # floor produced this interpretation. Transient -- do not pin it.
    if report.get("semantic_mode") == "DETERMINISTIC_FALLBACK":
        return False, "DETERMINISTIC_FALLBACK"

    remediation = report.get("remediation_cost")
    if isinstance(remediation, dict) and remediation.get("remediation_semantic_status") in _TRANSIENT_LLM_STATUSES:
        return False, f"REMEDIATION_{remediation.get('remediation_semantic_status')}"

    financial = report.get("financial_analysis")
    if isinstance(financial, dict) and financial.get("financial_semantic_status") in _TRANSIENT_LLM_STATUSES:
        return False, f"FINANCIAL_{financial.get('financial_semantic_status')}"

    return True, "OK"


def _prune_expired_locked(now: float, ttl: float) -> None:
    expired = [k for k, (stored_at, _) in _CACHE.items() if now - stored_at >= ttl]
    for k in expired:
        del _CACHE[k]


def get_cached_analysis(cache_key: str) -> Optional[dict[str, Any]]:
    settings = get_settings()
    ttl = float(settings.analysis_cache_ttl_seconds)
    now = time.monotonic()
    with _LOCK:
        entry = _CACHE.get(cache_key)
        if entry is None:
            return None
        stored_at, data = entry
        if now - stored_at >= ttl:
            del _CACHE[cache_key]
            return None
        _CACHE.move_to_end(cache_key)  # LRU touch
        return data


def set_cached_analysis(cache_key: str, data: dict[str, Any]) -> bool:
    """Store an analysis result. Returns True if stored, False if the result
    was declined (non-cacheable state) -- see ``_is_cacheable``."""
    cacheable, _reason = _is_cacheable(data)
    if not cacheable:
        return False

    settings = get_settings()
    max_entries = max(1, int(settings.analysis_cache_max_entries))
    ttl = float(settings.analysis_cache_ttl_seconds)
    now = time.monotonic()

    with _LOCK:
        _prune_expired_locked(now, ttl)
        _CACHE[cache_key] = (now, data)
        _CACHE.move_to_end(cache_key)
        while len(_CACHE) > max_entries:
            _CACHE.popitem(last=False)  # evict least-recently-used
    return True


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def cache_stats() -> dict[str, int]:
    with _LOCK:
        return {"entries": len(_CACHE)}
