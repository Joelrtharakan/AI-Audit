"""Phase 2 + 3 -- bounded analysis cache and enforced degraded/failure exclusion.

The cache must be bounded (size + TTL + LRU eviction) and must NEVER store a
DEGRADED / deterministic-fallback / transient-LLM-failure result -- enforced at
the cache boundary, not merely by callers.
"""
from __future__ import annotations

import threading

import pytest

from app.agent import cache as cache_mod
from app.config import get_settings


@pytest.fixture(autouse=True)
def _fresh_cache():
    cache_mod.clear_cache()
    yield
    cache_mod.clear_cache()


def _ok(tag: str = "x") -> dict:
    return {"report": {"analysis_mode": "LLM", "semantic_mode": "CANONICAL_LLM", "tag": tag}}


# --------------------------------------------------------------------------- #
# Phase 2: bounded behavior
# --------------------------------------------------------------------------- #

def test_cache_hit_and_miss():
    assert cache_mod.get_cached_analysis("k1") is None            # miss
    assert cache_mod.set_cached_analysis("k1", _ok("v1")) is True
    assert cache_mod.get_cached_analysis("k1")["report"]["tag"] == "v1"  # hit
    assert cache_mod.get_cached_analysis("k2") is None            # miss


def test_ttl_expiration(monkeypatch):
    monkeypatch.setattr(get_settings(), "analysis_cache_ttl_seconds", 100.0)

    t = [1000.0]
    monkeypatch.setattr(cache_mod.time, "monotonic", lambda: t[0])

    cache_mod.set_cached_analysis("k", _ok())
    t[0] = 1099.0
    assert cache_mod.get_cached_analysis("k") is not None   # still fresh
    t[0] = 1101.0
    assert cache_mod.get_cached_analysis("k") is None       # expired
    assert cache_mod.cache_stats()["entries"] == 0          # expired entry dropped


def test_max_size_and_lru_eviction(monkeypatch):
    monkeypatch.setattr(get_settings(), "analysis_cache_max_entries", 3)

    for k in ("a", "b", "c"):
        cache_mod.set_cached_analysis(k, _ok(k))
    assert cache_mod.cache_stats()["entries"] == 3

    # touch "a" so "b" becomes least-recently-used
    cache_mod.get_cached_analysis("a")
    cache_mod.set_cached_analysis("d", _ok("d"))  # forces one eviction

    assert cache_mod.cache_stats()["entries"] == 3
    assert cache_mod.get_cached_analysis("b") is None      # evicted (LRU)
    assert cache_mod.get_cached_analysis("a") is not None
    assert cache_mod.get_cached_analysis("c") is not None
    assert cache_mod.get_cached_analysis("d") is not None


def test_insertion_after_eviction(monkeypatch):
    monkeypatch.setattr(get_settings(), "analysis_cache_max_entries", 2)
    cache_mod.set_cached_analysis("a", _ok("a"))
    cache_mod.set_cached_analysis("b", _ok("b"))
    cache_mod.set_cached_analysis("c", _ok("c"))  # evicts "a"
    assert cache_mod.get_cached_analysis("a") is None
    cache_mod.set_cached_analysis("a", _ok("a2"))  # re-insert
    assert cache_mod.get_cached_analysis("a")["report"]["tag"] == "a2"
    assert cache_mod.cache_stats()["entries"] == 2


def test_concurrent_access_does_not_corrupt(monkeypatch):
    monkeypatch.setattr(get_settings(), "analysis_cache_max_entries", 50)

    def worker(base: int):
        for i in range(200):
            key = f"k{(base + i) % 80}"
            cache_mod.set_cached_analysis(key, _ok(key))
            cache_mod.get_cached_analysis(key)

    threads = [threading.Thread(target=worker, args=(b,)) for b in range(0, 400, 50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert cache_mod.cache_stats()["entries"] <= 50   # bound never exceeded


# --------------------------------------------------------------------------- #
# Phase 3: degraded / failure exclusion at the boundary
# --------------------------------------------------------------------------- #

def test_degraded_result_is_not_cached():
    data = {"report": {"analysis_mode": "DEGRADED", "semantic_mode": "DETERMINISTIC"}}
    assert cache_mod.set_cached_analysis("k", data) is False
    assert cache_mod.get_cached_analysis("k") is None


def test_deterministic_fallback_is_not_cached():
    data = {"report": {"analysis_mode": "LLM", "semantic_mode": "DETERMINISTIC_FALLBACK"}}
    assert cache_mod.set_cached_analysis("k", data) is False
    assert cache_mod.get_cached_analysis("k") is None


@pytest.mark.parametrize("status", ["LLM_UNAVAILABLE", "LLM_TIMEOUT"])
def test_transient_remediation_failure_is_not_cached(status):
    data = {"report": {"analysis_mode": "LLM", "semantic_mode": "CANONICAL_LLM",
                       "remediation_cost": {"remediation_semantic_status": status}}}
    assert cache_mod.set_cached_analysis("k", data) is False


@pytest.mark.parametrize("status", ["LLM_UNAVAILABLE", "LLM_TIMEOUT"])
def test_transient_financial_failure_is_not_cached(status):
    data = {"report": {"analysis_mode": "LLM", "semantic_mode": "CANONICAL_LLM",
                       "financial_analysis": {"financial_semantic_status": status}}}
    assert cache_mod.set_cached_analysis("k", data) is False


def test_missing_report_is_not_cached():
    assert cache_mod.set_cached_analysis("k", {"report": None}) is False
    assert cache_mod.set_cached_analysis("k", {}) is False


def test_valid_deterministic_analysis_is_cached():
    # A pure deterministic-floor interpretation (flag off) is a legitimate,
    # reusable result -- it must NOT be mistaken for a degraded one.
    data = {"report": {"analysis_mode": "DETERMINISTIC", "semantic_mode": "DETERMINISTIC"}}
    assert cache_mod.set_cached_analysis("k", data) is True
    assert cache_mod.get_cached_analysis("k") is not None


def test_successful_result_is_cached_and_preserves_payload():
    remediation = {"remediation_semantic_status": "OK", "ai_provenance": {"model": "m", "prompt_version": "v"}}
    data = {"report": {"analysis_mode": "LLM", "semantic_mode": "CANONICAL_LLM", "remediation_cost": remediation},
            "ai_metadata": {"model": "m", "prompt_version": "v", "suggestion_id": "sid"}}
    assert cache_mod.set_cached_analysis("k", data) is True
    got = cache_mod.get_cached_analysis("k")
    assert got["report"]["remediation_cost"]["ai_provenance"] == {"model": "m", "prompt_version": "v"}
    assert got["ai_metadata"]["suggestion_id"] == "sid"


def test_first_call_degraded_second_call_is_independent_execution():
    """A degraded first result must not become a cache hit for the retry."""
    key = "same-finding"
    degraded = {"report": {"analysis_mode": "DEGRADED", "semantic_mode": "DETERMINISTIC"}}
    assert cache_mod.set_cached_analysis(key, degraded) is False
    # retry path checks the cache first -- must be a miss, forcing real execution
    assert cache_mod.get_cached_analysis(key) is None
    healthy = _ok("healthy-retry")
    assert cache_mod.set_cached_analysis(key, healthy) is True
    assert cache_mod.get_cached_analysis(key)["report"]["tag"] == "healthy-retry"
