"""In-process observability counters for the LLM synthesis boundary.

Deliberately NOT a full metrics/exposition stack (no Prometheus client, no
persistence) -- this is the minimal structured counter set the production-
hardening pass asked for: cumulative counts a caller can read back (e.g. for
a health/debug endpoint or a periodic log line), on top of the already-rich
PER-CALL logging `ollama_client.py` emits (elapsed_ms, prompt_eval_count,
tokens_generated, tokens_per_second, done_reason). That per-call logging is
the source of truth for any one request; these counters are the cumulative
view across requests within this process's lifetime.

Thread/task-safety: increments are simple dict mutations under the GIL,
which is sufficient for this process-local, best-effort counter (not a
correctness-critical ledger) -- no lock needed for the same reason the
stdlib's own `collections.Counter.__getitem__`-then-`+=1` pattern is
considered acceptable for non-atomic-but-monotonic counters elsewhere in
this codebase.
"""

from __future__ import annotations

_COUNTERS: dict[str, int] = {
    "llm_primary_attempted": 0,
    "llm_primary_success": 0,
    "llm_primary_timeout": 0,
    "llm_primary_invalid_json": 0,
    "llm_recovery_attempted": 0,
    "llm_recovery_success": 0,
    "llm_recovery_timeout": 0,
    "llm_recovery_invalid_json": 0,
    "deterministic_fallback": 0,
    "hypotheses_generated": 0,
    "hypotheses_rejected": 0,
    "hypotheses_provenance_rejected": 0,
}


def increment(name: str, by: int = 1) -> None:
    """Increment a named counter. Unknown names are created on first use
    (never raises) so a new counter can be added at a call site without a
    matching edit here, but the module-level dict above documents the
    canonical set."""
    _COUNTERS[name] = _COUNTERS.get(name, 0) + by


def snapshot() -> dict[str, int]:
    """Return a copy of the current cumulative counters."""
    return dict(_COUNTERS)


def reset() -> None:
    """Test-only: zero every counter. Never called from production code
    paths -- counters are meant to accumulate for the life of the
    process."""
    for key in _COUNTERS:
        _COUNTERS[key] = 0
