"""In-process observability for the LLM synthesis boundary.

Deliberately NOT a full metrics/exposition stack (no Prometheus client, no
persistence) -- this is the structured counter + latency/token telemetry set
the production-hardening pass asked for: cumulative counts and running
averages a caller can read back (e.g. the /health/llm-metrics endpoint or a
periodic log line), on top of the already-rich PER-CALL logging
`ollama_client.py` emits (elapsed_ms, prompt_eval_count, tokens_generated,
tokens_per_second, done_reason). That per-call logging is the ground-truth
audit trail for any one request; what this module adds is the aggregated
view across requests within this process's lifetime, plus a bounded ring
buffer of recent per-request execution records.

Thread/task-safety: increments are simple dict mutations under the GIL,
sufficient for this process-local, best-effort telemetry (not a
correctness-critical ledger) -- no lock needed, same reasoning as any other
non-atomic-but-monotonic counter elsewhere in this codebase.

Privacy: nothing in this module ever accepts or stores finding text,
evidence text, claim text, prompts, or LLM response bodies -- only counts,
durations, token numbers, node/model identifiers, and short structured
reason codes. Callers must never pass raw finding/evidence/prompt/response
content into any function here.
"""

from __future__ import annotations

import time

_COUNTERS: dict[str, int] = {
    "llm_primary_attempted": 0,
    "llm_primary_success": 0,
    "llm_primary_timeout": 0,
    "llm_primary_invalid_json": 0,
    "llm_primary_other_failure": 0,
    "llm_recovery_attempted": 0,
    "llm_recovery_success": 0,
    "llm_recovery_timeout": 0,
    "llm_recovery_invalid_json": 0,
    "llm_recovery_other_failure": 0,
    "deterministic_fallback": 0,
    # Semantically separated per Phase 5: an LLM-proposed hypothesis and a
    # deterministic-fallback-generated hypothesis are different populations
    # -- conflating them would misrepresent how much of the report's
    # causal content actually came from the model.
    "llm_hypotheses_generated": 0,
    "llm_hypotheses_accepted": 0,
    "llm_hypotheses_rejected": 0,
    "deterministic_hypotheses_generated": 0,
}

# Structured validation-event reason taxonomy (Phase 4): deliberately a
# small, fixed set rather than one enum member per guard function -- each
# reason groups a family of related guards (e.g. every causal-specificity
# guard in causal_guard.py reports "unsupported_causal_specificity") so the
# taxonomy stays meaningful at a glance instead of fragmenting into dozens
# of near-duplicate reasons.
VALIDATION_REASONS = frozenset({
    "missing_provenance",
    "invalid_provenance",
    "unsupported_causation",
    "unsupported_causal_specificity",
    "unsupported_impact",
    "invalid_hypothesis",
    "other",
})

# Per-reason rejection/repair counters, keyed "validation_rejections:<reason>"
# / "validation_repairs:<reason>" inside _COUNTERS itself (single dict, no
# second parallel structure) so `snapshot()`/`reset()` stay single-source.


def _reason_key(kind: str, reason: str) -> str:
    safe_reason = reason if reason in VALIDATION_REASONS else "other"
    return f"{kind}:{safe_reason}"


def record_validation_rejection(reason: str, node: str) -> None:
    """Structured replacement for trace-message substring inference
    ("dropped" in message). `reason` must be one of VALIDATION_REASONS
    (silently coerced to "other" otherwise -- never raises, since a
    misclassified metric is far cheaper than a crashed request). `node`
    is recorded only as a short caller identifier (e.g.
    "final_evidence_verification"), never as free text."""
    increment("validation_rejections_total")
    increment(_reason_key("validation_rejections", reason))
    increment(f"validation_rejections_by_node:{node}")


def record_validation_repair(reason: str, node: str) -> None:
    """Structured replacement for trace-message substring inference for
    the REPAIR case (content mutated/downgraded, not dropped entirely) --
    same reason taxonomy and node-identifier discipline as
    record_validation_rejection."""
    increment("validation_repairs_total")
    increment(_reason_key("validation_repairs", reason))
    increment(f"validation_repairs_by_node:{node}")


# Running sums for average latency/token computation -- kept separate from
# _COUNTERS since these are denominators/numerators for an average, not
# standalone counts a caller would read directly.
_SUMS: dict[str, int] = {
    "llm_primary_elapsed_ms_total": 0,
    "llm_primary_prompt_tokens_total": 0,
    "llm_primary_output_tokens_total": 0,
    "llm_recovery_elapsed_ms_total": 0,
    "llm_recovery_prompt_tokens_total": 0,
    "llm_recovery_output_tokens_total": 0,
}

# Bounded ring buffer of recent per-request execution records (Phase 7):
# request_id, node, model, elapsed_ms, prompt_tokens, output_tokens,
# failure_type, timestamp. No finding/evidence/prompt/response text ever
# stored here. `request_id` is an opaque uuid4 hex fragment generated in
# core_synthesis_node -- it carries no information about the finding itself
# and is reused across a single synthesis call's primary/recovery/fallback
# stages so the three can be correlated without storing case content.
_MAX_RECENT = 200
_RECENT_EXECUTIONS: list[dict] = []


def increment(name: str, by: int = 1) -> None:
    """Increment a named counter. Unknown names are created on first use
    (never raises) so a new counter can be added at a call site without a
    matching edit here, but the module-level dict above documents the
    canonical set."""
    _COUNTERS[name] = _COUNTERS.get(name, 0) + by


def record_execution(
    *,
    request_id: str | None,
    node: str,
    model: str,
    phase: str,  # "primary" | "recovery" | "fallback"
    elapsed_ms: int | None = None,
    prompt_tokens: int | None = None,
    output_tokens: int | None = None,
    failure_type: str | None = None,
) -> None:
    """Record one LLM call's execution telemetry: latency/token sums (for
    running averages) and a bounded recent-executions ring buffer. Never
    accepts or stores finding text or evidence content -- only the
    identifiers/numbers listed above. `request_id` should be the SAME
    value across a single synthesis call's primary/recovery/fallback
    stages (Phase 7 correlation), not regenerated per stage."""
    if elapsed_ms is not None:
        _SUMS[f"llm_{phase}_elapsed_ms_total"] = _SUMS.get(f"llm_{phase}_elapsed_ms_total", 0) + elapsed_ms
    if prompt_tokens is not None:
        _SUMS[f"llm_{phase}_prompt_tokens_total"] = _SUMS.get(f"llm_{phase}_prompt_tokens_total", 0) + prompt_tokens
    if output_tokens is not None:
        _SUMS[f"llm_{phase}_output_tokens_total"] = _SUMS.get(f"llm_{phase}_output_tokens_total", 0) + output_tokens

    _RECENT_EXECUTIONS.append({
        "request_id": request_id,
        "node": node,
        "model": model,
        "phase": phase,
        "elapsed_ms": elapsed_ms,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "failure_type": failure_type,
        "recorded_at": time.time(),
    })
    if len(_RECENT_EXECUTIONS) > _MAX_RECENT:
        del _RECENT_EXECUTIONS[: len(_RECENT_EXECUTIONS) - _MAX_RECENT]


def snapshot() -> dict[str, int]:
    """Return a copy of the current cumulative counters."""
    return dict(_COUNTERS)


def aggregated() -> dict:
    """Return the stable, externally-consumed metrics contract (Phase 3):
    the shape /health/llm-metrics returns. Internal counter names
    (_COUNTERS keys) are NOT renamed to match this contract -- this
    function is the single translation point, so the internal API used by
    core_synthesis.py/final_evidence_verification.py stays stable even if
    the external response shape needs to change independently."""
    c = snapshot()
    primary_success = c.get("llm_primary_success", 0)
    recovery_success = c.get("llm_recovery_success", 0)

    llm_hyp_generated = c.get("llm_hypotheses_generated", 0)
    llm_hyp_accepted = c.get("llm_hypotheses_accepted", 0)

    return {
        "llm_success_count": primary_success + recovery_success,
        "llm_timeout_count": c.get("llm_primary_timeout", 0) + c.get("llm_recovery_timeout", 0),
        "llm_invalid_json_count": c.get("llm_primary_invalid_json", 0) + c.get("llm_recovery_invalid_json", 0),
        "llm_other_failure_count": c.get("llm_primary_other_failure", 0) + c.get("llm_recovery_other_failure", 0),

        "primary_attempt_count": c.get("llm_primary_attempted", 0),
        "primary_success_count": primary_success,
        "primary_timeout_count": c.get("llm_primary_timeout", 0),
        "primary_invalid_json_count": c.get("llm_primary_invalid_json", 0),
        "primary_other_failure_count": c.get("llm_primary_other_failure", 0),

        "recovery_attempt_count": c.get("llm_recovery_attempted", 0),
        "recovery_success_count": recovery_success,
        "recovery_timeout_count": c.get("llm_recovery_timeout", 0),
        "recovery_invalid_json_count": c.get("llm_recovery_invalid_json", 0),
        "recovery_other_failure_count": c.get("llm_recovery_other_failure", 0),

        "deterministic_fallback_count": c.get("deterministic_fallback", 0),

        "llm_hypotheses_generated": llm_hyp_generated,
        "llm_hypotheses_accepted": llm_hyp_accepted,
        "llm_hypotheses_rejected": c.get("llm_hypotheses_rejected", 0),
        "deterministic_hypotheses_generated": c.get("deterministic_hypotheses_generated", 0),
        # Backward-compatible aliases (Section 5's example schema names) --
        # same numbers, non-LLM-specific field names for a caller that
        # doesn't care about the LLM/deterministic split.
        "hypotheses_generated": llm_hyp_generated + c.get("deterministic_hypotheses_generated", 0),
        "hypotheses_accepted": llm_hyp_accepted,
        "hypotheses_rejected": c.get("llm_hypotheses_rejected", 0),

        "provenance_rejections": (
            c.get("validation_rejections:missing_provenance", 0)
            + c.get("validation_rejections:invalid_provenance", 0)
        ),
        "causal_guard_rejections": (
            c.get("validation_rejections:unsupported_causation", 0)
            + c.get("validation_rejections:unsupported_causal_specificity", 0)
            + c.get("validation_rejections:unsupported_impact", 0)
            + c.get("validation_rejections:invalid_hypothesis", 0)
        ),
        "validation_rejections_total": c.get("validation_rejections_total", 0),
        "validation_repairs": c.get("validation_repairs_total", 0),

        "average_primary_latency_ms": (
            round(_SUMS["llm_primary_elapsed_ms_total"] / primary_success, 1) if primary_success else None
        ),
        "average_primary_output_tokens": (
            round(_SUMS["llm_primary_output_tokens_total"] / primary_success, 1) if primary_success else None
        ),
        "average_recovery_latency_ms": (
            round(_SUMS["llm_recovery_elapsed_ms_total"] / recovery_success, 1) if recovery_success else None
        ),
        "average_recovery_output_tokens": (
            round(_SUMS["llm_recovery_output_tokens_total"] / recovery_success, 1) if recovery_success else None
        ),

        "recent_execution_count": len(_RECENT_EXECUTIONS),
    }


def recent_executions() -> list[dict]:
    """Return a copy of the bounded recent-executions ring buffer."""
    return list(_RECENT_EXECUTIONS)


def reset() -> None:
    """Test-only: zero every counter/sum and clear recent executions. Never
    called from production code paths -- counters are meant to accumulate
    for the life of the process."""
    for key in list(_COUNTERS.keys()):
        _COUNTERS[key] = 0
    for key in _SUMS:
        _SUMS[key] = 0
    _RECENT_EXECUTIONS.clear()
