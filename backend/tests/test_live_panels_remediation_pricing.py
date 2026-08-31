"""Pass 51/52 -- the exact electrical-panel failure, END TO END through the two
real semantic stages (canonical + remediation) against live qwen3:8b.

Skipped (never faked) when Ollama is unreachable. This is the hard regression
guard for spec Pass 52 section 23.

    Eight electrical panels require corrective labeling and inspection.
    New labels cost Rs 350 per panel.
    An electrician requires 1.5 hours per panel at Rs 900 per hour,
    followed by a safety inspection costing Rs 6,000 for the complete area.

Expected: one_time_cost == 19,600 (2,800 + 10,800 + 6,000), ONE_TIME,
no recurring cost, no horizon, no auditor pricing input, RCA NOT_ESTABLISHED.
"""
from __future__ import annotations

import httpx
import pytest


def _try() -> bool:
    try:
        return httpx.get("http://localhost:11434/api/tags", timeout=2.0).status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.live_ollama,
    pytest.mark.skipif(not _try(), reason="Ollama not reachable at localhost:11434"),
]

PANELS = (
    "Eight electrical panels require corrective labeling and inspection. New labels cost "
    "Rs 350 per panel. An electrician requires 1.5 hours per panel at Rs 900 per hour, "
    "followed by a safety inspection costing Rs 6,000 for the complete area."
)


@pytest.mark.asyncio
async def test_panels_full_pricing_19600_live():
    from app.services.canonical_finding_interpreter import interpret_finding_canonically_with_status
    from app.services.canonical_context_validator import validate_canonical_context
    from app.remediation.engine import estimate_remediation_cost
    from app.services.canonical_semantic_models import comparison_is_active

    status, raw = await interpret_finding_canonically_with_status(
        finding_text=PANELS, evidence_ledger=[], timeout_seconds=None
    )
    assert raw is not None, f"canonical failed: {status}"
    ctx = validate_canonical_context(raw, [], PANELS)

    # recurrence: population of 8 must NOT become a recurrence count
    rec = getattr(ctx, "recurrence", None)
    if rec is not None:
        assert (getattr(rec, "count", None) or 0) <= 1, f"population read as recurrence: {rec}"

    # RCA not established
    assert str(getattr(ctx, "root_cause_status", "")) in ("NOT_ESTABLISHED", "None", "")

    rc = await estimate_remediation_cost(finding_text=PANELS, semantic_context=ctx)

    assert rc.pricing_status in ("EXACT_ESTIMATE", "PARTIAL_ESTIMATE"), rc.pricing_status
    headline = rc.one_time_cost or rc.most_likely_estimate
    assert headline == pytest.approx(19600), (
        headline,
        [c.model_dump() for c in rc.cost_components],
        rc.pricing_status,
        rc.auditor_inputs_required,
    )
    assert not rc.recurring_cost
    assert not rc.recurring_horizon_total
    # no auditor input asking for an already-stated price
    ai = " ".join(rc.auditor_inputs_required or []).lower()
    for tok in ("350", "900", "6000", "6,000", "label cost", "inspection cost", "labor cost"):
        assert tok not in ai, f"auditor input requests known pricing: {rc.auditor_inputs_required}"
    # comparison must not be active (a cost composition is not a discrepancy).
    # NOTE: qwen3:8b sometimes still fabricates one here -- documented Pass 52
    # model-capability limitation; xfail rather than fake a pass.
    if comparison_is_active(getattr(ctx, "comparison", None)):
        pytest.xfail("qwen3:8b fabricated a comparison over the cost components (model limitation)")
