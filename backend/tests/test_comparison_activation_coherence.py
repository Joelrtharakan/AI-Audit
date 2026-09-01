"""Pass 53: a false comparison over independent cost components must NOT
activate (no reconciliation obligation, no Q_COMPARABILITY), while a genuine
actual-vs-standard comparison still does.

Deterministic layer only: hand-built SemComparison objects through
`validate_canonical_context` -> `comparison_is_active`. This is the structural
coherence guard (spec Pass 53 section 11/21), NOT a "multiple prices -> no
comparison" rule -- it reads only the model's own comparison fields.
"""
from __future__ import annotations

import pytest

from app.services.canonical_context_validator import validate_canonical_context
from app.services.canonical_semantic_models import (
    CanonicalFindingContext,
    SemComparison,
    comparison_is_active,
)


def _ctx(cmp_kwargs: dict, finding_text: str = "") -> CanonicalFindingContext:
    c = CanonicalFindingContext(comparison=SemComparison(**cmp_kwargs))
    return validate_canonical_context(c, [], finding_text)


# ---------------------------------------------------------------- NON-comparison
FALSE_COMPARISONS = [
    dict(
        left="label cost", right="electrician labor cost", reference="E1,E2",
        status="UNRESOLVED_COMPARISON",
        why_comparable="costs for labeling and inspection are listed separately",
        comparison_basis="remediation cost", direction="UNKNOWN",
    ),
    dict(
        left="replacement part price", right="inspection price",
        status="ACTUAL_CONFLICT", why_comparable="both are remediation costs",
        comparison_basis="cost", direction="UNKNOWN",
    ),
    dict(
        left="material rate", right="labour rate", status="UNRESOLVED_COMPARISON",
        why_comparable="different components of the same remediation",
        direction="UNKNOWN",
    ),
]


@pytest.mark.parametrize("kw", FALSE_COMPARISONS)
def test_false_comparison_over_cost_components_is_inactive(kw):
    ctx = _ctx(kw)
    assert not comparison_is_active(getattr(ctx, "comparison", None)), (
        getattr(ctx, "comparison", None)
    )


# ---------------------------------------------------------------- TRUE comparison
TRUE_COMPARISONS = [
    dict(left="actual temperature 92 C", right="limit 80 C", reference="limit of 80 C",
         status="ACTUAL_CONFLICT", why_comparable="the process must stay within the stated limit",
         direction="ABOVE"),
    dict(left="measured pressure 7 bar", right="required 5 bar",
         reference="specification requires 5 bar", status="ACTUAL_CONFLICT",
         why_comparable="measured value must meet the specification", direction="ABOVE"),
    dict(left="supplier quote Rs 90,000", right="approved budget Rs 75,000",
         reference="approved budget of Rs 75,000", status="ACTUAL_CONFLICT",
         why_comparable="the quotation is expected to be within the approved budget",
         direction="ABOVE", magnitude=15000.0),
    dict(left="actual output 850 units", right="target 1000 units",
         reference="target of 1,000 units", status="ACTUAL_CONFLICT",
         why_comparable="output is measured against the production target", direction="BELOW"),
    dict(left="pump vibration 8 mm/s", right="allowable 4 mm/s", reference="allowable 4 mm/s",
         status="ACTUAL_CONFLICT", why_comparable="vibration must stay within the allowable limit",
         direction="ABOVE"),
    # direction UNKNOWN but left/right carry the numbers -> still coherent
    dict(left="measured 48.2 mL", right="specification 50.0 mL",
         status="ACTUAL_CONFLICT",
         why_comparable="the fill volume must conform to specification", direction="UNKNOWN"),
]


@pytest.mark.parametrize("kw", TRUE_COMPARISONS)
def test_genuine_comparison_still_activates(kw):
    ft = f"{kw['left']} against {kw['right']}"
    ctx = _ctx(kw, ft)
    assert comparison_is_active(getattr(ctx, "comparison", None)), (
        getattr(ctx, "comparison", None)
    )


from app.services.canonical_semantic_models import SemRecurrence


def test_bare_recurrence_count_with_no_event_or_period_is_cleared():
    """Pass 54: 'two-day audit' -> a stray recurrence {count: 2} with no
    `event` and no `period` is not a coherent repetition -> cleared."""
    c = CanonicalFindingContext(recurrence=SemRecurrence(count=2))
    out = validate_canonical_context(c, [], "the supplier will undergo a two-day quality audit")
    assert getattr(out, "recurrence", None) is None


def test_genuine_recurrence_with_event_and_period_survives():
    c = CanonicalFindingContext(
        recurrence=SemRecurrence(count=4, event="failures", period="the previous six months")
    )
    out = validate_canonical_context(
        c, [], "four failures occurred in the previous six months"
    )
    assert getattr(out, "recurrence", None) is not None
    assert out.recurrence.count == 4


def test_condition_recurrence_named_event_survives_even_with_onetime_remediation():
    c = CanonicalFindingContext(
        recurrence=SemRecurrence(event="packaging defects", period="recent deliveries")
    )
    out = validate_canonical_context(
        c, [], "following recurring packaging defects a two-day audit will be held"
    )
    assert getattr(out, "recurrence", None) is not None


def test_no_q_comparability_when_comparison_inactive():
    """The investigation-plan fallback gates Q_COMPARABILITY on
    comparison_is_active -- an inactive comparison yields no such question."""
    from app.agent.nodes.plan_investigation_fallback import _plan_from_canonical_structure

    ctx = _ctx(FALSE_COMPARISONS[0])
    _hyps, plan = _plan_from_canonical_structure(ctx, "the affected process")
    blob = repr(plan)
    assert "Q_COMPARABILITY" not in blob
    assert "directly comparable" not in blob
    assert "same scope" not in blob
