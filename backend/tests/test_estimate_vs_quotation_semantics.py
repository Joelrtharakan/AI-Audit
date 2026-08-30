"""Two independent quantified assessments with EXPLICITLY UNRESOLVED
comparability must not collapse into an operational-noncompliance finding.

Demonstrated defect: for

    "Engineering estimated that replacement of the equipment control module
     would cost approximately ₹3 lakh. Procurement obtained a supplier
     quotation of ₹4.2 lakh, but the scope of the quotation had not yet
     been confirmed to match the engineering estimate."

the deterministic resolver produced

    finding_subject   = "engineering estimate"        (evidence artifact!)
    condition         = "not used as required"        (fabricated requirement!)
    comparison        = discarded                      (₹3 lakh / ₹4.2 lakh lost)

The fix is structural (grammatical role: ACTOR / EVIDENCE-SOURCE /
MEASUREMENT / COMPARISON / AFFECTED-OBJECT), not a keyword blacklist, and
lives in the EARLIEST layer -- `resolve_deviation()` (the deterministic
floor). Assertions are on semantic invariants, not exact prose.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app.services.semantic_subject import resolve_deviation

DEMO = (
    "Engineering estimated that replacement of the equipment control module "
    "would cost approximately ₹3 lakh. Procurement obtained a supplier "
    "quotation of ₹4.2 lakh, but the scope of the quotation had not yet been "
    "confirmed to match the engineering estimate."
)

# (finding, expected_subject_substring, magnitude_1, magnitude_2)
MATRIX = [
    (  # A. estimate vs quotation (the demonstrated finding)
        DEMO,
        "equipment control module",
        "3 lakh", "4.2 lakh",
    ),
    (  # C. budget/estimate vs quotation, different currency magnitudes
        "Finance estimated the project at ₹10 lakh, while the contractor quoted "
        "₹14 lakh; the scope had not been reconciled.",
        "project",
        "10 lakh", "14 lakh",
    ),
    (  # E. engineering hours vs contractor hours
        "Engineering estimated repair at 50 hours, while the service provider "
        "quoted 80 hours; the work scope had not been confirmed equivalent.",
        "repair",
        "50 hours", "80 hours",
    ),
    (  # F. two measurements, unresolved population comparability
        "Quality reported 2% defect rate, while production records showed 5%; "
        "the populations were not yet confirmed comparable.",
        "defect rate",
        "2%", "5%",
    ),
    (  # L. lab vs independent lab, unresolved method equivalence
        "The laboratory reported an estimated concentration of X, while the "
        "independent laboratory reported Y; the methods were not confirmed "
        "equivalent.",
        "concentration",
        "X", "Y",
    ),
    (  # M. IT estimate vs vendor proposal
        "IT estimated 20 hours for the migration, while the vendor proposed 40 "
        "hours; the proposed scope had not been reconciled.",
        "migration",
        "20 hours", "40 hours",
    ),
]

_EVIDENCE_SOURCE_TOKENS = (
    "engineering", "procurement", "finance", "quality", "vendor", "contractor",
    "supplier", "laboratory", "it ", "estimate", "quotation", "quote",
    "proposal", "figure",
)


@pytest.mark.parametrize("finding,subj_sub,m1,m2", MATRIX, ids=[c[0][:40] for c in MATRIX])
def test_dual_assessment_comparability_semantics(finding, subj_sub, m1, m2):
    d = resolve_deviation(finding, [])

    subject = (d.finding_subject or d.subject or "").lower()

    # 1 + 7. correct canonical subject; no evidence-source / artifact promoted
    assert subj_sub.lower() in subject, f"subject={subject!r}"
    assert not subject.startswith(("engineering estimate", "supplier quotation",
                                   "the engineering estimate", "engineering ",
                                   "procurement ", "finance ", "quality "))
    assert "estimate" != subject and "quotation" != subject

    # 8. no unsupported "not used as required" / compliance framing
    blob = " ".join(str(x or "") for x in (d.deviation, d.condition,
                                           d.affected_process)).lower()
    assert "not used as required" not in blob
    assert "not adhered to" not in blob
    assert "noncompliance" not in blob and "non-compliance" not in blob

    # 2. observed condition is a comparability statement
    assert d.semantic_type == "COMPARISON"
    assert d.comparison_type == "UNRESOLVED_COMPARABILITY"

    # 3 + 4 + 5. comparison + both magnitudes preserved (with provenance)
    lr = f"{d.comparison_left} || {d.comparison_right}".lower()
    assert m1.lower() in lr and m2.lower() in lr
    assert d.comparison_basis
    cond_blob = f"{d.deviation} {d.condition}".lower()
    assert m1.lower() in cond_blob and m2.lower() in cond_blob

    # 6. evidence provenance retained on each side
    assert any(t in d.comparison_left.lower() for t in ("estimate", "report", "figure"))
    assert any(t in d.comparison_right.lower()
               for t in ("quotation", "quote", "proposal", "report", "figure"))

    # 9. no established cause / requirement asserted
    assert d.requirement_status == "UNKNOWN"

    # 12 + 13. no manufactured percentage difference / loss / remediation price
    for junk in ("% difference", "percent difference", "loss of", "overrun",
                 "savings of", "remediation cost"):
        assert junk not in blob


def test_demo_finding_does_not_promote_engineering_estimate():
    d = resolve_deviation(DEMO, [])
    assert (d.finding_subject or "").lower() not in ("engineering estimate",
                                                     "engineering", "procurement")
    assert "replacement" in (d.finding_subject or "").lower()


def test_single_assessment_is_not_treated_as_dual_comparison():
    """No second assessment -> the dual-comparison branch must not fire."""
    d = resolve_deviation(
        "Engineering estimated that the repair would cost ₹3 lakh.", []
    )
    assert d.comparison_type != "UNRESOLVED_COMPARABILITY"


def test_demo_finding_canonical_state_end_to_end():
    """10. `understand_finding_node` (flag OFF, no LLM) must carry the
    corrected COMPARISON semantics into `canonical_finding_state` -- the
    comparison spans two sentences and must survive per-segment filtering."""
    from app.agent.nodes.understanding import understand_finding_node
    from app.models.agent import InvestigateRequest

    st = {
        "request": InvestigateRequest(finding_text=DEMO),
        "evidence_ledger": [], "trace": [], "errors": [],
        "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
    }
    with patch("app.agent.nodes.understanding.get_llm_client", lambda **k: None):
        out = asyncio.run(understand_finding_node(st))
    cf = out["canonical_finding_state"]
    subj = (cf.finding_subject or "").lower()
    assert "equipment control module" in subj
    assert subj not in ("engineering estimate", "scope", "engineering", "procurement")
    assert cf.semantic_type == "COMPARISON"
    assert cf.comparison_type == "UNRESOLVED_COMPARABILITY"
    blob = f"{cf.deviation_condition} {cf.observed_deviation}".lower()
    assert "3 lakh" in blob and "4.2 lakh" in blob
    assert "not used as required" not in blob
    assert "engineering estimate" in (cf.comparison_left or "").lower()
    assert "supplier quotation" in (cf.comparison_right or "").lower()


def test_two_assessments_with_confirmed_equivalent_scope_not_flagged_unresolved():
    """G. comparability explicitly ESTABLISHED -> not an unresolved-comparability
    finding (the negation gate must require an actual negation)."""
    d = resolve_deviation(
        "Engineering estimated repair at 50 hours and the contractor quoted 55 "
        "hours; the work scope was confirmed equivalent.", []
    )
    assert d.comparison_type != "UNRESOLVED_COMPARABILITY"
