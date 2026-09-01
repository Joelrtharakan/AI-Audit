"""Pass 55: a remediation PRICING INPUT (or an arithmetic aggregate of priced
components) that the canonical LLM lists as an `information_gap` /
`investigation_plan` step / `investigation_activities` entry must be routed
OUT of the root-cause investigation plan -- the canonical layer owns semantic
gaps, the remediation layer + calculator own pricing and arithmetic.

The firewall is a cross-field consistency check on the model's OWN output: a
gap whose words match a `pricing_basis` / `pricing_evidence_needed` /
`scope_evidence_needed` the model itself declared is, by the model's own
classification, a pricing input. It reads no finding text and adds no
finding-text keyword classifier.
"""
from __future__ import annotations

import pytest

from app.services.canonical_context_validator import validate_canonical_context
from app.services.canonical_semantic_models import (
    CanonicalFindingContext,
    SemInvestigationStep,
    SemPricingItem,
    SemRemediationAction,
)

FRIDGE_TEXT = (
    "A pharmacy refrigerator requires replacement. The approved quotation is Rs 1,45,000. "
    "Delivery is Rs 8,000 and installation is Rs 12,000. Validation requires 4 hours at "
    "Rs 1,500 per hour."
)


def _fridge_ctx(**over) -> CanonicalFindingContext:
    base = dict(
        root_cause_status="NOT_ESTABLISHED",
        remediation_obligation="ESTABLISHED_CORRECTIVE_OBLIGATION",
        information_gaps=[
            "quotation for refrigerator",
            "installation cost",
            "biomedical-engineering work",
            "total remediation cost",
            "the requirement establishing that the refrigerator must be replaced",
        ],
        investigation_plan=[
            SemInvestigationStep(unknown="total cost components"),
            SemInvestigationStep(unknown="the cause of the refrigerator failure"),
        ],
        investigation_activities=[
            SemRemediationAction(action_id="v1", activity="Verify the installation cost",
                                 disposition="INVESTIGATION"),
        ],
        remediation_activities=[
            SemRemediationAction(action_id="r1", activity="Replace the pharmacy refrigerator",
                                 disposition="IMMEDIATE_CORRECTION",
                                 pricing_evidence_needed="quotation for refrigerator",
                                 scope_evidence_needed="single refrigerator"),
            SemRemediationAction(action_id="r2", activity="Install the refrigerator",
                                 disposition="IMMEDIATE_CORRECTION",
                                 pricing_evidence_needed="installation cost"),
            SemRemediationAction(action_id="r3", activity="Validate the refrigerator",
                                 disposition="IMMEDIATE_CORRECTION",
                                 pricing_evidence_needed="biomedical-engineering work"),
        ],
        pricing_information=[
            SemPricingItem(action_id="r1", pricing_basis="quotation for refrigerator",
                           observed_value_in_finding="Rs 145000"),
            SemPricingItem(action_id="r2", pricing_basis="installation cost",
                           observed_value_in_finding="Rs 8000"),
            SemPricingItem(action_id="r3", pricing_basis="biomedical-engineering work",
                           observed_value_in_finding="Rs 1500 per hour"),
        ],
    )
    base.update(over)
    return CanonicalFindingContext(**base)


def test_pricing_input_gaps_are_removed_semantic_gap_kept():
    out = validate_canonical_context(_fridge_ctx(), [], FRIDGE_TEXT)
    assert out.information_gaps == [
        "the requirement establishing that the refrigerator must be replaced"
    ]
    assert [s.unknown for s in out.investigation_plan] == ["the cause of the refrigerator failure"]
    assert out.investigation_activities == []
    # remediation untouched
    assert len(out.remediation_activities) == 3


def test_arithmetic_aggregate_gap_is_removed():
    for agg in ("total remediation cost", "the total cost", "reconcile the cost components",
                "verify total remediation cost", "confirm the cost breakdown"):
        out = validate_canonical_context(
            _fridge_ctx(information_gaps=[agg, "the governing requirement for replacement"]),
            [], FRIDGE_TEXT,
        )
        assert agg not in out.information_gaps, agg
        assert "the governing requirement for replacement" in out.information_gaps


def test_firewall_no_op_when_no_pricing_information():
    """No pricing_information / pricing_evidence_needed -> nothing to key off ->
    every gap is preserved (the firewall never guesses)."""
    ctx = CanonicalFindingContext(
        root_cause_status="NOT_ESTABLISHED",
        information_gaps=["the cause of the failure", "the applicable requirement",
                          "the total cost"],
    )
    out = validate_canonical_context(ctx, [], "x")
    assert set(out.information_gaps) == {
        "the cause of the failure", "the applicable requirement", "the total cost",
    }


def test_genuine_pricing_conflict_gap_survives():
    """A gap about WHICH of two conflicting approved records governs is a real
    evidence-conflict semantic gap -- its words are not a subset of any single
    declared pricing basis, and it is not a pure aggregate."""
    ctx = _fridge_ctx(information_gaps=[
        "which of the two conflicting approved quotation records governs the replacement",
        "installation cost",
    ])
    out = validate_canonical_context(ctx, [], FRIDGE_TEXT)
    assert any("conflicting" in g for g in out.information_gaps)
    assert "installation cost" not in out.information_gaps


def test_requirement_and_cause_gaps_never_dropped():
    for g in ("the governing requirement", "the cause of the refrigerator failure",
              "which specification applies", "whether a previous CAPA was implemented",
              "the scope of the affected population"):
        out = validate_canonical_context(_fridge_ctx(information_gaps=[g]), [], FRIDGE_TEXT)
        assert out.information_gaps == [g], g
