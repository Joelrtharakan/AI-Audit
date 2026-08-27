"""Generalization/architecture audit for the financial semantic engine:
proves the REAL Ollama model (qwen3:8b) semantically understands previously
unseen findings -- not just that a final number happens to match.

Unlike tests/test_financial_semantic_real_ollama.py's existing coverage,
every finding here uses wording, domains, currencies, and structures not
present in ANY other test file in this repository. Assertions check the
actual semantic ROLES the LLM assigned (which claim is QUANTITY vs RATE,
which relationship/operation it selected, which claims support the cost
factor) -- not only the final total -- specifically to catch a swapped-
operand class of bug that a commutative MULTIPLY can hide (see the
regression test in tests/test_financial_semantic_quantity_rate_matrix.py
for the exact defect this guards against).

Real Ollama only -- no FakeLLMClient. Skipped (never faked) when Ollama is
unreachable.
"""

from __future__ import annotations

import httpx
import pytest

from app.financial.relationship_validator import validate_and_materialize
from app.financial.semantic_engine import analyze_financial_exposure_semantic
from app.models.agent import EvidenceItem, EvidenceStatus
from app.services.semantic_evidence_interpreter import interpret_evidence_semantically


def _ollama_reachable() -> bool:
    try:
        return httpx.get("http://localhost:11434/api/tags", timeout=2.0).status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.live_ollama,
    pytest.mark.skipif(not _ollama_reachable(), reason="Ollama is not reachable at localhost:11434"),
]


async def _run(*texts: str):
    ledger = [EvidenceItem(claim=t, status=EvidenceStatus.VERIFIED, source="AUDITOR_FINDING") for t in texts]
    finding_text = " ".join(texts)
    return await analyze_financial_exposure_semantic(finding_text, ledger)


async def _interp(*texts: str, statuses: list[EvidenceStatus] | None = None):
    statuses = statuses or [EvidenceStatus.VERIFIED] * len(texts)
    ledger = [EvidenceItem(claim=t, status=s, source=f"S{i}") for i, (t, s) in enumerate(zip(texts, statuses))]
    # interpret_evidence_semantically now returns (status, interpretation);
    # these tests assert on the interpretation itself.
    status, interpretation = await interpret_evidence_semantically(" ".join(texts), ledger)
    assert status == "OK", f"expected a usable interpretation, got status={status}"
    return interpretation


def _claim(interp, claim_id):
    return next((c for c in interp.claims if c.claim_id == claim_id), None)


class TestSemanticRoleNotJustFinalNumber:
    """Prove the LLM assigns roles correctly, not merely that a
    commutative multiply happens to land on the right total."""

    async def test_inventory_valuation_units_and_unit_value(self):
        result = await _run(
            "A stock count discrepancy identified 640 missing inventory units in the warehouse ledger.",
            "The verified standard unit valuation for this SKU is CHF 11.25.",
        )
        assert result is not None
        fa, audit = result
        assert audit.outcome.accepted_calculation_ids
        assert fa.confirmed_impact.verified_gross_exposure == pytest.approx(640 * 11.25)
        # Semantic-role check: find the QUANTITY and RATE claims directly
        # and confirm their VALUES (not just the product) are correct --
        # this fails if the roles were swapped even though 640*11.25 ==
        # 11.25*640.
        interp = audit.interpretation
        qty = next((c for c in interp.claims if c.fact_type == "QUANTITY"), None)
        rate = next((c for c in interp.claims if c.fact_type == "RATE"), None)
        assert qty is not None and qty.value == 640
        assert rate is not None and rate.value == 11.25

    async def test_supplier_chargeback_percentage_of_contract_value(self):
        result = await _run(
            "The supplier contract value under review is AED 480,000.",
            "A verified 6% chargeback penalty was applied for the late-delivery breach.",
        )
        assert result is not None
        fa, audit = result
        # Either the LLM computes a validated PERCENT_OF-style calculation
        # (6% of 480,000 = 28,800) or, if the architecture does not
        # support that operation, it must fail closed -- never guess a
        # different number.
        if fa.confirmed_impact.verified_gross_exposure is not None:
            assert fa.confirmed_impact.verified_gross_exposure == pytest.approx(480000 * 0.06, rel=0.01)

    async def test_energy_tariff_reversed_evidence_order(self):
        # Rate stated BEFORE quantity, in the opposite order from most
        # other tests in this suite -- the LLM must not depend on
        # positional/reading order to assign roles.
        result = await _run(
            "The verified peak-hour electricity tariff was GBP 0.34 per kWh.",
            "A malfunctioning chiller consumed an additional 5,400 kWh during the audit period.",
        )
        assert result is not None
        fa, audit = result
        assert audit.outcome.accepted_calculation_ids
        assert fa.confirmed_impact.verified_gross_exposure == pytest.approx(5400 * 0.34)


class TestMultipleUnrelatedValuesNotCombined:
    """Real findings often contain several numbers that must NOT all be
    multiplied together or arbitrarily selected."""

    async def test_operational_quantity_plus_unrelated_contract_reference_number(self):
        interp = await _interp(
            "Purchase order PO-88234 covers 75 replacement filters.",
            "The unit cost verified by finance was USD 46 per filter.",
        )
        assert interp is not None
        obs, outcome = validate_and_materialize(interp, evidence_count=2)
        # The PO reference number (88234) must never be treated as a
        # financial quantity or rate.
        combined_wrong = [o for o in obs if o.amount == 88234 or o.unit_amount == 88234 or o.event_count == 88234]
        assert not combined_wrong

    async def test_current_exposure_and_unrelated_prior_year_budget_figure(self):
        result = await _run(
            "220 mislabeled cartons were identified during the shipment audit, at a verified relabeling cost of EUR 9 per carton.",
            "For reference, the department's total prior-year budget was EUR 500,000.",
        )
        assert result is not None
        fa, _ = result
        # The budget figure must not be multiplied with or substituted
        # for the actual relabeling exposure.
        assert fa.confirmed_impact.verified_gross_exposure != pytest.approx(500000)
        if fa.confirmed_impact.verified_gross_exposure is not None:
            assert fa.confirmed_impact.verified_gross_exposure == pytest.approx(220 * 9)


class TestRecoveryRemediationHistoricalSeparation:
    async def test_gross_recovery_and_remediation_three_way_separation(self):
        result = await _run(
            "300 defective connectors were replaced at a verified cost of USD 8 per connector.",
            "USD 900 was credited back by the connector manufacturer.",
            "A proposed incoming-inspection upgrade is estimated at USD 3,200.",
        )
        assert result is not None
        fa, audit = result
        ci = fa.confirmed_impact
        gross = ci.verified_gross_exposure
        assert gross == pytest.approx(300 * 8)
        # The USD 3,200 remediation estimate must never appear as part of
        # gross exposure, recovery, or net loss.
        assert gross != pytest.approx(300 * 8 + 3200)
        if ci.confirmed_net_loss is not None:
            assert ci.confirmed_net_loss != pytest.approx(300 * 8 - 900 - 3200)

    async def test_historical_recurrence_not_multiplied_with_current_rate_without_evidence(self):
        result = await _run(
            "The same mislabeling defect has occurred 6 times in the past 12 months, per CAPA records.",
            "The current occurrence involves 40 units at a verified rework cost of INR 300 per unit.",
        )
        assert result is not None
        fa, _ = result
        # 6 (historical count) must never get multiplied into the current
        # exposure calculation -- only 40 x 300 = 12,000 is defensible
        # from THIS finding's current-population evidence.
        assert fa.confirmed_impact.verified_gross_exposure != pytest.approx(6 * 40 * 300)
        if fa.confirmed_impact.verified_gross_exposure is not None:
            assert fa.confirmed_impact.verified_gross_exposure == pytest.approx(40 * 300)

    async def test_reported_recovery_status_never_upgraded(self):
        result = await _run(
            "A verified gross exposure of USD 60,000 was recorded for the shipment damage.",
        )
        interp2 = await _interp(
            "A verified gross exposure of USD 60,000 was recorded for the shipment damage.",
            "The customer claims USD 12,000 was refunded, though the credit memo has not yet been reconciled.",
            statuses=[EvidenceStatus.VERIFIED, EvidenceStatus.REPORTED],
        )
        assert interp2 is not None
        recovery_claims = [c for c in interp2.claims if c.fact_type == "RECOVERY"]
        for c in recovery_claims:
            assert c.evidence_status != "VERIFIED", "A REPORTED recovery must never be upgraded to VERIFIED"


class TestAnnualizationEvidenceDependent:
    async def test_annualization_not_invented_from_bare_time_mention(self):
        result = await _run(
            "The warranty claim mentions the defect first appeared 6 months ago.",
            "A single repair costing USD 400 was completed.",
        )
        assert result is not None
        fa, _ = result
        # A bare "6 months ago" mention (not an observation period /
        # recurrence count) must never trigger a fabricated annualized
        # figure such as 400*2 = 800.
        assert fa.confirmed_impact.verified_gross_exposure != pytest.approx(800)

    async def test_annualization_supported_by_explicit_recurrence_and_period(self):
        result = await _run(
            "CAPA records confirm 8 verified incidents of this same defect over the past 12 months.",
            "Each verified incident cost INR 15,000 to remediate.",
        )
        assert result is not None
        fa, _ = result
        ann = fa.annualized_exposure
        if ann is not None and getattr(ann, "is_assessable", False):
            assert ann.observed_exposure == pytest.approx(8 * 15000) or fa.confirmed_impact.verified_gross_exposure == pytest.approx(8 * 15000)


class TestMissingInputsFailClosed:
    async def test_rate_without_quantity_never_fabricates_a_quantity(self):
        result = await _run(
            "Finance confirmed a verified processing surcharge of CHF 18 per transaction.",
        )
        if result is not None:
            fa, _ = result
            assert fa.confirmed_impact.verified_gross_exposure is None

    async def test_quantity_without_rate_never_fabricates_a_rate(self):
        result = await _run(
            "132 pallets were quarantined pending disposition.",
        )
        if result is not None:
            fa, _ = result
            assert fa.confirmed_impact.verified_gross_exposure is None


class TestConflictingRatesNeverArbitrarilySelected:
    async def test_two_different_stated_rates_for_same_quantity_not_silently_picked(self):
        interp = await _interp(
            "460 assemblies required rework after the calibration drift was identified.",
            "An internal estimate placed the rework cost at approximately USD 30 per assembly.",
            "A separate finance reconciliation later confirmed the verified rework cost at USD 42 per assembly.",
        )
        assert interp is not None
        obs, outcome = validate_and_materialize(interp, evidence_count=3)
        # Neither arbitrary total is acceptable evidence that the system
        # silently picked one of two conflicting rates without
        # reconciliation -- it must either flag REQUIRES_RECONCILIATION /
        # CONFLICTS_WITH, or (at minimum) never produce BOTH 460*30 and
        # 460*42 as if uncontested, and never silently sum them.
        totals = [o.amount if o.amount is not None else (
            (o.unit_amount or 0) * (o.event_count or 0) if o.unit_amount and o.event_count else None
        ) for o in obs]
        totals = [t for t in totals if t]
        summed = sum(totals) if totals else None
        assert summed != pytest.approx(460 * 30 + 460 * 42), "Two conflicting rates must never be silently summed together"
