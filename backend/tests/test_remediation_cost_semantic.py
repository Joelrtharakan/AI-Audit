"""End-to-end coverage for the Remediation Cost Estimation semantic pipeline
(app.remediation.interpreter / provider_normalization / validator / calculator /
engine).

No real LLM / network calls -- a `FakeLLMClient` returns hand-authored JSON
standing in for what a provider would plausibly produce. The property under test
throughout: the LLM's `proposed_result_value` is NEVER the rendered number --
only the deterministically re-executed value is authoritative.
"""

from __future__ import annotations

import json

import pytest

from app.remediation.engine import estimate_remediation_cost
from app.remediation.models import CostBasis, RemediationEstimateStatus
from app.models.agent import EvidenceItem, EvidenceStatus


class FakeLLMClient:
    def __init__(self, response: str | None = None, raise_exc: Exception | None = None):
        self.response = response
        self.raise_exc = raise_exc
        self.calls = 0

    async def chat_completion(self, messages, temperature=0.0, response_format_json=True, **kwargs):
        self.calls += 1
        if self.raise_exc:
            raise self.raise_exc
        return self.response


def _ev(claim: str, status: EvidenceStatus = EvidenceStatus.VERIFIED) -> EvidenceItem:
    return EvidenceItem(claim=claim, status=status, source="test")


async def _run(response, evidence, finding="A required control was found missing.", **kw):
    return await estimate_remediation_cost(
        finding_text=finding,
        evidence_ledger=evidence,
        client=FakeLLMClient(response=json.dumps(response) if not isinstance(response, str) else response),
        **kw,
    )


# --------------------------------------------------------------------------
# 1. Evidence-backed multiply → deterministic quantity × unit_cost
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evidenced_multiply_uses_executor_not_llm_number():
    interp = {
        "strategy": {"remediation_summary": "retrain affected staff", "remediation_type": "training",
                     "interpretation_confidence": "MEDIUM"},
        "activities": [{"activity_id": "A0", "description": "Deliver refresher training", "derived_from": "RECOMMENDED_CAPA"}],
        "cost_components": [{
            "component_id": "C0", "description": "Refresher training delivery",
            "activity_ids": ["A0"], "cost_category": "training",
            "quantity": 40, "quantity_unit": "person", "quantity_basis": "EVIDENCED",
            "unit_cost": 1200, "unit_cost_basis": "REPORTED", "currency": "INR",
            "amount_type": "PER_UNIT", "recurrence": "ONE_TIME",
            "source_reference_ids": ["E0", "E1"], "interpretation_confidence": "MEDIUM",
        }],
        "calculation_proposals": [{
            "calculation_id": "K0", "operation": "MULTIPLY", "component_ids": ["C0"],
            "produces": "MOST_LIKELY", "proposed_result_value": 999999, "reason": "qty x rate",
        }],
        "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
    }
    ev = [_ev("40 staff require refresher training"), _ev("Training vendor quoted INR 1,200 per person", EvidenceStatus.REPORTED)]
    res = await _run(interp, ev)
    assert res.status == RemediationEstimateStatus.EVIDENCE_BACKED
    assert res.most_likely_estimate == 48000.0          # 40 × 1200, NOT 999999
    assert res.one_time_cost == 48000.0
    assert res.currency == "INR"
    comp = res.cost_components[0]
    assert comp.calculated_amount == 48000.0
    assert comp.calculation_formula.startswith("40")
    # audit trace records the disagreement, never the rendered number
    assert any(t.disagreement for t in res.calculation_traces)


# --------------------------------------------------------------------------
# 2. Assumed pricing → ASSUMPTION_BASED, classification ASSUMED
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_assumed_pricing_is_assumption_based():
    interp = {
        "strategy": {"remediation_summary": "draft a procedure", "interpretation_confidence": "LOW"},
        "cost_components": [{
            "component_id": "C0", "description": "Procedure drafting effort",
            "cost_category": "labor", "quantity": 20, "quantity_unit": "hour",
            "quantity_basis": "ASSUMED", "unit_cost": 1500, "unit_cost_basis": "ASSUMED",
            "currency": "INR", "amount_type": "PER_HOUR", "recurrence": "ONE_TIME",
            "assumptions": ["Assumed 20 hours of effort", "Assumed internal labor rate INR 1500/hr"],
        }],
        "calculation_proposals": [{"calculation_id": "K0", "operation": "MULTIPLY",
                                   "component_ids": ["C0"], "produces": "MOST_LIKELY"}],
        "overall_status": "ASSUMPTION_BASED", "estimability": "ESTIMABLE",
    }
    res = await _run(interp, [_ev("No procedure exists for the task")])
    assert res.status == RemediationEstimateStatus.ASSUMPTION_BASED
    assert res.most_likely_estimate == 30000.0
    assert res.estimate_classification == CostBasis.ASSUMED
    assert any("assumed" in a.lower() for a in res.assumptions)


# --------------------------------------------------------------------------
# 3. No pricing basis at all → NOT_ASSESSABLE with a professional reason
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_pricing_basis_not_assessable_professional_text():
    interp = {
        "strategy": {"remediation_summary": "define and perform the overdue verification"},
        "activities": [{"activity_id": "A0", "description": "Draft the verification procedure", "derived_from": "FINDING"}],
        "cost_components": [{
            "component_id": "C0", "description": "Procedure drafting effort", "cost_category": "labor",
            "quantity_basis": "NOT_ESTABLISHED", "unit_cost_basis": "NOT_ESTABLISHED",
            "amount_type": "COMPONENT", "recurrence": "ONE_TIME",
        }],
        "overall_status": "NOT_ASSESSABLE", "estimability": "NOT_ASSESSABLE",
        "not_assessable_reason": "PRICING_BASIS_UNAVAILABLE",
        "evidence_improves_estimate": ["an internal labor rate", "an effort estimate in hours"],
    }
    res = await _run(interp, [_ev("A required verification was not performed")])
    assert res.status == RemediationEstimateStatus.NOT_ASSESSABLE
    assert res.low_estimate is None and res.most_likely_estimate is None
    assert "cannot be reliably estimated" in res.not_assessable_reason
    for bad in ("LLM", "schema", "parser", "INVALID", "None"):
        assert bad not in res.not_assessable_reason
    # qualitative reasoning is preserved even when no number survives
    assert res.implementation_activities == ["Draft the verification procedure"]
    assert res.evidence_improves_estimate


# --------------------------------------------------------------------------
# 4. Single verified implementation cost preserved (not replaced)
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_verified_cost_preserved():
    interp = {
        "strategy": {"remediation_summary": "replace the failed component"},
        "cost_components": [{
            "component_id": "C0", "description": "Replacement part (already procured)",
            "cost_category": "replacement", "unit_cost": 85000, "unit_cost_basis": "VERIFIED",
            "currency": "USD", "amount_type": "TOTAL", "recurrence": "ONE_TIME",
            "source_reference_ids": ["E0"],
        }],
        "overall_status": "EVIDENCE_BACKED", "estimability": "SINGLE_VERIFIED_COST",
    }
    res = await _run(interp, [_ev("Replacement part purchase order confirms USD 85,000")])
    assert res.low_estimate == res.most_likely_estimate == res.high_estimate == 85000.0
    assert res.estimate_classification == CostBasis.VERIFIED
    assert res.estimation_method == "single verified implementation total preserved"


# --------------------------------------------------------------------------
# 5. Honest failure: provider raises / unparseable output
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_provider_unavailable_fails_closed():
    res = await estimate_remediation_cost(
        finding_text="x", evidence_ledger=[_ev("y")],
        client=FakeLLMClient(raise_exc=RuntimeError("boom")),
    )
    assert res.status == RemediationEstimateStatus.NOT_ASSESSABLE
    assert res.remediation_semantic_status == "LLM_UNAVAILABLE"
    assert "cannot be reliably estimated" in res.not_assessable_reason


@pytest.mark.asyncio
async def test_unparseable_output_fails_closed():
    res = await estimate_remediation_cost(
        finding_text="x", evidence_ledger=[_ev("y")],
        client=FakeLLMClient(response="not json at all {{{"),
    )
    assert res.status == RemediationEstimateStatus.NOT_ASSESSABLE
    assert res.remediation_semantic_status in ("LLM_INVALID", "LLM_UNAVAILABLE")


@pytest.mark.asyncio
async def test_no_evidence_and_no_finding():
    res = await estimate_remediation_cost(finding_text="", evidence_ledger=[], client=FakeLLMClient(response="{}"))
    assert res.status == RemediationEstimateStatus.NOT_ASSESSABLE
    assert res.remediation_semantic_status == "NO_EVIDENCE"


# --------------------------------------------------------------------------
# 7. Partial estimate: some activities priced, others not (spec section 14)
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_partial_estimate_prices_what_it_can():
    interp = {
        "strategy": {"remediation_summary": "replace the unit and revalidate the process"},
        "activities": [
            {"activity_id": "A0", "description": "Procure and install replacement unit", "derived_from": "EVIDENCE"},
            {"activity_id": "A1", "description": "Revalidate the downstream process", "derived_from": "CAPA"},
        ],
        "cost_components": [
            {"component_id": "C0", "description": "Replacement unit + install", "activity_ids": ["A0"],
             "cost_category": "replacement", "unit_cost": 240000, "unit_cost_basis": "REPORTED",
             "currency": "INR", "amount_type": "COMPONENT", "recurrence": "ONE_TIME", "source_reference_ids": ["E0"]},
            {"component_id": "C1", "description": "Process revalidation effort", "activity_ids": ["A1"],
             "cost_category": "validation", "unit_cost_basis": "NOT_ESTABLISHED",
             "amount_type": "COMPONENT", "recurrence": "ONE_TIME"},
        ],
        "overall_status": "EVIDENCE_BACKED",
    }
    res = await _run(interp, [_ev("Supplier quotation for the replacement unit including installation is INR 240,000", EvidenceStatus.REPORTED)])
    assert res.status == RemediationEstimateStatus.EVIDENCE_BACKED
    assert res.most_likely_estimate == 240000.0
    assert res.is_partial_estimate is True
    assert "Process revalidation effort" in res.unpriced_activities
    assert any("could not be priced" in u for u in res.uncertainty_reasons)
    # no fabricated cost for the unpriced activity
    unpriced = [c for c in res.cost_components if c.component_id == "C1"][0]
    assert unpriced.calculated_amount is None


@pytest.mark.asyncio
async def test_additive_components_are_summed_not_ranged_end_to_end():
    """Spec section 23: Component cost = X, Installation cost = Y, both required
    -> one-time = X + Y, low == most_likely == high. Never low=Y, ml=X, high=X+Y."""
    interp = {
        "strategy": {"remediation_summary": "install a monitoring capability"},
        "cost_components": [
            {"component_id": "C0", "description": "monitoring hardware", "cost_category": "equipment",
             "unit_cost": 300000, "unit_cost_basis": "REPORTED", "currency": "INR",
             "amount_type": "COMPONENT", "recurrence": "ONE_TIME", "source_reference_ids": ["E0"]},
            {"component_id": "C1", "description": "installation and commissioning", "cost_category": "installation",
             "unit_cost": 75000, "unit_cost_basis": "REPORTED", "currency": "INR",
             "amount_type": "COMPONENT", "recurrence": "ONE_TIME", "source_reference_ids": ["E0"]},
        ],
        # even an adversarial produces=LOW/HIGH set must not create a range
        "calculation_proposals": [
            {"calculation_id": "K0", "operation": "SUM", "component_ids": ["C1"], "produces": "LOW"},
            {"calculation_id": "K1", "operation": "SUM", "component_ids": ["C0"], "produces": "MOST_LIKELY"},
            {"calculation_id": "K2", "operation": "SUM", "component_ids": ["C0", "C1"], "produces": "HIGH"},
        ],
        "overall_status": "EVIDENCE_BACKED",
    }
    res = await _run(interp, [_ev("Quotation: monitoring hardware INR 300,000; installation and commissioning INR 75,000", EvidenceStatus.REPORTED)])
    assert res.one_time_cost == 375000.0
    assert res.low_estimate == res.most_likely_estimate == res.high_estimate == 375000.0


# --------------------------------------------------------------------------
# 6. report_generator integration: report.remediation_cost is populated,
#    financial_analysis is untouched
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_report_generator_populates_remediation_cost(monkeypatch):
    from app.agent.nodes.report_generator import generate_report_node
    from app.agent.state import AgentState  # noqa: F401
    from app.config import get_settings
    from app.models.agent import AgentTraceStep, InvestigateRequest

    get_settings.cache_clear()
    monkeypatch.setenv("REMEDIATION_COST_ESTIMATION_ENABLED", "true")
    get_settings.cache_clear()

    interp = {
        "strategy": {"remediation_summary": "recalibrate and revalidate", "remediation_type": "modification"},
        "cost_components": [{
            "component_id": "C0", "description": "external calibration service",
            "cost_category": "external services", "unit_cost": 15000, "unit_cost_basis": "REPORTED",
            "currency": "INR", "amount_type": "TOTAL", "recurrence": "ONE_TIME",
            "source_reference_ids": ["E0"],
        }],
        "overall_status": "EVIDENCE_BACKED",
    }
    fake = FakeLLMClient(response=json.dumps(interp))
    monkeypatch.setattr("app.remediation.interpreter.get_llm_client", lambda *a, **k: fake)

    state = {
        "request": InvestigateRequest(finding_text="An instrument was found out of calibration."),
        "evidence_ledger": [_ev("A calibration lab quoted INR 15,000 for recalibration", EvidenceStatus.REPORTED)],
        "trace": [AgentTraceStep.ok("start")],
        "errors": [],
        "iteration_count": 0,
    }
    result = await generate_report_node(state)
    report = result["report"]

    assert report.remediation_cost is not None
    assert report.remediation_cost.most_likely_estimate == 15000.0
    assert report.remediation_cost.status.value == "EVIDENCE_BACKED"
    # financial analysis remains its own independent object
    assert report.financial_analysis is not None
    assert report.remediation_cost is not report.financial_analysis

    get_settings.cache_clear()
