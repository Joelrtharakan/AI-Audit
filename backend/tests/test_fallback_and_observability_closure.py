"""Pass 63 (spec §7/§11/§18/§21): operational observability + fallback
certification. Structured fixtures only -- no live model, no finding-text
parsing.
"""
from __future__ import annotations

import json

import pytest

from app.models.agent import EvidenceItem, EvidenceStatus
from app.remediation.engine import estimate_remediation_cost, honest_not_assessable
from app.services import llm_metrics


class _FakeLLM:
    def __init__(self, p): self._p = p if isinstance(p, str) else json.dumps(p)
    async def chat_completion(self, m, **k): return self._p


class _Boom:
    def __init__(self, exc): self._e = exc
    async def chat_completion(self, m, **k): raise self._e


def _ev(c): return EvidenceItem(claim=c, status=EvidenceStatus.VERIFIED, source="t")


_PRICED = {
    "strategy": {"remediation_summary": "replace the part", "remediation_type": "corrective",
                 "interpretation_confidence": "HIGH"},
    "activities": [{"activity_id": "A0", "description": "Replace the part", "derived_from": "FINDING"}],
    "cost_components": [{
        "component_id": "C0", "description": "Replacement part", "activity_ids": ["A0"],
        "cost_category": "parts", "value_kind": "REMEDIATION_COST",
        "quantity": 2, "quantity_unit": "part", "quantity_basis": "EVIDENCED",
        "unit_cost": 5000, "unit_cost_basis": "VERIFIED", "currency": "INR",
        "amount_type": "PER_UNIT", "recurrence": "ONE_TIME", "source_reference_ids": ["E0"],
    }],
    "calculation_proposals": [], "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
}


# --- §7/§18: semantic-reliability metrics -----------------------------------
@pytest.mark.asyncio
async def test_semantic_reliability_metrics_are_recorded_once_per_result():
    before = llm_metrics.snapshot().get("remediation_runs_total", 0)
    await estimate_remediation_cost(finding_text="x", evidence_ledger=[_ev("2 parts at INR 5000")],
                                    client=_FakeLLM(_PRICED))
    r2 = honest_not_assessable("LLM_TIMEOUT", "MODEL_TIMEOUT")
    from app.remediation.engine import _enforce_result_consistency
    _enforce_result_consistency(r2)  # a re-stamp must NOT re-count
    after = llm_metrics.snapshot()
    assert after["remediation_runs_total"] == before + 2  # priced + timeout, not 3
    assert after["remediation_model_timeout"] >= 1


@pytest.mark.asyncio
async def test_metrics_never_contain_finding_text():
    await estimate_remediation_cost(
        finding_text="A very distinctive SECRET_FINDING_STRING_XYZ occurred.",
        evidence_ledger=[_ev("2 parts at INR 5000")], client=_FakeLLM(_PRICED))
    blob = json.dumps(llm_metrics.aggregated())
    assert "SECRET_FINDING_STRING_XYZ" not in blob
    assert "remediation" in json.loads(blob)  # the new sub-object exists


# --- §11/§21: fallback certification --------------------------------------
@pytest.mark.asyncio
async def test_provider_failure_result_is_explicit_and_review_gated():
    for exc_label, exc in [("timeout", TimeoutError("timed out")),
                           ("provider", RuntimeError("cannot connect"))]:
        rc = await estimate_remediation_cost(
            finding_text="x", evidence_ledger=[_ev("y")], client=_Boom(exc))
        assert str(rc.status) .endswith("NOT_ASSESSABLE") or rc.status.name == "NOT_ASSESSABLE"
        assert rc.review_required is True
        assert rc.one_time_cost is None and rc.recurring_cost is None
        # provenance identifies the failure -- not "no evidence"
        assert rc.ai_provenance["remediation_semantic_status"] in ("LLM_UNAVAILABLE", "LLM_TIMEOUT", "LLM_INVALID")
        assert "not an evidence gap" in rc.not_assessable_reason.lower() or \
               "system-availability" in rc.not_assessable_reason.lower() or \
               "did not respond" in rc.not_assessable_reason.lower()


@pytest.mark.asyncio
async def test_fallback_never_produces_a_trusted_headline_without_evidence():
    # canonical unavailable + NO priceable remediation evidence -> the
    # deterministic scope floor must not manufacture a number.
    rc = await estimate_remediation_cost(
        finding_text="A control weakness was identified.",
        evidence_ledger=[_ev("A control weakness was identified.")],
        client=_FakeLLM({"strategy": {"remediation_summary": "", "interpretation_confidence": "LOW"},
                         "activities": [], "cost_components": [], "calculation_proposals": [],
                         "overall_status": "NOT_ASSESSABLE", "estimability": "NOT_ASSESSABLE"}))
    assert rc.one_time_cost is None
    assert rc.review_required is True
    assert rc.ai_provenance  # provenance always present


# --- §15: model swappability (config-only) --------------------------------
def test_configured_model_override_needs_no_code_change(monkeypatch):
    from app.services.llm_client import get_llm_client
    default = get_llm_client().config.model
    over = get_llm_client(model="stronger-model:70b").config.model
    assert over == "stronger-model:70b" and over != default
    empty = get_llm_client(model="").config.model
    assert empty == default  # empty override -> global, never a silent 3rd model
