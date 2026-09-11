"""Pass 62 (spec §22/§23/§E): AI provenance must survive the serialization
boundary Python -> FastAPI response JSON, on every result path, and must
identify the model/prompt/schema actually used at generation time.

The ASP.NET DTO / database / retrieval leg is OUTSIDE this repository and is
recorded as NOT VERIFIED in the report -- this test proves the JSON contract
the ASP.NET side consumes.
"""
from __future__ import annotations

import json

import pytest

from app.models.agent import EvidenceItem, EvidenceStatus
from app.remediation.engine import estimate_remediation_cost, honest_not_assessable

_REQUIRED_KEYS = {
    "remediation_model", "remediation_prompt_version", "canonical_model",
    "canonical_prompt_version", "analysis_prompt_version", "provider",
    "semantic_schema_version", "remediation_semantic_status", "reasoning_source",
    "review_required", "pricing_status", "generated_at",
}


class _FakeLLM:
    def __init__(self, payload):
        self._p = payload if isinstance(payload, str) else json.dumps(payload)

    async def chat_completion(self, messages, **kw):
        return self._p


def _priced_interp():
    return {
        "strategy": {"remediation_summary": "replace the failed part",
                     "remediation_type": "corrective", "interpretation_confidence": "HIGH"},
        "activities": [{"activity_id": "A0", "description": "Replace the failed part",
                        "derived_from": "FINDING"}],
        "cost_components": [{
            "component_id": "C0", "description": "Replacement part", "activity_ids": ["A0"],
            "cost_category": "parts", "value_kind": "REMEDIATION_COST",
            "quantity": 3, "quantity_unit": "part", "quantity_basis": "EVIDENCED",
            "unit_cost": 4000, "unit_cost_basis": "VERIFIED", "currency": "INR",
            "amount_type": "PER_UNIT", "recurrence": "ONE_TIME", "source_reference_ids": ["E0"],
        }],
        "calculation_proposals": [], "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
    }


def _serialize_through_api(rc) -> dict:
    """Round-trip the result through JSON exactly as FastAPI serialises the
    nested `report.remediation_cost` object in the /investigate response."""
    from app.models.agent import RemediationCostResult
    raw = json.loads(rc.model_dump_json())
    # re-parse to prove the field survives a full model round trip
    back = RemediationCostResult.model_validate(raw)
    assert isinstance(back.ai_provenance, dict) and back.ai_provenance
    # and that the response schema exposes it to the ASP.NET consumer
    assert "ai_provenance" in RemediationCostResult.model_json_schema()["properties"]
    return {"report": {"remediation_cost": raw}}


@pytest.mark.asyncio
async def test_provenance_survives_api_serialization_priced_path():
    rc = await estimate_remediation_cost(
        finding_text="Three parts require replacement at INR 4,000 each.",
        evidence_ledger=[EvidenceItem(claim="3 parts at INR 4000", status=EvidenceStatus.VERIFIED, source="t")],
        client=_FakeLLM(_priced_interp()),
    )
    body = _serialize_through_api(rc)
    prov = body["report"]["remediation_cost"]["ai_provenance"]
    assert _REQUIRED_KEYS <= set(prov), _REQUIRED_KEYS - set(prov)
    assert prov["pricing_status"] == "EXACT_ESTIMATE"
    assert prov["review_required"] is True
    assert prov["generated_at"].endswith("+00:00")  # explicit UTC


@pytest.mark.asyncio
async def test_provenance_present_on_provider_failure_path():
    rc = honest_not_assessable("LLM_TIMEOUT", "MODEL_TIMEOUT")
    body = _serialize_through_api(rc)
    prov = body["report"]["remediation_cost"]["ai_provenance"]
    assert _REQUIRED_KEYS <= set(prov)
    assert prov["remediation_semantic_status"] == "LLM_TIMEOUT"
    assert prov["review_required"] is True


@pytest.mark.asyncio
async def test_provenance_records_the_configured_override_model(monkeypatch):
    from app.config import get_settings
    s = get_settings()
    monkeypatch.setattr(s, "remediation_cost_model", "some-stronger-model:99b")
    rc = await estimate_remediation_cost(
        finding_text="Three parts require replacement at INR 4,000 each.",
        evidence_ledger=[EvidenceItem(claim="3 parts at INR 4000", status=EvidenceStatus.VERIFIED, source="t")],
        client=_FakeLLM(_priced_interp()),
    )
    body = _serialize_through_api(rc)
    prov = body["report"]["remediation_cost"]["ai_provenance"]
    assert prov["remediation_model"] == "some-stronger-model:99b"


def test_provenance_carries_no_chain_of_thought():
    rc = honest_not_assessable("OK", "PRICING_BASIS_UNAVAILABLE")
    prov = rc.ai_provenance
    # keys are provenance identifiers only -- no reasoning transcript
    assert set(prov) == _REQUIRED_KEYS
    blob = " ".join(str(v) for v in prov.values()).lower()
    for banned in ("chain_of_thought", "step 1", "because ", "i think", "let me"):
        assert banned not in blob, banned
