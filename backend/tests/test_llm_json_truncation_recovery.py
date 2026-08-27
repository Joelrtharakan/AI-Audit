"""Truncation-tolerant recovery in the provider-neutral JSON parser.

A response cut off by the model's output-token limit is a real,
provider-independent failure mode (architecture spec section 21): the
model's semantics can be perfectly correct while the JSON simply stopped
mid-structure. Before this, ANY such response raised -> LLM_INVALID -> the
entire financial analysis was discarded. Now the parser salvages the
complete prefix and the downstream schema + validator treat it as a
partial interpretation.

Genuinely malformed (non-truncation) output must still raise -- the
recovery only accepts a candidate that kept most of the original text and
at least one field.
"""

from __future__ import annotations

import json

import pytest

from app.services.llm.json_parser import parse_llm_json

_WELL_FORMED = json.dumps(
    {
        "finding": {"deviation": "rework", "interpretation_confidence": "HIGH"},
        "claims": [
            {"claim_id": "C0", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY", "value": 1000, "unit": "unit", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "C1", "source_evidence_ids": ["E1"], "fact_type": "RATE", "value": 250, "unit": "unit", "currency": "INR", "population": "CURRENT_FINDING", "evidence_status": "VERIFIED"},
            {"claim_id": "C2", "source_evidence_ids": ["E2"], "fact_type": "RECOVERY", "value": 40000, "currency": "INR", "population": "RECOVERY", "evidence_status": "REPORTED"},
        ],
        "relationships": [
            {"relationship_id": "R0", "relationship_type": "per-unit rate", "source_claim": "C1", "target_claim": "C0", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]},
        ],
        "calculation_proposals": [
            {"calculation_id": "CAL0", "operation": "MULTIPLY", "inputs": ["C0", "C1"], "relationship_ids": ["R0"], "proposed_result_value": 250000, "proposed_result_currency": "INR", "reason": "rate applies to quantity"},
        ],
        "cost_factor": {"selected_factor": "REWORK_COST", "supporting_claim_ids": ["C0", "C1"], "confidence": "HIGH"},
        "financial_relevance": "CONFIRMED",
        "quantification": {"status": "QUANTIFIED"},
    },
    indent=2,
)


@pytest.mark.parametrize("cut", list(range(120, len(_WELL_FORMED) + 1, 23)))
def test_every_truncation_point_recovers_to_valid_dict(cut: int):
    out = parse_llm_json(_WELL_FORMED[:cut])
    assert isinstance(out, dict)
    # whatever survived must itself be structurally coherent
    json.dumps(out)


def test_recovered_prefix_preserves_leading_claims():
    # cut partway through the third claim -> first two claims must survive intact
    cut = _WELL_FORMED.index('"C2"') - 5
    out = parse_llm_json(_WELL_FORMED[:cut])
    ids = [c.get("claim_id") for c in out.get("claims", [])]
    assert ids[:2] == ["C0", "C1"]
    assert out["claims"][0]["value"] == 1000


def test_full_response_is_untouched():
    assert parse_llm_json(_WELL_FORMED) == json.loads(_WELL_FORMED)


@pytest.mark.parametrize("junk", ["", "not json", "{bad", "[1,2,3]", '{"a":}', "Sorry, I cannot help."])
def test_non_truncation_garbage_still_raises(junk: str):
    with pytest.raises(Exception):
        parse_llm_json(junk)
