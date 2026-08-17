from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from app.agent.nodes.core_synthesis import (
    _classify_failure,
    core_synthesis_node,
    parse_core_synthesis_output,
)
from app.models.agent import (
    CanonicalFindingState,
    CoreSynthesisOutput,
    EvidenceItem,
    EvidenceStatus,
    InvestigateRequest,
    RootCauseStatus,
)
from app.services.llm_client import LLMNetworkError, LLMTimeoutError
from app.services.llm_json import extract_json_str, parse_llm_json


def test_valid_json_extraction():
    raw = '{"root_cause": {"status": "NOT_ESTABLISHED", "category": "TO_BE_CONFIRMED"}}'
    parsed = parse_llm_json(raw)
    assert parsed["root_cause"]["status"] == "NOT_ESTABLISHED"


def test_fenced_json_extraction():
    raw = '```json\n{"root_cause": {"status": "NOT_ESTABLISHED", "category": "TO_BE_CONFIRMED"}}\n```'
    parsed = parse_llm_json(raw)
    assert parsed["root_cause"]["status"] == "NOT_ESTABLISHED"


def test_prose_surrounded_json_extraction():
    raw = 'Here is the requested output:\n```json\n{"root_cause": {"status": "NOT_ESTABLISHED", "category": "TO_BE_CONFIRMED"}}\n```\nHope this helps!'
    parsed = parse_llm_json(raw)
    assert parsed["root_cause"]["status"] == "NOT_ESTABLISHED"


def test_think_tag_surrounded_json():
    raw = '<think>Let me reason about this carefully...</think>\n{"root_cause": {"status": "NOT_ESTABLISHED", "category": "TO_BE_CONFIRMED"}}'
    parsed = parse_llm_json(raw)
    assert parsed["root_cause"]["status"] == "NOT_ESTABLISHED"


def test_trailing_comma_json_extraction():
    raw = '{"root_cause": {"status": "NOT_ESTABLISHED", "category": "TO_BE_CONFIRMED",}, "five_why": {"steps": [],},}'
    parsed = parse_llm_json(raw)
    assert parsed["root_cause"]["status"] == "NOT_ESTABLISHED"


def test_empty_response_raises_value_error():
    with pytest.raises(ValueError):
        parse_llm_json("")


def test_malformed_json_raises_json_decode_error():
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json("Not a JSON object at all")


def test_parse_core_synthesis_output_valid():
    raw = json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED",
            "candidate_hypotheses": [
                {
                    "id": "H1",
                    "name": "TEST_HYP",
                    "statement": "Test statement.",
                    "supporting_claim_ids": ["C1"],
                    "status": "POSSIBLE",
                }
            ],
        },
        "five_why": {
            "steps": [
                {
                    "level": 1,
                    "question": "Why did X happen?",
                    "answer": "Because Y was not verified.",
                    "status": "UNKNOWN",
                }
            ]
        },
        "contributing_factors": [],
    })
    parsed_dict, model = parse_core_synthesis_output(raw)
    assert isinstance(model, CoreSynthesisOutput)
    assert len(model.root_cause.candidate_hypotheses) == 1
    assert model.root_cause.candidate_hypotheses[0].id == "H1"


def test_classify_failure_taxonomy():
    assert _classify_failure(LLMTimeoutError("timeout"), {}) == "TIMEOUT"
    assert _classify_failure(LLMNetworkError("network"), {}) == "PROVIDER_FAILURE"
    assert _classify_failure(json.JSONDecodeError("err", "doc", 0), {}) == "JSON_PARSE_ERROR"
    assert _classify_failure(ValueError("empty"), {}) == "JSON_PARSE_ERROR"
    assert _classify_failure(None, {"hit_output_limit": True}) == "OUTPUT_TRUNCATED"


@pytest.mark.asyncio
async def test_core_synthesis_partial_hypothesis_rejection_preserves_valid_synthesis():
    """When LLM returns one valid hypothesis and one hypothesis with invalid provenance,
    the invalid hypothesis is rejected, but analysis_mode remains LLM (not deterministic fallback)."""
    finding_text = "The weekly verification record was incomplete for the reporting period."
    ledger = [
        EvidenceItem(claim="the weekly verification record was incomplete", source="finding_text", status=EvidenceStatus.VERIFIED),
    ]

    llm_payload = json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED",
            "candidate_hypotheses": [
                {
                    "id": "H1",
                    "name": "VALID_HYPOTHESIS",
                    "statement": "The weekly verification record was not completed during the shift.",
                    "supporting_claim_ids": ["C1"],
                    "status": "POSSIBLE",
                    "evidence_needed": "Weekly verification log",
                },
                {
                    "id": "H2",
                    "name": "INVALID_PROVENANCE_HYPOTHESIS",
                    "statement": "The operator forgot to sign because of bad training.",
                    "supporting_claim_ids": ["C999"],  # Non-existent claim
                    "status": "POSSIBLE",
                    "evidence_needed": "Training records",
                },
            ],
            "narrative": "The weekly verification record was incomplete.",
        },
        "five_why": {
            "steps": [
                {
                    "level": 1,
                    "question": "Why was the weekly verification record incomplete?",
                    "answer": "The available evidence does not establish why the record was incomplete.",
                    "status": "UNKNOWN",
                }
            ],
            "is_complete": False,
            "status_note": "evidence boundary reached",
        },
        "contributing_factors": [],
    })

    client_mock = AsyncMock()
    client_mock.chat_completion.return_value = llm_payload

    with patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=client_mock), \
         patch("app.services.ollama_client.get_last_call_metadata", return_value={"eval_count": 100, "hit_output_limit": False}):
        state = {
            "request": InvestigateRequest(finding_text=finding_text),
            "evidence_ledger": ledger,
            "trace": [],
            "canonical_finding_state": CanonicalFindingState(
                raw_finding=finding_text,
                finding_subject="weekly verification record",
                observed_deviation="weekly verification record was incomplete",
            ),
        }
        res = await core_synthesis_node(state)

        print("TRACE:", [t.message for t in res["trace"]])
        print("ERRORS:", res.get("errors"))
        print("EXECUTION:", res.get("synthesis_execution"))
        assert res["analysis_mode"] == "LLM"
        assert res["synthesis_execution"]["source"] == "PRIMARY_LLM"
        # H2 was dropped due to invalid provenance, H1 survived
        hyps = res["root_cause"].candidate_hypotheses
        assert len(hyps) == 1
        assert hyps[0].id == "H1"
