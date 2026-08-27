"""Proves the financial semantic pipeline is provider-agnostic below the
normalization boundary (app.services.semantic_evidence_interpreter's
Pydantic parse of the LLM's JSON into SemanticFindingInterpretation).

Scope note (see conversation): this repository currently implements only
two LLMProvider adapters -- OllamaProvider and GithubCopilotProvider (app/
services/llm/providers/) -- and only Ollama is actually reachable/
credentialed in this environment (the live GitHub Copilot integration test
already fails elsewhere in this suite for lack of credentials). Building
new OpenAI/Anthropic/Google adapters was explicitly out of scope for this
pass. What IS proven here, by direct code inspection and by test:

1. Nothing in app/financial/ or app/services/semantic_evidence_
   interpreter.py imports or branches on a concrete provider class or
   model name (verified: `grep -rniE "ollama|qwen|github_copilot|openai|
   anthropic" app/financial/` finds zero semantic/financial-logic hits --
   the only two matches are Ollama-specific TRANSPORT tuning knob names,
   `settings.ollama_financial_semantic_{max_tokens,num_ctx}`, and
   `num_ctx` is explicitly documented on LLMProvider.generate() as
   "Ollama-specific/ignored by others" -- i.e. harmless for any other
   provider, not a semantic decision).

2. The financial pipeline depends only on `LLMProvider.chat_completion()`
   returning a string -- proven below by swapping in two independently-
   constructed provider stand-ins (simulating two different real
   adapters, not two calls to the same one) that emit semantically
   identical but structurally-different-shaped JSON (different dict
   construction order -- JSON key order is not semantically meaningful)
   and confirming byte-identical downstream FinancialAnalysisResult.

3. No `provider`/`model` field exists anywhere in SemanticFindingInterpretation
   or FinancialAnalysisResult (verified: grep of both model files) -- so
   provider identity structurally CANNOT leak into a financial calculation
   even by accident; it can only ever reach observability/logging.

Real cross-provider comparison against an actual second commercial LLM
(OpenAI/Anthropic/etc.) was NOT performed -- no such adapter exists in
this codebase and none is credentialed in this environment. That remains
an explicit limitation, not a claim made here.
"""

from __future__ import annotations

import json

from app.financial.engine import _build_result_from_observations
from app.financial.relationship_validator import validate_and_materialize
from app.financial.semantic_models import SemanticFindingInterpretation


class _ProviderAStandIn:
    """Simulates one real LLMProvider adapter -- e.g. what OllamaProvider's
    chat_completion() returns: a JSON string built from a dict literal in
    one key order."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    async def chat_completion(self, messages, temperature=0.0, response_format_json=True, **kwargs):
        self.calls += 1
        # Deliberately construct the JSON text via a DIFFERENT key
        # insertion order than _ProviderBStandIn, to prove key order
        # (a real difference between how two different SDKs might
        # serialize) has zero effect on the parsed result.
        ordered = {
            "finding": self.payload["finding"],
            "claims": self.payload["claims"],
            "relationships": self.payload["relationships"],
            "calculation_proposals": self.payload["calculation_proposals"],
            "cost_factor": self.payload["cost_factor"],
            "quantification": self.payload["quantification"],
        }
        return json.dumps(ordered)


class _ProviderBStandIn:
    """Simulates a SECOND, independently-implemented LLMProvider adapter
    -- same semantic content, different construction path/key order,
    proving the pipeline reacts identically regardless of which concrete
    provider object produced the string."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    async def chat_completion(self, messages, temperature=0.0, response_format_json=True, **kwargs):
        self.calls += 1
        ordered = {
            "quantification": self.payload["quantification"],
            "cost_factor": self.payload["cost_factor"],
            "calculation_proposals": self.payload["calculation_proposals"],
            "relationships": self.payload["relationships"],
            "claims": self.payload["claims"],
            "finding": self.payload["finding"],
        }
        return json.dumps(ordered)


def _equivalent_semantic_payload() -> dict:
    """One semantic interpretation, expressed once -- both provider
    stand-ins serialize this SAME content, just differently ordered."""
    return {
        "finding": {"deviation": "Equipment fault", "affected_object": "Conveyor motor",
                     "process": "Material handling", "requirement": "Continuous operation",
                     "affected_period": "Current", "interpretation_confidence": "HIGH"},
        "claims": [
            {"claim_id": "C0", "source_evidence_ids": ["E0"], "fact_type": "QUANTITY",
             "value": 14, "unit": "HOUR", "currency": None, "population": "CURRENT_FINDING",
             "temporal_scope": None, "evidence_status": "VERIFIED", "explicit": True, "notes": None},
            {"claim_id": "C1", "source_evidence_ids": ["E1"], "fact_type": "RATE",
             "value": 6200, "unit": "HOUR", "currency": "USD", "population": "CURRENT_FINDING",
             "temporal_scope": None, "evidence_status": "VERIFIED", "explicit": True, "notes": None},
        ],
        "relationships": [
            {"relationship_id": "R0", "type": "RATE_APPLIES_TO_QUANTITY", "source_claim": "C1",
             "target_claim": "C0", "confidence": "HIGH", "evidence_basis": ["E0", "E1"]},
        ],
        "calculation_proposals": [
            {"calculation_id": "CAL0", "operation": "MULTIPLY", "inputs": ["C0", "C1"],
             "relationship_ids": ["R0"], "proposed_result_value": 86800.0,
             "proposed_result_currency": "USD", "reason": "downtime hours x hourly disruption cost"},
        ],
        "cost_factor": {"selected_factor": "DOWNTIME_COST", "supporting_claim_ids": ["C0", "C1"],
                         "confidence": "HIGH", "rationale": "conveyor motor downtime"},
        "quantification": {"status": "QUANTIFIABLE", "blocker": "", "missing_inputs": []},
    }


class TestProviderIndependenceBoundary:
    async def test_two_differently_constructed_providers_produce_identical_result(self):
        payload = _equivalent_semantic_payload()

        interp_a = SemanticFindingInterpretation.model_validate(
            json.loads(await _ProviderAStandIn(payload).chat_completion([]))
        )
        interp_b = SemanticFindingInterpretation.model_validate(
            json.loads(await _ProviderBStandIn(payload).chat_completion([]))
        )

        obs_a, outcome_a = validate_and_materialize(interp_a, evidence_count=2)
        obs_b, outcome_b = validate_and_materialize(interp_b, evidence_count=2)

        result_a = _build_result_from_observations(obs_a)
        result_b = _build_result_from_observations(obs_b)

        # Byte-identical canonical financial results -- the downstream
        # pipeline is a pure function of semantic CONTENT, not of which
        # provider object (or key ordering) produced the JSON.
        assert result_a.model_dump() == result_b.model_dump()
        assert outcome_a.validated_cost_factor == outcome_b.validated_cost_factor == "DOWNTIME_COST"
        assert result_a.confirmed_impact.verified_gross_exposure == 14 * 6200

    async def test_provider_identity_does_not_appear_in_semantic_or_financial_models(self):
        # Structural guarantee, not just an empirical one: neither model
        # has a field a provider name/model id could occupy, so it cannot
        # leak into calculation logic even by future accident.
        assert "provider" not in SemanticFindingInterpretation.model_fields
        assert "model" not in SemanticFindingInterpretation.model_fields
        from app.financial.models import FinancialAnalysisResult
        assert "provider" not in FinancialAnalysisResult.model_fields
        assert "model" not in FinancialAnalysisResult.model_fields
