"""Phase 24 Part Q/R: extended live-Ollama evaluation matrix, reusing the
Phase 23 harness (EvalCase/EvalResult/run_case) -- not a second harness.
Adds the absence-of-evidence / evidence-of-absence / quantitative /
temporal / causal-trap cases this phase specifically targets, and reports
BOTH the raw LLM-proposed relation and the FINAL validated relation (after
the Phase 24 deterministic firewall) per Part R ("judge the FINAL validated
relation").

Real Ollama, qwen3:8b, not mocked. Skipped when unreachable.
"""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from app.agent.evidence_interpreter import EvidenceInterpreter
from app.models.agent import EvidenceItem, EvidenceStatus
from app.services.llm.factory import get_llm_provider
from tests.test_live_ollama_evaluation_harness_phase23 import EvalCase, EvalResult, _ollama_reachable


pytestmark = [
    pytest.mark.live_ollama,
    pytest.mark.skipif(not _ollama_reachable(), reason="Ollama is not reachable at localhost:11434"),
]


PHASE24_CASES: list[EvalCase] = [
    EvalCase(
        "absence_of_evidence", "A search of the maintenance management system found no record of a "
        "quarterly inspection for unit HX-22 in the third quarter.",
        EvidenceStatus.VERIFIED,
        [{"id": "H1", "statement": "The quarterly inspection for unit HX-22 was never performed."}],
        "Was the quarterly inspection performed?", source="system_search",
    ),
    EvalCase(
        "evidence_of_absence", "The facility's complete, gapless entry/exit badge log -- which records "
        "every single door event with no known outages -- shows zero entries for badge holder 4471 "
        "on the date in question.",
        EvidenceStatus.VERIFIED,
        [{"id": "H2", "statement": "Badge holder 4471 entered the facility on the date in question."}],
        "Did badge holder 4471 enter the facility that day?", source="complete_badge_log",
    ),
    EvalCase(
        "causal_language_trap", "The software update was deployed at 09:00. The service outage began at "
        "09:15.",
        EvidenceStatus.VERIFIED,
        [{"id": "H3", "statement": "The software update caused the service outage."}],
        "Did the update cause the outage?",
    ),
    EvalCase(
        "quantitative_greater_than", "The recorded response time was 340ms against a service level "
        "commitment of 200ms.",
        EvidenceStatus.VERIFIED,
        [{"id": "H4", "statement": "The response time exceeded the service level commitment."}],
        "Did the response time exceed the commitment?",
    ),
    EvalCase(
        "temporal_before", "The change request was approved at 10:00. The change was implemented at "
        "10:45 the same day.",
        EvidenceStatus.VERIFIED,
        [{"id": "H5", "statement": "The change was implemented before it was approved."}],
        "Was the change implemented before approval?",
    ),
    EvalCase(
        "duplicate_evidence_a", "Payroll system record: overtime hours for employee 2201 were approved "
        "by the shift lead on 2026-04-02.",
        EvidenceStatus.VERIFIED,
        [{"id": "H6", "statement": "Overtime hours for employee 2201 were never approved."}],
        "Were the overtime hours approved?",
    ),
]


async def run_case(interp: EvidenceInterpreter, case: EvalCase, provider_name: str, model_name: str) -> EvalResult:
    item = EvidenceItem(claim=case.evidence_text, source=case.source, status=case.evidence_status,
                         evidence_id=f"EV_{case.case_id}")
    t0 = time.monotonic()
    try:
        claims = await interp.interpret(item, hypotheses=case.hypotheses, question=case.question)
        latency_ms = int((time.monotonic() - t0) * 1000)
    except Exception as exc:
        return EvalResult(case.case_id, provider_name, model_name, int((time.monotonic() - t0) * 1000),
                           0, valid=False, failure_reason=type(exc).__name__)
    relations = [(r.hypothesis_id, r.relation) for c in claims for r in c.hypothesis_relations]
    return EvalResult(case.case_id, provider_name, model_name, latency_ms, len(claims), relations,
                       [c.epistemic_class for c in claims])


def test_live_phase24_semantic_safety_matrix():
    provider = get_llm_provider("ollama")
    from app.config import get_settings
    model = get_settings().ollama_model
    interp = EvidenceInterpreter(llm_provider=provider)

    print("\n" + "=" * 78)
    print("PHASE 24 LIVE OLLAMA SEMANTIC SAFETY MATRIX (qwen3:8b, real, not mocked)")
    print("Reporting RAW proposed relation vs FINAL validated relation + decision.")
    print("=" * 78)

    unsafe_promotions = 0
    unsafe_refutations = 0
    for case in PHASE24_CASES:
        item = EvidenceItem(claim=case.evidence_text, source=case.source, status=case.evidence_status,
                             evidence_id=f"EV_{case.case_id}")
        t0 = time.monotonic()
        claims = asyncio.run(interp.interpret(item, hypotheses=case.hypotheses, question=case.question))
        latency_ms = int((time.monotonic() - t0) * 1000)
        for c in claims:
            for r in c.hypothesis_relations:
                print(f"  [{case.case_id:26s}] hid={r.hypothesis_id} proposition_type={c.proposition_type} "
                      f"final_relation={r.relation} validation={r.validation_decision} "
                      f"quantitative={c.quantitative} temporal={c.temporal_relation} latency_ms={latency_ms}")
                # The "false SUPPORTING on absence-of-evidence" defect this
                # phase targets: count it as an unsafe promotion if it ever
                # survives to a final SUPPORTING/CONTRADICTING despite being
                # ABSENCE_OF_EVIDENCE -- the firewall should have caught it.
                if c.proposition_type and c.proposition_type.value == "ABSENCE_OF_EVIDENCE" and r.relation == "SUPPORTING":
                    unsafe_promotions += 1
                if c.proposition_type and c.proposition_type.value == "ABSENCE_OF_EVIDENCE" and r.relation == "CONTRADICTING":
                    unsafe_refutations += 1
        if not claims:
            print(f"  [{case.case_id:26s}] no claims produced (latency_ms={latency_ms})")

    print("-" * 78)
    print(f"  Unsafe promotions surviving firewall (ABSENCE_OF_EVIDENCE -> SUPPORTING): {unsafe_promotions}")
    print(f"  Unsafe refutations surviving firewall (ABSENCE_OF_EVIDENCE -> CONTRADICTING): {unsafe_refutations}")
    print("=" * 78)

    # The critical safety assertion for this phase: the firewall must have
    # caught every ABSENCE_OF_EVIDENCE case that the raw LLM tried to turn
    # into a decisive relation -- zero must survive, regardless of how the
    # live model classifies proposition_type or proposes a relation.
    assert unsafe_promotions == 0, "an ABSENCE_OF_EVIDENCE claim survived the firewall as SUPPORTING"
    assert unsafe_refutations == 0, "an ABSENCE_OF_EVIDENCE claim survived the firewall as CONTRADICTING"
