"""Phase 22 Part L/S: full live-Ollama validation matrix. Every test in this
file calls a REAL local Ollama server (qwen3:8b, http://localhost:11434)
through the real production EvidenceInterpreter / provider-factory path --
nothing here is mocked. Skipped (never faked) when Ollama is unreachable.

Each scenario asserts only STRUCTURAL correctness (provenance, valid
epistemic_class, capped status) since live model wording is not scripted --
consistent with Phase 21's live test and this phase's "do not require exact
natural-language wording" instruction.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from app.agent.evidence_interpreter import EvidenceInterpreter, derive_hypothesis_relevance
from app.agent.graph import build_agent_graph
from app.agent.invariants import evaluate_all_invariants
from app.models.agent import AgentTraceStep, EvidenceItem, EvidenceStatus, InvestigateRequest
from app.services.evidence_provider import EvidenceProvider
from app.services.llm.factory import get_llm_provider


def _ollama_reachable() -> bool:
    try:
        return httpx.get("http://localhost:11434/api/tags", timeout=2.0).status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.live_ollama,
    pytest.mark.skipif(not _ollama_reachable(), reason="Ollama is not reachable at localhost:11434"),
]


def _interp() -> EvidenceInterpreter:
    return EvidenceInterpreter(llm_provider=get_llm_provider("ollama"))


def _assert_structurally_valid(claims, evidence_id):
    for c in claims:
        assert c.evidence_ids == [evidence_id]
        assert c.epistemic_class in {"SUPPORTING", "CONTRADICTING", "OBSERVED", "REPORTED", "UNKNOWN"}
        assert c.extraction_status == "EXTRACTED"
    return claims


# 1. Objective supporting evidence (domain: equipment calibration)
def test_live_objective_supporting_evidence():
    item = EvidenceItem(
        claim="Calibration system record: pressure gauge PG-14 was calibrated on 2026-02-01, "
              "certificate CAL-4471, valid through 2027-02-01.",
        source="calibration_system", status=EvidenceStatus.VERIFIED, evidence_id="EV1",
    )
    claims = asyncio.run(_interp().interpret(
        item, hypothesis_statement="The pressure gauge was used while out of calibration.",
        hypothesis_id="H1", question="Was the gauge in calibration at the time of use?",
    ))
    _assert_structurally_valid(claims, "EV1")
    print(f"\n[1 supporting] {[(c.epistemic_class, c.status) for c in claims]}")


# 2. Objective contradicting evidence (domain: financial approval)
def test_live_objective_contradicting_evidence():
    item = EvidenceItem(
        claim="Finance system record: purchase order PO-8821 for $42,000 shows no manager "
              "approval signature field completed before the payment was released.",
        source="finance_system", status=EvidenceStatus.VERIFIED, evidence_id="EV2",
    )
    claims = asyncio.run(_interp().interpret(
        item, hypothesis_statement="The purchase order was properly approved before payment.",
        hypothesis_id="H2", question="Was manager approval obtained before payment release?",
    ))
    _assert_structurally_valid(claims, "EV2")
    print(f"\n[2 contradicting] {[(c.epistemic_class, c.status) for c in claims]}")


# 3. Conflicting evidence (domain: shipment receipt) -- across two items
def test_live_conflicting_evidence_across_two_items():
    item_a = EvidenceItem(
        claim="Warehouse receiving log shows shipment SH-330 was received and signed for on 2026-03-02.",
        source="receiving_log", status=EvidenceStatus.VERIFIED, evidence_id="EV3A",
    )
    item_b = EvidenceItem(
        claim="The site supervisor stated the shipment was never received at the dock.",
        source="supervisor_statement", status=EvidenceStatus.REPORTED, evidence_id="EV3B",
    )
    interp = _interp()
    claims_a = asyncio.run(interp.interpret(item_a, "The shipment was never received.", "H3"))
    claims_b = asyncio.run(interp.interpret(item_b, "The shipment was never received.", "H3"))
    _assert_structurally_valid(claims_a, "EV3A")
    _assert_structurally_valid(claims_b, "EV3B")
    print(f"\n[3 conflicting] a={[(c.epistemic_class) for c in claims_a]} b={[(c.epistemic_class) for c in claims_b]}")


# 4. Reported testimony (domain: laboratory sample handling)
def test_live_reported_testimony():
    item = EvidenceItem(
        claim="The lab technician stated that the sample was refrigerated immediately after collection.",
        source="interview_notes", status=EvidenceStatus.REPORTED, evidence_id="EV4",
    )
    claims = asyncio.run(_interp().interpret(
        item, hypothesis_statement="The sample was not refrigerated in time, compromising results.",
        hypothesis_id="H4",
    ))
    _assert_structurally_valid(claims, "EV4")
    # REPORTED evidence item must NEVER produce a VERIFIED claim.
    assert all(c.status != EvidenceStatus.VERIFIED for c in claims)
    print(f"\n[4 reported] {[(c.epistemic_class, c.status) for c in claims]}")


# 5. Missing evidence (domain: vendor qualification) -- empty/unavailable item
def test_live_missing_evidence_produces_no_fabricated_claim():
    item = EvidenceItem(claim="", source="none", status=EvidenceStatus.UNVERIFIED, evidence_id="EV5")
    claims = asyncio.run(_interp().interpret(
        item, hypothesis_statement="The vendor was qualified before the first purchase order.",
        hypothesis_id="H5",
    ))
    assert claims == [], "empty evidence text must never produce a fabricated claim, live model or not"


# 6. Ambiguous evidence (domain: incident timeline)
def test_live_ambiguous_evidence():
    item = EvidenceItem(
        claim="Security log shows badge access to the server room around the time of the incident, "
              "but the exact minute is unclear due to a clock synchronization issue noted in the log header.",
        source="security_log", status=EvidenceStatus.VERIFIED, evidence_id="EV6",
    )
    claims = asyncio.run(_interp().interpret(
        item, hypothesis_statement="An unauthorized person accessed the server room during the incident window.",
        hypothesis_id="H6",
    ))
    _assert_structurally_valid(claims, "EV6")
    print(f"\n[6 ambiguous] {[(c.epistemic_class, c.qualifier if hasattr(c,'qualifier') else None) for c in claims]}")


# 7. Multiple claims in one evidence item (domain: maintenance workorder)
def test_live_multiple_claims_in_one_item():
    item = EvidenceItem(
        claim="Workorder WO-991 record: preventive maintenance was completed on the conveyor motor on "
              "2026-01-10 by technician R. Alvarez. A follow-up inspection was scheduled for 2026-04-10 "
              "but no record of that follow-up inspection exists in the system.",
        source="cmms_workorder", status=EvidenceStatus.VERIFIED, evidence_id="EV7",
    )
    claims = asyncio.run(_interp().interpret(
        item, hypothesis_statement="Scheduled preventive maintenance follow-up was not performed.",
        hypothesis_id="H7",
    ))
    _assert_structurally_valid(claims, "EV7")
    print(f"\n[7 multi-claim] {len(claims)} claim(s): {[(c.epistemic_class, c.text[:60]) for c in claims]}")


# 8. Quantitative evidence (domain: environmental monitoring)
def test_live_quantitative_evidence():
    item = EvidenceItem(
        claim="Environmental monitoring system: cleanroom particle count logged at 4,850 particles/m3 "
              "at 14:32 on 2026-02-18, against an alert limit of 3,520 particles/m3.",
        source="ems_system", status=EvidenceStatus.VERIFIED, evidence_id="EV8",
    )
    claims = asyncio.run(_interp().interpret(
        item, hypothesis_statement="The cleanroom particle count exceeded the alert limit.",
        hypothesis_id="H8",
    ))
    _assert_structurally_valid(claims, "EV8")
    print(f"\n[8 quantitative] {[(c.epistemic_class, c.text[:80]) for c in claims]}")


# ---------------------------------------------------------------------------
# Part S: at least one real graph.ainvoke() through live Ollama end to end
# ---------------------------------------------------------------------------

class _LiveGraphEvidenceProvider(EvidenceProvider):
    async def acquire(self, request):
        if "H1" in request.hypothesis_ids:
            return EvidenceItem(
                claim="Training system record: the operator completed the revised checklist "
                      "training module on 2026-01-05, confirmed by digital signature.",
                source="training_system", status=EvidenceStatus.VERIFIED,
            )
        return EvidenceItem(claim="", source="none", status=EvidenceStatus.UNVERIFIED,
                             hypothesis_relevance="UNAVAILABLE")


def test_live_compiled_graph_with_real_ollama():
    """Real graph.ainvoke() using the real OllamaProvider for evidence
    interpretation (core_synthesis/understanding LLM calls are mocked off
    only to keep this test fast/deterministic in what hypotheses get
    generated -- the evidence-interpretation path, which is what this
    phase is validating live, is NOT mocked)."""
    from unittest.mock import patch

    graph = build_agent_graph()
    interpreter = _interp()
    provider = _LiveGraphEvidenceProvider()
    state = {
        "request": InvestigateRequest(finding_text=(
            "Four employees failed to complete the revised inspection checklist. "
            "One employee reported insufficient training. "
            "Another employee reported workload pressure. "
            "The supervisor reported poor discipline."
        )),
        "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
        "evidence_ledger": [], "errors": [], "trace": [AgentTraceStep.ok("start")],
        "evidence_provider": provider, "evidence_interpreter": interpreter,
    }
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        final_state = asyncio.run(graph.ainvoke(state))

    is_valid, violations = evaluate_all_invariants(final_state)
    assert not any("INV-INVEST-028" in v for v in violations)
    assert not any("INV-INVEST-029" in v for v in violations)
    print(f"\n[live compiled graph] claims={len(final_state.get('evidence_claims') or [])} "
          f"history={len(final_state.get('hypothesis_history') or [])} "
          f"requests={len(final_state.get('evidence_requests') or [])}")
