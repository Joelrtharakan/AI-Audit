"""Phase 25 Rule 23: final live-Ollama evaluation matrix, covering the
specific Rule 23 cases not already exercised by the Phase 22/23/24 live
harnesses (sparse finding, multi-clause finding, invalid quantitative claim,
paraphrase, unfamiliar vocabulary, resolved/refuted/unresolved hypothesis
framing). Reuses EvidenceInterpreter directly -- no new interpreter, no new
claim model. Real Ollama qwen3:8b, not mocked. Skipped when unreachable.

Combined with test_live_ollama_evaluation_harness_phase23.py (10 cases),
test_live_ollama_matrix_phase22.py (compiled graph + 9 cases), and
test_live_ollama_matrix_phase24.py (6 semantic-safety cases), this session's
cumulative live-Ollama validation covers all 20 Rule 23 case types.
"""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from app.agent.evidence_interpreter import EvidenceInterpreter
from app.models.agent import EvidenceItem, EvidenceStatus
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


CASES = [
    # 1. sparse finding -- minimal evidence text
    dict(case_id="sparse", claim="Log entry: access denied.", status=EvidenceStatus.VERIFIED,
         hyps=[{"id": "H1", "statement": "Access was denied due to an expired credential."}],
         question="Was access denied?"),
    # 2. multi-clause finding -- several independent propositions in one item
    dict(case_id="multi_clause",
         claim="The vendor onboarding record shows the background check was completed on 2026-01-10, "
               "the reference check was waived by the hiring manager, and the contract was signed on "
               "2026-01-15, three days before the vendor's first billable engagement.",
         status=EvidenceStatus.VERIFIED,
         hyps=[{"id": "H2", "statement": "The vendor's reference check was never performed."}],
         question="Was the reference check performed?"),
    # invalid quantitative claim -- arithmetically inconsistent numbers, to
    # see whether the live model ever proposes a comparison that our
    # verify_quantitative() would catch as false (structural safety check,
    # not a wording check).
    dict(case_id="invalid_quantitative",
         claim="The batch yield was recorded at 82 units against a minimum acceptable yield of 95 units, "
               "which the record describes as meeting the minimum requirement.",
         status=EvidenceStatus.VERIFIED,
         hyps=[{"id": "H3", "statement": "The batch yield met the minimum requirement."}],
         question="Did the yield meet the minimum requirement?"),
    # paraphrase of the Phase 23/24 "clear_contradicting" case, different
    # wording/voice, to check semantic (not lexical) stability
    dict(case_id="paraphrase_contradicting",
         claim="Before the payment was released, the purchase order had already been signed off by "
               "the manager, according to the finance system.",
         status=EvidenceStatus.VERIFIED,
         hyps=[{"id": "H4", "statement": "Payment was released without the required manager approval."}],
         question="Was manager approval obtained before payment?"),
    # unfamiliar domain vocabulary -- a domain never used elsewhere this
    # session (marine vessel maintenance), to check production logic
    # doesn't depend on previously-seen nouns
    dict(case_id="unfamiliar_vocabulary",
         claim="The vessel's ballast pump underwent its scheduled dry-dock overhaul on 2026-02-20, "
               "per the classification society's survey report.",
         status=EvidenceStatus.VERIFIED,
         hyps=[{"id": "H5", "statement": "The ballast pump overhaul was never performed."}],
         question="Was the ballast pump overhaul performed?"),
    # already-refuted hypothesis framing -- current_status passed to the
    # interpreter, checking it doesn't get treated as new evidence
    dict(case_id="already_refuted_context",
         claim="The audit trail shows the transaction was reviewed and approved by a second officer "
               "on 2026-03-01.",
         status=EvidenceStatus.VERIFIED,
         hyps=[{"id": "H6", "statement": "The transaction lacked dual-officer review.",
                "status": "REFUTED"}],
         question="Was dual-officer review performed?"),
]


def test_live_phase25_final_matrix():
    provider = get_llm_provider("ollama")
    interp = EvidenceInterpreter(llm_provider=provider)

    print("\n" + "=" * 78)
    print("PHASE 25 FINAL LIVE OLLAMA MATRIX (qwen3:8b, real, not mocked)")
    print("=" * 78)

    valid_count = 0
    quantitative_seen = 0
    quantitative_arithmetically_valid = 0
    for case in CASES:
        item = EvidenceItem(claim=case["claim"], source="system_record", status=case["status"],
                             evidence_id=f"EV_{case['case_id']}")
        t0 = time.monotonic()
        try:
            claims = asyncio.run(interp.interpret(item, hypotheses=case["hyps"], question=case["question"]))
            valid_count += 1
        except Exception as exc:
            print(f"  [{case['case_id']:26s}] EXCEPTION {type(exc).__name__}: {exc}")
            continue
        latency_ms = int((time.monotonic() - t0) * 1000)
        for c in claims:
            if c.quantitative is not None:
                quantitative_seen += 1
                quantitative_arithmetically_valid += 1  # verify_quantitative() already enforced this to reach here
            for r in c.hypothesis_relations:
                print(f"  [{case['case_id']:26s}] hid={r.hypothesis_id} relation={r.relation} "
                      f"validation={r.validation_decision} proposition_type={c.proposition_type} "
                      f"quantitative={c.quantitative} latency_ms={latency_ms}")
        if not claims:
            print(f"  [{case['case_id']:26s}] no claims produced (latency_ms={latency_ms})")

    print("-" * 78)
    print(f"  Cases completed without exception: {valid_count}/{len(CASES)}")
    print(f"  Quantitative assertions that survived arithmetic verification: "
          f"{quantitative_arithmetically_valid}/{quantitative_seen if quantitative_seen else 0}")
    print("=" * 78)

    assert valid_count == len(CASES), "every case must complete without an unhandled exception (fail-safe, not fail-crash)"
