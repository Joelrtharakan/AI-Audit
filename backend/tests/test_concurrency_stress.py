"""High-concurrency batch stress test (offline, deterministic-fallback path).

Complements test_concurrent_isolation.py (which exercises a single node,
root_cause_node, across 4 concurrent findings) by running the FULL
understanding -> plan -> synthesis -> report -> verification pipeline for
many distinct findings at once, all forced onto the LLM-free deterministic
path (same mocking approach as test_golden_20_scenarios.py) so the run stays
fast and offline. The goal is to catch any module-level mutable state
(caches, counters, shared mutable defaults) that would leak claim IDs,
financial amounts, or root-cause content between concurrently running
requests.
"""

from __future__ import annotations

import asyncio

import pytest
from unittest.mock import patch

from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.nodes.understanding import understand_finding_node
from app.models.agent import InvestigateRequest

# Distinct findings, each with a unique fact that must never appear in any
# other case's report (amount, actor, or subject noun).
_CASES = {
    "dup_payment_a": (
        "A duplicate payment of INR 125,000 was made to supplier Alpha Traders "
        "for invoice INV-9001.",
        ["125,000", "Alpha Traders"],
    ),
    "dup_payment_b": (
        "A duplicate payment of INR 87,500 was made to supplier Beta Components "
        "for invoice INV-4477.",
        ["87,500", "Beta Components"],
    ),
    "training_gap": (
        "Four operators on the Kettle-3 line performed the revised cleaning "
        "procedure without completing the mandatory retraining.",
        ["Kettle-3"],
    ),
    "temp_excursion": (
        "Refrigerator R-22 temperature logs in the Cold Storage department were "
        "found missing entries for the evening shift on three consecutive days; "
        "staff confirmed they had not been retrained on the revised SOP-COLD-001.",
        ["R-22", "Cold Storage"],
    ),
    "calibration_lapse": (
        "The torque wrench used on the Assembly-7 station was found to be "
        "past its calibration due date by 45 days.",
        ["Assembly-7"],
    ),
    "missing_signature": (
        "The batch release record for Lot L-2201 was missing the QA "
        "supervisor's countersignature.",
        ["L-2201"],
    ),
    "control_bypass": (
        "The change-control approval step was bypassed for the packaging "
        "line firmware update deployed on Line-9.",
        ["Line-9"],
    ),
    "supplier_defect": (
        "Incoming inspection rejected 12 units from supplier Gamma Industries "
        "due to out-of-spec dimensions.",
        ["Gamma Industries"],
    ),
}


async def _run_pipeline(finding_text: str) -> dict:
    state = {
        "request": InvestigateRequest(finding_text=finding_text),
        "evidence_ledger": [],
        "iteration_count": 0,
        "tool_call_count": 0,
        "critic_iteration": 0,
        "trace": [],
        "errors": [],
    }
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        state = await understand_finding_node(state)
        state = await plan_investigation_node(state)
        state = await core_synthesis_node(state)
        state = await generate_report_node(state)
        state = await final_evidence_verification_node(state)
    return state


@pytest.mark.asyncio
async def test_high_concurrency_batch_state_isolation():
    """Runs 8 distinct findings, 5x each (40 concurrent full-pipeline
    invocations), and asserts every report only contains facts unique to
    its own finding -- no cross-request bleed under concurrent execution."""
    names = list(_CASES.keys()) * 5

    async def _run(name: str):
        text, _ = _CASES[name]
        state = await _run_pipeline(text)
        is_valid, violations = evaluate_all_invariants(state)
        return name, state, is_valid, violations

    results = await asyncio.gather(*(_run(name) for name in names))

    for name, state, is_valid, violations in results:
        assert is_valid, f"case {name!r}: invariant violations {violations}"
        report = state.get("report")
        assert report is not None, f"case {name!r}: no report produced"

        report_text = report.model_dump_json()
        _, own_markers = _CASES[name]
        for marker in own_markers:
            assert marker in report_text, (
                f"case {name!r}: expected own marker {marker!r} missing from report"
            )
        for other_name, (_, other_markers) in _CASES.items():
            if other_name == name:
                continue
            for marker in other_markers:
                assert marker not in report_text, (
                    f"case {name!r}: found foreign marker {marker!r} from "
                    f"case {other_name!r} -- possible cross-request state bleed"
                )
