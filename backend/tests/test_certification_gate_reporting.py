"""Phase 9.1 -- certification GATE reporting (governance only).

`evaluate_certification_gates` interprets a certification tally into four
INDEPENDENT gates. It touches no oracle, no scoring, no case classification --
only the already-computed tally. The bug it fixes: the CLI printed
"AUTONOMOUSLY ELIGIBLE" whenever SILENT_MATERIAL_ERROR == 0, ignoring
CAUGHT_ERROR (genuine but safely-caught capability failures).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from certify_semantic_model import evaluate_certification_gates  # noqa: E402


# 1. The current qwen3:8b state: 16 CORRECT, 6 CAUGHT_ERROR, 0 SILENT -> autonomy FAIL.
def test_current_state_safety_pass_autonomy_fail():
    g = evaluate_certification_gates(
        {"CORRECT": 16, "CAUGHT_ERROR": 6, "SILENT_MATERIAL_ERROR": 0}, n=22)
    assert g["SEMANTIC_SAFETY_GATE"] == "PASS"
    assert g["HUMAN_REVIEWED_PRODUCTION_GATE"] == "PASS (CONDITIONAL)"
    assert g["AUTONOMOUS_CAPABILITY_GATE"] == "FAIL"
    assert "NOT ELIGIBLE" in g["FINAL_STATUS"].upper()
    assert "AUTONOMOUSLY ELIGIBLE" not in g["FINAL_STATUS"]


# 2. Zero silent errors alone must NOT auto-pass the autonomy gate.
def test_zero_silent_errors_does_not_imply_autonomy():
    g = evaluate_certification_gates(
        {"CORRECT": 20, "CAUGHT_ERROR": 2, "SILENT_MATERIAL_ERROR": 0}, n=22)
    assert g["SEMANTIC_SAFETY_GATE"] == "PASS"
    assert g["AUTONOMOUS_CAPABILITY_GATE"] == "FAIL"


# 3. Any silent material error fails BOTH the safety gate and the autonomy gate.
def test_silent_material_error_fails_safety_and_autonomy():
    g = evaluate_certification_gates(
        {"CORRECT": 15, "CAUGHT_ERROR": 5, "SILENT_MATERIAL_ERROR": 2}, n=22)
    assert g["SEMANTIC_SAFETY_GATE"] == "FAIL"
    assert g["AUTONOMOUS_CAPABILITY_GATE"] == "FAIL"
    assert g["HUMAN_REVIEWED_PRODUCTION_GATE"] == "FAIL"
    assert "BLOCKED" in g["FINAL_STATUS"].upper()


# 4. Full held-out capability satisfied (only CORRECT / SAFE_ABSTENTION) -> autonomy MAY pass.
def test_full_capability_allows_autonomy_pass():
    g = evaluate_certification_gates(
        {"CORRECT": 20, "SAFE_ABSTENTION": 2, "CAUGHT_ERROR": 0, "SILENT_MATERIAL_ERROR": 0}, n=22)
    assert g["SEMANTIC_SAFETY_GATE"] == "PASS"
    assert g["AUTONOMOUS_CAPABILITY_GATE"] == "PASS"


def test_all_correct_allows_autonomy_pass():
    g = evaluate_certification_gates({"CORRECT": 22, "SILENT_MATERIAL_ERROR": 0}, n=22)
    assert g["AUTONOMOUS_CAPABILITY_GATE"] == "PASS"


# 5. The CLI can never print a bare "AUTONOMOUSLY ELIGIBLE" while CAUGHT_ERROR > 0.
def test_cli_never_prints_autonomously_eligible_with_caught_errors(capsys, monkeypatch):
    import asyncio

    import certify_semantic_model as cs

    async def _fake_certify(model, ids):
        return {"model": model, "status": "DONE", "n": 22,
                "tally": {"CORRECT": 16, "CAUGHT_ERROR": 6, "SILENT_MATERIAL_ERROR": 0},
                "silent_error_classes": {}, "results": []}

    monkeypatch.setattr(cs, "certify", _fake_certify)
    monkeypatch.setattr(sys, "argv", ["certify_semantic_model.py", "qwen3:8b"])
    asyncio.run(cs.main())
    out = capsys.readouterr().out
    assert "AUTONOMOUSLY ELIGIBLE" not in out
    assert "AUTONOMOUS_CAPABILITY_GATE" in out and "FAIL" in out
    assert "SEMANTIC_SAFETY_GATE" in out and "PASS" in out
    assert "NOT ELIGIBLE" in out.upper()


def test_regression_gate_is_reported_as_not_evaluated_here():
    g = evaluate_certification_gates(
        {"CORRECT": 16, "CAUGHT_ERROR": 6, "SILENT_MATERIAL_ERROR": 0}, n=22)
    assert "NOT EVALUATED" in g["REGRESSION_GATE"].upper()


def test_gates_are_four_independent_values_not_one_boolean():
    g = evaluate_certification_gates(
        {"CORRECT": 16, "CAUGHT_ERROR": 6, "SILENT_MATERIAL_ERROR": 0}, n=22)
    for k in ("SEMANTIC_SAFETY_GATE", "REGRESSION_GATE",
              "HUMAN_REVIEWED_PRODUCTION_GATE", "AUTONOMOUS_CAPABILITY_GATE"):
        assert k in g
    # safety can PASS while capability FAILs -- the whole point
    assert g["SEMANTIC_SAFETY_GATE"] == "PASS"
    assert g["AUTONOMOUS_CAPABILITY_GATE"] == "FAIL"
