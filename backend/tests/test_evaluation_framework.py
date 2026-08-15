"""Pytest unit test for the Production-Grade Evaluation Framework.
Validates dataset integrity, rubric scoring, failure detection, and artifact generation.
"""

import json
from pathlib import Path
import pytest

from tests.evaluation.findings_dataset import FINDINGS_DATASET
from tests.evaluation.golden_expectations import GOLDEN_EXPECTATIONS
from tests.evaluation.failure_codes import FailureCode, Severity
from tests.evaluation.readiness_gate import ReadinessConfig, evaluate_production_readiness
from tests.evaluation.scoring import score_finding_output
from tests.evaluation.report import generate_evaluation_artifacts


def test_findings_dataset_has_at_least_25_items():
    assert len(FINDINGS_DATASET) >= 25, f"Dataset must contain at least 25 items, found {len(FINDINGS_DATASET)}"
    categories = set(f["category"] for f in FINDINGS_DATASET)
    assert len(categories) >= 6, "Dataset must cover multiple categories"


def test_golden_expectations_cover_all_findings():
    for f in FINDINGS_DATASET:
        fid = f["id"]
        assert fid in GOLDEN_EXPECTATIONS, f"Golden expectation missing for finding ID {fid}"
        exp = GOLDEN_EXPECTATIONS[fid]
        assert "observation_quality" in exp
        assert "root_cause_status" in exp
        assert "required_evidence" in exp


def test_scoring_rubric_fact_preservation_and_discipline():
    finding_id = "F007"
    finding_text = "Five production records contained incomplete entries in the environmental control section."
    golden = GOLDEN_EXPECTATIONS[finding_id]

    mock_agent_output = {
        "observation_quality": type("OQ", (), {"status": "SUFFICIENT"})(),
        "root_cause": type("RC", (), {
            "status": "NOT_ESTABLISHED",
            "narrative": "Root cause not established from available evidence.",
            "candidate_hypotheses": [
                type("H", (), {
                    "id": "H1",
                    "name": "EXECUTION_OMISSION",
                    "statement": "The required activity associated with five production records may not have been performed.",
                    "evidence_needed": "five production records, environmental logs, audit trail",
                })(),
            ],
        })(),
        "five_why": type("FW", (), {
            "steps": [
                type("Step", (), {"step_number": 1, "status": "VERIFIED", "explanation": "Five records incomplete."})(),
                type("Step", (), {"step_number": 2, "status": "UNKNOWN", "explanation": "Evidence insufficient."})(),
            ]
        })(),
        "investigation_plan": type("IP", (), {
            "questions": [
                type("Q", (), {"evidence": "five production records, environmental logs, audit trail"})()
            ]
        })(),
        "capa_analysis": type("CAPA", (), {"status": "INVESTIGATION_REQUIRED"})(),
        "impact_assessment": type("IA", (), {"narrative": "Impact pending assessment."})(),
        "ca_draft": type("CAD", (), {"root_cause_category": "TO_BE_CONFIRMED"})(),
    }

    result = score_finding_output(finding_id, finding_text, golden, mock_agent_output)
    assert result.total_score >= 90.0, f"Expected high score for clean evidence-bound output, got {result.total_score}"
    assert len(result.failures) == 0, f"Expected 0 failures, got {result.failures}"


def test_scoring_detects_unsupported_root_cause_and_blame():
    finding_id = "F023"
    finding_text = "The shift supervisor stated during the audit interview that they believe the operator was careless."
    golden = GOLDEN_EXPECTATIONS[finding_id]

    bad_agent_output = {
        "observation_quality": type("OQ", (), {"status": "SUFFICIENT"})(),
        "root_cause": type("RC", (), {
            "status": "ESTABLISHED",
            "narrative": "Root cause confirmed as operator carelessness and human error.",
            "candidate_hypotheses": [
                type("H", (), {
                    "id": "H1",
                    "name": "HUMAN_ERROR_CARELESSNESS",
                    "statement": "Operator carelessness caused omission.",
                    "evidence_needed": "none",
                })(),
            ],
        })(),
        "five_why": type("FW", (), {
            "steps": [
                type("Step", (), {"step_number": 1, "status": "VERIFIED", "explanation": "Supervisor statement."})(),
                type("Step", (), {"step_number": 2, "status": "VERIFIED", "explanation": "Operator was careless."})(),
                type("Step", (), {"step_number": 3, "status": "VERIFIED", "explanation": "Lack of attention."})(),
            ]
        })(),
        "capa_analysis": type("CAPA", (), {"status": "AI_SUGGESTED"})(),
        "impact_assessment": type("IA", (), {"narrative": "Patient safety and recall risk."})(),
        "ca_draft": type("CAD", (), {"root_cause_category": "HUMAN_ERROR"})(),
    }

    result = score_finding_output(finding_id, finding_text, golden, bad_agent_output)
    assert result.total_score < 70.0, f"Expected low score for bad output, got {result.total_score}"
    assert len(result.failures) > 0, "Expected failures for bad output"
    codes = [f.failure_code for f in result.failures]
    assert FailureCode.HUMAN_BLAME_WITHOUT_EVIDENCE in codes or FailureCode.UNSUPPORTED_ROOT_CAUSE in codes


def test_production_readiness_gate_rejects_critical_failures():
    failures = [
        type("F", (), {"failure_code": FailureCode.FACT_INVENTION, "severity": Severity.CRITICAL})(),
    ]
    readiness = evaluate_production_readiness(
        overall_score=95.0,
        dimension_averages={"root_cause_discipline": 95.0, "evidence_boundary": 95.0, "fact_preservation": 95.0, "capa_discipline": 95.0},
        failures=failures,
    )
    assert not readiness.passed, "Readiness gate must reject run with CRITICAL failures"
    assert len(readiness.reasons) >= 1


def test_generate_evaluation_artifacts(tmp_path: Path):
    from tests.evaluation.scoring import FindingScoreResult
    mock_results = [
        FindingScoreResult(
            finding_id="F001",
            total_score=95.0,
            dimension_scores={"fact_preservation": 100.0},
            failures=[],
        )
    ]
    readiness = evaluate_production_readiness(
        overall_score=95.0,
        dimension_averages={"fact_preservation": 100.0},
        failures=[],
    )

    paths = generate_evaluation_artifacts(
        overall_score=95.0,
        dimension_averages={"fact_preservation": 100.0},
        finding_results=mock_results,
        all_failures=[],
        readiness=readiness,
        output_dir=tmp_path,
    )

    assert paths["results"].exists()
    assert paths["patterns"].exists()
    assert paths["regression"].exists()
    assert paths["report"].exists()

    report_text = paths["report"].read_text(encoding="utf-8")
    assert "LQMS AI Agent Evaluation & Regression Report" in report_text
