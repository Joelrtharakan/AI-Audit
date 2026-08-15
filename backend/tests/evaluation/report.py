"""Report Generator for Evaluation Framework.
Produces markdown reports, evaluation results JSON, failure patterns JSON, and regression tests JSON.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from tests.evaluation.readiness_gate import ReadinessResult


def generate_evaluation_artifacts(
    overall_score: float,
    dimension_averages: Dict[str, float],
    finding_results: List[Any],
    all_failures: List[Any],
    readiness: ReadinessResult,
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. evaluation_results.json
    results_data = {
        "overall_score": overall_score,
        "dimension_averages": dimension_averages,
        "readiness_gate": {
            "passed": readiness.passed,
            "reasons": readiness.reasons,
        },
        "findings": [
            {
                "finding_id": r.finding_id,
                "total_score": r.total_score,
                "dimension_scores": r.dimension_scores,
                "failures_count": len(r.failures),
                "failures": [
                    {
                        "code": f.failure_code.value,
                        "severity": f.severity.value,
                        "explanation": f.explanation,
                        "expected": f.expected_behavior,
                        "actual": f.actual_output,
                    }
                    for f in r.failures
                ],
            }
            for r in finding_results
        ],
    }

    results_path = output_dir / "evaluation_results.json"
    results_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")

    # 2. failure_patterns.json
    failure_counts: Dict[str, Dict[str, Any]] = {}
    for f in all_failures:
        code = f.failure_code.value
        if code not in failure_counts:
            failure_counts[code] = {
                "count": 0,
                "severity": f.severity.value,
                "examples": [],
            }
        failure_counts[code]["count"] += 1
        if len(failure_counts[code]["examples"]) < 3:
            failure_counts[code]["examples"].append(
                {"finding_id": f.finding_id, "explanation": f.explanation}
            )

    patterns_path = output_dir / "failure_patterns.json"
    patterns_path.write_text(json.dumps(failure_counts, indent=2), encoding="utf-8")

    # 3. regression_tests.json
    regression_tests = [
        {
            "finding_id": f.finding_id,
            "failure_code": f.failure_code.value,
            "severity": f.severity.value,
            "must_fix_condition": f.expected_behavior,
        }
        for f in all_failures
    ]
    regression_path = output_dir / "regression_tests.json"
    regression_path.write_text(json.dumps(regression_tests, indent=2), encoding="utf-8")

    # 4. evaluation_report.md
    md_content = []
    md_content.append("# LQMS AI Agent Evaluation & Regression Report\n")
    md_content.append(f"**Overall Evaluation Score:** `{overall_score:.1f} / 100`\n")

    gate_str = "PASSED" if readiness.passed else "FAILED"
    md_content.append(f"**Production Readiness Gate Status:** `{gate_str}`\n")

    if readiness.reasons:
        md_content.append("### Production Gate Reasons:")
        for r in readiness.reasons:
            md_content.append(f"- :warning: {r}")
        md_content.append("")

    md_content.append("## 1. Dimension Scores Summary\n")
    md_content.append("| Dimension | Average Score (%) | Status |")
    md_content.append("|---|---|---|")
    for dim, score in dimension_averages.items():
        status = "PASSED" if score >= 90.0 else "NEEDS IMPROVEMENT"
        md_content.append(f"| {dim.replace('_', ' ').title()} | {score:.1f}% | {status} |")

    md_content.append("\n## 2. Failure Pattern Analysis\n")
    if not failure_counts:
        md_content.append("Zero failure patterns detected across the dataset.\n")
    else:
        md_content.append("| Failure Code | Severity | Occurrences | Example |")
        md_content.append("|---|---|---|---|")
        for code, info in failure_counts.items():
            ex_str = info["examples"][0]["explanation"] if info["examples"] else ""
            md_content.append(f"| `{code}` | `{info['severity']}` | {info['count']} | {ex_str} |")

    md_content.append("\n## 3. Finding-by-Finding Breakdown\n")
    md_content.append("| Finding ID | Score (0-100) | Failures Count |")
    md_content.append("|---|---|---|")
    for r in finding_results:
        md_content.append(f"| `{r.finding_id}` | {r.total_score} | {len(r.failures)} |")

    report_path = output_dir / "evaluation_report.md"
    report_path.write_text("\n".join(md_content), encoding="utf-8")

    return {
        "results": results_path,
        "patterns": patterns_path,
        "regression": regression_path,
        "report": report_path,
    }
