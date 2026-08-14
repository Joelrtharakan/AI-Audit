"""Golden-set evaluation harness for POST /api/v1/analyze-finding.

Loads tests/golden/findings.jsonl, calls the live endpoint for each finding,
and reports accuracy against the expected root_cause status and a minimum
confidence level -- so prompt/logic changes can be measured, not eyeballed.
Also reports the GroundingReport (Part 3 hallucination guardrail) for every
case: any HARD violation the golden case didn't expect is itself a failure,
independent of the accuracy number.

Usage:
    python scripts/run_golden_eval.py \\
        --base-url http://localhost:8000 \\
        --api-key devkey123 \\
        --threshold 0.85

Exit code is non-zero if accuracy falls below --threshold OR any case produces
an unexpected HARD grounding violation, so this can gate CI.
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN_SET = BACKEND_ROOT / "tests" / "golden" / "findings.jsonl"

_CONFIDENCE_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def load_golden_set(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def call_analyze_finding(base_url: str, api_key: str, finding: dict, timeout: float) -> dict:
    resp = httpx.post(
        f"{base_url}/api/v1/analyze-finding",
        headers={"Content-Type": "application/json", "X-Internal-Api-Key": api_key},
        json={
            "finding_text": finding["finding_text"],
            "department": finding.get("department") or "",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def evaluate(base_url: str, api_key: str, golden_set: list[dict], timeout: float) -> list[dict]:
    results = []
    for finding in golden_set:
        row = {
            "id": finding["id"],
            "finding_text": finding["finding_text"],
            "expected_status": finding["expected_status"],
            "expected_confidence_min": finding["expected_confidence_min"],
            # Golden cases are not expected to trigger hallucination violations --
            # that failure mode is covered by dedicated unit tests instead. A future
            # golden case could opt in with "expected_hard_violation_count" if needed.
            "expected_hard_violation_count": finding.get("expected_hard_violation_count", 0),
            "notes": finding.get("notes", ""),
        }
        try:
            response = call_analyze_finding(base_url, api_key, finding, timeout)
        except httpx.HTTPError as exc:
            row.update(
                error=str(exc),
                actual_status=None,
                actual_confidence=None,
                reasoning="",
                status_correct=False,
                confidence_correct=False,
                correct=False,
                hard_violation_count=0,
                unexpected_violation=False,
                grounding_violations=[],
            )
            results.append(row)
            continue

        actual_status = response["root_cause"]["status"]
        actual_confidence = response["analysis"]["confidence"]
        reasoning = response["root_cause"].get("classification_reasoning") or ""
        grounding = response.get("grounding_report") or {}
        hard_violations = grounding.get("hard_violations", [])

        status_correct = actual_status == finding["expected_status"]
        confidence_correct = (
            _CONFIDENCE_RANK.get(actual_confidence, -1) >= _CONFIDENCE_RANK.get(finding["expected_confidence_min"], 0)
        )
        unexpected_violation = len(hard_violations) > row["expected_hard_violation_count"]

        row.update(
            error=None,
            actual_status=actual_status,
            actual_confidence=actual_confidence,
            reasoning=reasoning,
            status_correct=status_correct,
            confidence_correct=confidence_correct,
            correct=status_correct and confidence_correct,
            hard_violation_count=len(hard_violations),
            unexpected_violation=unexpected_violation,
            grounding_violations=hard_violations,
        )
        results.append(row)
    return results


def print_report(results: list[dict], threshold: float) -> tuple[float, bool]:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total else 0.0
    any_unexpected_violation = any(r["unexpected_violation"] for r in results)

    print(f"\n{'=' * 70}")
    print(f"GOLDEN EVAL REPORT -- {correct}/{total} correct ({accuracy:.1%})")
    print(f"{'=' * 70}\n")

    mismatches = [r for r in results if not r["correct"] or r["unexpected_violation"]]
    if mismatches:
        print(f"MISMATCHES / VIOLATIONS ({len(mismatches)}):\n")
        for r in mismatches:
            print(f"  [{r['id']}] {r['finding_text'][:80]}...")
            if r["error"]:
                print(f"    ERROR: {r['error']}")
            else:
                print(f"    expected_status={r['expected_status']}  actual_status={r['actual_status']}"
                      f"{'  <-- MISMATCH' if not r['status_correct'] else ''}")
                print(f"    expected_confidence_min={r['expected_confidence_min']}  actual_confidence={r['actual_confidence']}"
                      f"{'  <-- MISMATCH' if not r['confidence_correct'] else ''}")
                if r["reasoning"]:
                    print(f"    reasoning: {r['reasoning'][:150]}")
                if r["unexpected_violation"]:
                    print(f"    UNEXPECTED HARD VIOLATION(S): {r['hard_violation_count']}")
                    for v in r["grounding_violations"]:
                        print(f"      - field={v.get('field')} entity={v.get('claimed_entity')!r} note={v.get('note')}")
            print(f"    notes: {r['notes']}\n")
    else:
        print("No mismatches.\n")

    # Confidence distribution -- flags the exact symptom this pipeline was calibrated
    # against: everything collapsing into one confidence bucket.
    distribution: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "N/A": 0}
    for r in results:
        distribution[r["actual_confidence"] or "N/A"] += 1

    print("CONFIDENCE DISTRIBUTION:")
    scored = total - distribution["N/A"]
    for level in ("LOW", "MEDIUM", "HIGH"):
        count = distribution[level]
        pct = (count / scored * 100) if scored else 0.0
        bar = "#" * int(pct / 2)
        print(f"  {level:<8} {count:>3}  {pct:5.1f}%  {bar}")
    if distribution["N/A"]:
        print(f"  {'ERROR':<8} {distribution['N/A']:>3}  (request failed)")

    if scored:
        max_pct = max(distribution[lvl] / scored for lvl in ("LOW", "MEDIUM", "HIGH"))
        if max_pct > 0.70:
            print(f"\n  WARNING: {max_pct:.0%} of results cluster in a single confidence bucket "
                  f"-- calibration may be collapsed.")

    # Grounding guardrail aggregate -- rare backstop (good) vs firing constantly (the
    # generation prompt itself needs work, not just the guardrail).
    total_violations = sum(r["hard_violation_count"] for r in results)
    cases_with_violations = sum(1 for r in results if r["hard_violation_count"] > 0)
    print(f"\nGROUNDING GUARDRAIL: {total_violations} hard violation(s) across {cases_with_violations}/{total} case(s)"
          f"{' -- ' + str(sum(1 for r in results if r['unexpected_violation'])) + ' UNEXPECTED' if any_unexpected_violation else ''}")

    print(f"\n{'=' * 70}")
    status_line = f"Threshold: {threshold:.0%}  |  Actual: {accuracy:.1%}"
    passed = accuracy >= threshold and not any_unexpected_violation
    print(f"{status_line}  |  Unexpected violations: {'YES' if any_unexpected_violation else 'no'}  |  "
          f"{'PASS' if passed else 'FAIL'}")
    print(f"{'=' * 70}\n")

    return accuracy, any_unexpected_violation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="devkey123")
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--threshold", type=float, default=0.85, help="Minimum accuracy to exit 0 (default 0.85).")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds.")
    args = parser.parse_args()

    golden_set = load_golden_set(args.golden_set)
    print(f"Loaded {len(golden_set)} findings from {args.golden_set}")
    print(f"Evaluating against {args.base_url} ...")

    results = evaluate(args.base_url, args.api_key, golden_set, args.timeout)
    accuracy, any_unexpected_violation = print_report(results, args.threshold)

    sys.exit(0 if (accuracy >= args.threshold and not any_unexpected_violation) else 1)


if __name__ == "__main__":
    main()
