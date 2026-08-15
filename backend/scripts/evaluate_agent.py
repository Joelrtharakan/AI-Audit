#!/usr/bin/env python3
"""CLI Entrypoint for LQMS AI Agent Evaluation & Regression Framework.

Usage:
  python scripts/evaluate_agent.py
  python scripts/evaluate_agent.py --finding F001
  python scripts/evaluate_agent.py --category procedure
  python scripts/evaluate_agent.py --regression
  python scripts/evaluate_agent.py --report
  python scripts/evaluate_agent.py --compare results/previous_results.json
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from tests.evaluation.adapter import AgentAdapter
from tests.evaluation.evaluator import AgentEvaluator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_agent")


async def main():
    parser = argparse.ArgumentParser(description="LQMS AI Agent Evaluation & Regression Framework")
    parser.add_argument("--offline", action="store_true", help="Run evaluation in fast deterministic mode without external API calls")
    parser.add_argument("--unseen", action="store_true", help="Run evaluation on 30 unseen findings blind benchmark")
    parser.add_argument("--adversarial", action="store_true", help="Run evaluation on 10 adversarial/prompt-injection findings benchmark")
    parser.add_argument("--finding", type=str, help="Filter evaluation to a specific finding ID (e.g. F001 or U001 or ADV001)")
    parser.add_argument("--category", type=str, help="Filter evaluation to a specific category (e.g. procedure)")
    parser.add_argument("--regression", action="store_true", help="Run in regression mode using saved regression_tests.json")
    parser.add_argument("--report", action="store_true", help="Generate and print detailed evaluation report summary")
    parser.add_argument("--compare", type=str, help="Compare current run results against a previous results JSON file")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save evaluation artifacts")

    args = parser.parse_args()
    output_path = Path(args.output_dir)

    adapter = AgentAdapter(offline=args.offline)
    evaluator = AgentEvaluator(adapter=adapter)

    target_dataset = None
    target_goldens = None

    if args.unseen:
        from tests.evaluation.unseen_findings_dataset import UNSEEN_FINDINGS_DATASET, UNSEEN_GOLDEN_EXPECTATIONS
        logger.info("Evaluating on 30 UNSEEN findings dataset...")
        target_dataset = UNSEEN_FINDINGS_DATASET
        target_goldens = UNSEEN_GOLDEN_EXPECTATIONS
    elif args.adversarial:
        from tests.evaluation.adversarial_findings_dataset import ADVERSARIAL_FINDINGS_DATASET, ADVERSARIAL_GOLDEN_EXPECTATIONS
        logger.info("Evaluating on 10 ADVERSARIAL findings dataset...")
        target_dataset = ADVERSARIAL_FINDINGS_DATASET
        target_goldens = ADVERSARIAL_GOLDEN_EXPECTATIONS

    if args.regression:
        logger.info("Running Regression Test Suite...")
        res = await evaluator.run_evaluation(
            output_dir=output_path,
            dataset=target_dataset,
            golden_expectations=target_goldens,
        )
    else:
        res = await evaluator.run_evaluation(
            category=args.category,
            finding_id=args.finding,
            output_dir=output_path,
            dataset=target_dataset,
            golden_expectations=target_goldens,
        )


    print("\n" + "=" * 60)
    print("      LQMS AI AGENT EVALUATION & REGRESSION SUMMARY")
    print("=" * 60)
    print(f"Overall Evaluation Score : {res['overall_score']} / 100")
    print(f"Findings Evaluated       : {res['findings_evaluated']}")
    print(f"Total Failures Detected  : {res['total_failures']}")
    print(f"Production Gate Status   : {'PASSED' if res['readiness_gate'].passed else 'FAILED'}")

    if res['readiness_gate'].reasons:
        print("\nProduction Gate Issues:")
        for reason in res['readiness_gate'].reasons:
            print(f"  - {reason}")

    print("\nDimension Averages (%):")
    for dim, score in res['dimension_averages'].items():
        print(f"  - {dim:<28}: {score}%")

    if args.compare and Path(args.compare).exists():
        print("\n" + "-" * 60)
        print("              BEFORE vs AFTER COMPARISON")
        print("-" * 60)
        try:
            prev_data = json.loads(Path(args.compare).read_text(encoding="utf-8"))
            prev_score = prev_data.get("overall_score", 0.0)
            score_diff = res["overall_score"] - prev_score
            sign = "+" if score_diff >= 0 else ""
            print(f"Previous Score : {prev_score} / 100")
            print(f"Current Score  : {res['overall_score']} / 100")
            print(f"Score Delta    : {sign}{score_diff:.1f} pts")
        except Exception as exc:
            logger.error("Failed to parse previous comparison JSON: %s", exc)

    print("\nEvaluation artifacts written to:")
    for name, p in res["artifact_paths"].items():
        print(f"  - {name:<12}: {p}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
