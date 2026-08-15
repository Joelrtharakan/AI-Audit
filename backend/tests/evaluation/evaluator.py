"""Core Evaluation Engine for LQMS AI Agent.
Runs evaluation over findings dataset and computes comprehensive scores.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tests.evaluation.adapter import AgentAdapter
from tests.evaluation.findings_dataset import FINDINGS_DATASET
from tests.evaluation.golden_expectations import GOLDEN_EXPECTATIONS
from tests.evaluation.readiness_gate import ReadinessConfig, evaluate_production_readiness
from tests.evaluation.report import generate_evaluation_artifacts
from tests.evaluation.scoring import FindingScoreResult, score_finding_output

logger = logging.getLogger(__name__)


class AgentEvaluator:
    """Production-grade evaluation engine for the LQMS AI Agent."""

    def __init__(self, adapter: Optional[AgentAdapter] = None):
        self.adapter = adapter or AgentAdapter()

    async def evaluate_finding_by_id(self, finding_id: str) -> FindingScoreResult:
        finding = next((f for f in FINDINGS_DATASET if f["id"] == finding_id), None)
        if not finding:
            raise ValueError(f"Finding ID {finding_id} not found in dataset.")

        golden = GOLDEN_EXPECTATIONS.get(finding_id, {})
        agent_output = await self.adapter.analyze(
            finding_text=finding["finding_text"],
            departments=finding.get("departments"),
        )
        return score_finding_output(finding_id, finding["finding_text"], golden, agent_output)

    async def run_evaluation(
        self,
        category: Optional[str] = None,
        finding_id: Optional[str] = None,
        output_dir: Path = Path("results"),
        readiness_config: ReadinessConfig = ReadinessConfig(),
        dataset: Optional[List[Dict[str, Any]]] = None,
        golden_expectations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        target_findings = dataset if dataset is not None else FINDINGS_DATASET
        goldens = golden_expectations if golden_expectations is not None else GOLDEN_EXPECTATIONS

        if finding_id:
            target_findings = [f for f in target_findings if f["id"] == finding_id]
        elif category:
            target_findings = [f for f in target_findings if f["category"].lower() == category.lower()]


        finding_results: List[FindingScoreResult] = []
        all_failures: List[Any] = []

        logger.info("Starting evaluation on %d findings...", len(target_findings))

        for item in target_findings:
            fid = item["id"]
            golden = goldens.get(fid, {})
            logger.info("Evaluating finding %s (%s)...", fid, item["category"])


            agent_output = await self.adapter.analyze(
                finding_text=item["finding_text"],
                departments=item.get("departments"),
            )

            res = score_finding_output(fid, item["finding_text"], golden, agent_output)
            finding_results.append(res)
            all_failures.extend(res.failures)

        # Compute averages
        total_score = (
            sum(r.total_score for r in finding_results) / len(finding_results) if finding_results else 0.0
        )

        dim_keys = [
            "fact_preservation",
            "observation_quality",
            "root_cause_discipline",
            "evidence_boundary",
            "hypothesis_quality",
            "evidence_recommendations",
            "capa_discipline",
            "impact_assessment",
            "consistency",
            "5why_causal_coherence",
            "unsupported_specificity",
            "causal_leap_detection",
        ]


        dim_averages = {}
        for dk in dim_keys:
            avg_val = (
                sum(r.dimension_scores.get(dk, 0.0) for r in finding_results) / len(finding_results)
                if finding_results
                else 0.0
            )
            dim_averages[dk] = round(avg_val, 1)

        readiness = evaluate_production_readiness(
            overall_score=total_score,
            dimension_averages=dim_averages,
            failures=all_failures,
            config=readiness_config,
        )

        artifact_paths = generate_evaluation_artifacts(
            overall_score=round(total_score, 1),
            dimension_averages=dim_averages,
            finding_results=finding_results,
            all_failures=all_failures,
            readiness=readiness,
            output_dir=output_dir,
        )

        return {
            "overall_score": round(total_score, 1),
            "dimension_averages": dim_averages,
            "readiness_gate": readiness,
            "findings_evaluated": len(finding_results),
            "total_failures": len(all_failures),
            "artifact_paths": artifact_paths,
            "results": finding_results,
        }
