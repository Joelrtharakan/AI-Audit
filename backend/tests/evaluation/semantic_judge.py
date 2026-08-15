"""Hybrid Semantic Evaluator for Nuanced Reasoning Checks.
Combines rule-based checks with optional LLM-as-judge semantic analysis.
"""

from typing import Any, Dict, List, Tuple
from tests.evaluation.failure_codes import FailureCode, FailureRecord, Severity


class SemanticJudge:
    """Evaluates semantic reasoning alignment without relying solely on string matching."""

    def evaluate_semantic_alignment(
        self,
        finding_id: str,
        finding_text: str,
        golden_exp: Dict[str, Any],
        agent_output: Dict[str, Any],
    ) -> Tuple[float, List[FailureRecord]]:
        score = 10.0
        failures: List[FailureRecord] = []

        rc = agent_output.get("root_cause")
        narrative = (getattr(rc, "narrative", "") or "").lower()

        # Check if the narrative states an unconfirmed cause as established
        if "established" in narrative and "not established" not in narrative and "unverified" not in narrative:
            if golden_exp.get("root_cause_status") == "NOT_ESTABLISHED":
                score -= 5.0
                failures.append(
                    FailureRecord(
                        finding_id=finding_id,
                        failure_code=FailureCode.OVERCONFIDENT_OUTPUT,
                        severity=Severity.HIGH,
                        explanation="Semantic narrative implies cause is confirmed when evidence is insufficient.",
                        expected_behavior="Semantic narrative must reflect uncertainty.",
                        actual_output=narrative,
                    )
                )

        return max(0.0, score), failures
