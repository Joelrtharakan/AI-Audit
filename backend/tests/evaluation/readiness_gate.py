"""Production Readiness Gate for LQMS AI Agent.
Evaluates overall scores and failure severity counts against configurable production thresholds.
"""

from typing import Any, Dict, List, NamedTuple
from tests.evaluation.failure_codes import Severity


class ReadinessConfig(NamedTuple):
    overall_score_threshold: float = 90.0
    root_cause_discipline_threshold: float = 90.0
    evidence_boundary_threshold: float = 90.0
    fact_preservation_threshold: float = 95.0
    capa_discipline_threshold: float = 90.0
    max_critical_failures: int = 0
    max_high_failures: int = 2


class ReadinessResult(NamedTuple):
    passed: bool
    reasons: List[str]
    config_used: ReadinessConfig


def evaluate_production_readiness(
    overall_score: float,
    dimension_averages: Dict[str, float],
    failures: List[Any],
    config: ReadinessConfig = ReadinessConfig(),
) -> ReadinessResult:
    reasons = []

    if overall_score < config.overall_score_threshold:
        reasons.append(
            f"Overall score {overall_score:.1f} below threshold {config.overall_score_threshold:.1f}"
        )

    rc_score = dimension_averages.get("root_cause_discipline", 0.0)
    if rc_score < config.root_cause_discipline_threshold:
        reasons.append(
            f"Root Cause Discipline average {rc_score:.1f}% below threshold {config.root_cause_discipline_threshold:.1f}%"
        )

    eb_score = dimension_averages.get("evidence_boundary", 0.0)
    if eb_score < config.evidence_boundary_threshold:
        reasons.append(
            f"Evidence Boundary average {eb_score:.1f}% below threshold {config.evidence_boundary_threshold:.1f}%"
        )

    fact_score = dimension_averages.get("fact_preservation", 0.0)
    if fact_score < config.fact_preservation_threshold:
        reasons.append(
            f"Fact Preservation average {fact_score:.1f}% below threshold {config.fact_preservation_threshold:.1f}%"
        )

    capa_score = dimension_averages.get("capa_discipline", 0.0)
    if capa_score < config.capa_discipline_threshold:
        reasons.append(
            f"CAPA Discipline average {capa_score:.1f}% below threshold {config.capa_discipline_threshold:.1f}%"
        )

    critical_count = sum(1 for f in failures if getattr(f, "severity", None) == Severity.CRITICAL)
    if critical_count > config.max_critical_failures:
        reasons.append(
            f"Critical failures count {critical_count} exceeds maximum allowed {config.max_critical_failures}"
        )

    high_count = sum(1 for f in failures if getattr(f, "severity", None) == Severity.HIGH)
    if high_count > config.max_high_failures:
        reasons.append(
            f"High severity failures count {high_count} exceeds maximum allowed {config.max_high_failures}"
        )

    passed = len(reasons) == 0
    return ReadinessResult(passed=passed, reasons=reasons, config_used=config)
