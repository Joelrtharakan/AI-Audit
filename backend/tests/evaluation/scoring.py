"""Scoring Engine for 100-Point Evaluation Rubric.
Aggregates scores across 9 dimensions:
  1. Fact Preservation (15 pts)
  2. Observation Quality (10 pts)
  3. Root Cause Discipline (20 pts)
  4. Evidence Boundary (15 pts)
  5. Hypothesis Quality (10 pts)
  6. Evidence Recommendations (10 pts)
  7. CAPA Discipline (10 pts)
  8. Impact Assessment (5 pts)
  9. Consistency (5 pts)
"""

from typing import Any, Dict, List, NamedTuple, Tuple
from tests.evaluation.failure_codes import FailureRecord
from tests.evaluation.rules import (
    evaluate_5why_causal_coherence,
    evaluate_capa_discipline,
    evaluate_causal_leap_detection,
    evaluate_consistency,
    evaluate_evidence_boundary,
    evaluate_evidence_recommendations,
    evaluate_fact_preservation,
    evaluate_hypothesis_quality,
    evaluate_impact_assessment,
    evaluate_observation_quality,
    evaluate_root_cause_discipline,
    evaluate_unsupported_specificity,
)
from tests.evaluation.semantic_judge import SemanticJudge


class FindingScoreResult(NamedTuple):
    finding_id: str
    total_score: float
    dimension_scores: Dict[str, float]
    failures: List[FailureRecord]


def score_finding_output(
    finding_id: str,
    finding_text: str,
    golden_exp: Dict[str, Any],
    agent_output: Dict[str, Any],
) -> FindingScoreResult:
    all_failures: List[FailureRecord] = []

    # 1. Fact Preservation (15 pts)
    fact_score, f_fact = evaluate_fact_preservation(finding_id, finding_text, agent_output)
    all_failures.extend(f_fact)

    # 2. Observation Quality (10 pts)
    obs_score, f_obs = evaluate_observation_quality(finding_id, golden_exp, agent_output)
    all_failures.extend(f_obs)

    # 3. Root Cause Discipline (20 pts)
    rc_score, f_rc = evaluate_root_cause_discipline(finding_id, finding_text, golden_exp, agent_output)
    all_failures.extend(f_rc)

    # 4. Evidence Boundary (15 pts)
    ev_bound_score, f_eb = evaluate_evidence_boundary(finding_id, golden_exp, agent_output)
    all_failures.extend(f_eb)

    # 5. Hypothesis Quality (10 pts)
    hyp_score, f_hyp = evaluate_hypothesis_quality(finding_id, finding_text, golden_exp, agent_output)
    all_failures.extend(f_hyp)

    # 6. Evidence Recommendations (10 pts)
    ev_rec_score, f_er = evaluate_evidence_recommendations(finding_id, golden_exp, agent_output)
    all_failures.extend(f_er)

    # 7. CAPA Discipline (10 pts)
    capa_score, f_capa = evaluate_capa_discipline(finding_id, golden_exp, agent_output)
    all_failures.extend(f_capa)

    # 8. Impact Assessment (5 pts)
    impact_score, f_imp = evaluate_impact_assessment(finding_id, finding_text, agent_output)
    all_failures.extend(f_imp)

    # 9. Consistency (5 pts)
    cons_score, f_cons = evaluate_consistency(finding_id, agent_output)
    all_failures.extend(f_cons)

    # Dedicated Analytical Quality Dimensions (100% scale)
    fw_coherence, f_fw = evaluate_5why_causal_coherence(finding_id, agent_output)
    all_failures.extend(f_fw)

    unsup_spec, f_us = evaluate_unsupported_specificity(finding_id, finding_text, agent_output)
    all_failures.extend(f_us)

    causal_leap, f_cl = evaluate_causal_leap_detection(finding_id, finding_text, agent_output)
    all_failures.extend(f_cl)

    # Hybrid Semantic Judge
    judge = SemanticJudge()
    sem_score, f_sem = judge.evaluate_semantic_alignment(finding_id, finding_text, golden_exp, agent_output)
    all_failures.extend(f_sem)

    total_score = (
        fact_score
        + obs_score
        + rc_score
        + ev_bound_score
        + hyp_score
        + ev_rec_score
        + capa_score
        + impact_score
        + cons_score
    )

    dimension_scores = {
        "fact_preservation": round(fact_score / 15.0 * 100, 1),
        "observation_quality": round(obs_score / 10.0 * 100, 1),
        "root_cause_discipline": round(rc_score / 20.0 * 100, 1),
        "evidence_boundary": round(ev_bound_score / 15.0 * 100, 1),
        "hypothesis_quality": round(hyp_score / 10.0 * 100, 1),
        "evidence_recommendations": round(ev_rec_score / 10.0 * 100, 1),
        "capa_discipline": round(capa_score / 10.0 * 100, 1),
        "impact_assessment": round(impact_score / 5.0 * 100, 1),
        "consistency": round(cons_score / 5.0 * 100, 1),
        "5why_causal_coherence": round(fw_coherence, 1),
        "unsupported_specificity": round(unsup_spec, 1),
        "causal_leap_detection": round(causal_leap, 1),
    }

    return FindingScoreResult(
        finding_id=finding_id,
        total_score=round(total_score, 1),
        dimension_scores=dimension_scores,
        failures=all_failures,
    )

