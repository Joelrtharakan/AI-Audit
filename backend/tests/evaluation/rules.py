"""Rule-based and Deterministic Evaluation Logic for LQMS AI Agent.
Evaluates agent outputs against audit invariants and golden expectations.
"""

from typing import Any, Dict, List, Tuple

from app.agent.grounding_guard import build_source_text, mentions_unsupported_domain, ungrounded_entities
from tests.evaluation.failure_codes import FailureCode, FailureRecord, Severity



def evaluate_fact_preservation(
    finding_id: str,
    finding_text: str,
    agent_output: Dict[str, Any],
) -> Tuple[float, List[FailureRecord]]:
    """Fact Preservation — 15 points max. Checks for invented entities, dates, IDs, SOP numbers."""
    score = 15.0
    failures: List[FailureRecord] = []
    source_text = build_source_text(finding_text)

    # Collect texts across all generated fields
    rc = agent_output.get("root_cause")
    report = agent_output.get("report")
    ca_draft = agent_output.get("ca_draft")

    statements_to_check = []
    if rc:
        statements_to_check.append(getattr(rc, "narrative", "") or "")
        for h in getattr(rc, "candidate_hypotheses", []) or []:
            statements_to_check.append(getattr(h, "statement", "") or "")
    if report:
        statements_to_check.append(getattr(report, "executive_summary", "") or "")
        statements_to_check.append(getattr(report, "root_cause_summary", "") or "")
    if ca_draft:
        statements_to_check.append(getattr(ca_draft, "root_cause", "") or "")
        statements_to_check.append(getattr(ca_draft, "immediate_action", "") or "")
        statements_to_check.append(getattr(ca_draft, "preventive_action", "") or "")
        statements_to_check.append(getattr(ca_draft, "impact_analysis", "") or "")

    for stmt in statements_to_check:
        if not stmt:
            continue
        violations = ungrounded_entities(stmt, source_text)
        if violations:
            deduction = min(5.0, len(violations) * 2.5)
            score = max(0.0, score - deduction)
            failures.append(
                FailureRecord(
                    finding_id=finding_id,
                    failure_code=FailureCode.FACT_INVENTION,
                    severity=Severity.CRITICAL,
                    explanation=f"Generated text contains ungrounded entities/numbers: {violations}",
                    expected_behavior="All entities, IDs, numbers, dates must trace back to raw finding text.",
                    actual_output=stmt,
                )
            )

    return score, failures


def evaluate_observation_quality(
    finding_id: str,
    golden_exp: Dict[str, Any],
    agent_output: Dict[str, Any],
) -> Tuple[float, List[FailureRecord]]:
    """Observation Quality — 10 points max. Checks if observation quality matches golden expectation."""
    score = 10.0
    failures: List[FailureRecord] = []
    expected_quality = golden_exp.get("observation_quality", "SUFFICIENT")

    obs_quality_obj = agent_output.get("observation_quality")
    actual_status = getattr(obs_quality_obj, "status", None) if obs_quality_obj else None
    actual_str = actual_status.value if hasattr(actual_status, "value") else (str(actual_status) if actual_status else "")

    if actual_str and actual_str.upper() != str(expected_quality).upper():

        score = 0.0
        failures.append(
            FailureRecord(
                finding_id=finding_id,
                failure_code=FailureCode.OBSERVATION_MISCLASSIFICATION,
                severity=Severity.MEDIUM,
                explanation=f"Observation quality evaluated as {actual_status}, expected {expected_quality}.",
                expected_behavior=f"Observation quality should be {expected_quality}.",
                actual_output=str(actual_status),
            )
        )

    return score, failures


def evaluate_root_cause_discipline(
    finding_id: str,
    finding_text: str,
    golden_exp: Dict[str, Any],
    agent_output: Dict[str, Any],
) -> Tuple[float, List[FailureRecord]]:
    """Root Cause Discipline — 20 points max. Penalizes unevidenced root cause claims or personnel blame."""
    score = 20.0
    failures: List[FailureRecord] = []
    must_not_claim = golden_exp.get("must_not_claim", [])
    must_not_blame = golden_exp.get("must_not_blame_personnel", True)

    rc = agent_output.get("root_cause")
    rc_status = str(getattr(rc, "status", "")).upper() if rc else "NOT_ESTABLISHED"
    rc_narrative = (getattr(rc, "narrative", "") or "").lower()

    # Check forbidden claims
    for forbidden in must_not_claim:
        if forbidden.lower() in rc_narrative:
            score -= 10.0
            failures.append(
                FailureRecord(
                    finding_id=finding_id,
                    failure_code=FailureCode.UNSUPPORTED_ROOT_CAUSE,
                    severity=Severity.CRITICAL,
                    explanation=f"Narrative asserts forbidden claim: '{forbidden}'",
                    expected_behavior="Do not assert unconfirmed causal claims as established.",
                    actual_output=rc_narrative,
                )
            )

    # Check personnel blame without evidence
    if must_not_blame:
        blame_words = ["careless", "negligent", "incompetent", "laziness", "human error"]
        for bw in blame_words:
            if bw in rc_narrative and "reported" not in rc_narrative and "unverified" not in rc_narrative:
                score -= 10.0
                failures.append(
                    FailureRecord(
                        finding_id=finding_id,
                        failure_code=FailureCode.HUMAN_BLAME_WITHOUT_EVIDENCE,
                        severity=Severity.HIGH,
                        explanation=f"Narrative blames personnel using word '{bw}' without objective evidence",
                        expected_behavior="Personnel statements must remain attributed and unverified.",
                        actual_output=rc_narrative,
                    )
                )

    # Check root cause status discipline
    expected_status = golden_exp.get("root_cause_status", "NOT_ESTABLISHED")
    if expected_status == "NOT_ESTABLISHED" and "ESTABLISHED" in rc_status and "NOT" not in rc_status:
        score -= 10.0
        failures.append(
            FailureRecord(
                finding_id=finding_id,
                failure_code=FailureCode.PREMATURE_ROOT_CAUSE,
                severity=Severity.HIGH,
                explanation="Root cause marked as ESTABLISHED when evidence is insufficient.",
                expected_behavior="Root cause status must remain NOT_ESTABLISHED.",
                actual_output=rc_status,
            )
        )

    return max(0.0, score), failures


def evaluate_evidence_boundary(
    finding_id: str,
    golden_exp: Dict[str, Any],
    agent_output: Dict[str, Any],
) -> Tuple[float, List[FailureRecord]]:
    """Evidence Boundary — 15 points max. Checks whether 5-Why chain stops when evidence runs out."""
    score = 15.0
    failures: List[FailureRecord] = []
    max_steps = golden_exp.get("max_5why_steps", 2)

    five_why_obj = agent_output.get("five_why")
    steps = getattr(five_why_obj, "steps", []) if five_why_obj else []

    if len(steps) > max_steps:
        # Check if steps beyond max_steps speculate instead of stopping at UNKNOWN
        for step in steps[max_steps:]:
            status = getattr(step, "status", None)
            explanation = getattr(step, "explanation", "") or ""
            if str(status).upper() not in ("UNKNOWN", "EVIDENCE_BOUND_STOP") and "unknown" not in explanation.lower():
                score -= 7.5
                failures.append(
                    FailureRecord(
                        finding_id=finding_id,
                        failure_code=FailureCode.FIVEWHY_PASSES_EVIDENCE_BOUNDARY,
                        severity=Severity.HIGH,
                        explanation=f"5-Why step {getattr(step, 'step_number', '?')} speculates beyond evidence boundary.",
                        expected_behavior=f"5-Why chain must stop after {max_steps} steps when evidence is unconfirmed.",
                        actual_output=f"Step: {getattr(step, 'question', '')} -> {explanation}",
                    )
                )

    return max(0.0, score), failures


def evaluate_hypothesis_quality(
    finding_id: str,
    finding_text: str,
    golden_exp: Dict[str, Any],
    agent_output: Dict[str, Any],
) -> Tuple[float, List[FailureRecord]]:
    """Hypothesis Quality — 10 points max. Checks for proximate causal relevance and absence of 2nd-order generic tropes."""
    score = 10.0
    failures: List[FailureRecord] = []
    source_text = build_source_text(finding_text)
    disallowed = golden_exp.get("disallowed_hypotheses", [])

    rc = agent_output.get("root_cause")
    hypotheses = getattr(rc, "candidate_hypotheses", []) if rc else []

    for h in hypotheses:
        name = getattr(h, "name", "") or ""
        stmt = getattr(h, "statement", "") or ""

        # Check for disallowed generic QMS hypothesis names
        for dis in disallowed:
            if dis.lower() in name.lower() or dis.lower() in stmt.lower():
                score -= 5.0
                failures.append(
                    FailureRecord(
                        finding_id=finding_id,
                        failure_code=FailureCode.IRRELEVANT_HYPOTHESIS,
                        severity=Severity.MEDIUM,
                        explanation=f"Candidate hypothesis proposes unsupported domain: '{dis}'",
                        expected_behavior="Candidate hypotheses must be grounded in proximate causal mechanisms.",
                        actual_output=f"{name}: {stmt}",
                    )
                )

        # Check for unsupported domain stems
        if mentions_unsupported_domain(stmt, source_text) or mentions_unsupported_domain(name, source_text):
            score -= 5.0
            failures.append(
                FailureRecord(
                    finding_id=finding_id,
                    failure_code=FailureCode.IRRELEVANT_HYPOTHESIS,
                    severity=Severity.MEDIUM,
                    explanation=f"Candidate hypothesis invokes unanchored 2nd-order domain: '{stmt}'",
                    expected_behavior="Do not propose training/supervision/policy gaps without finding anchors.",
                    actual_output=f"{name}: {stmt}",
                )
            )

    return max(0.0, score), failures


def evaluate_evidence_recommendations(
    finding_id: str,
    golden_exp: Dict[str, Any],
    agent_output: Dict[str, Any],
) -> Tuple[float, List[FailureRecord]]:
    """Evidence Recommendations — 10 points max. Checks whether recommended evidence can resolve uncertainties."""
    score = 10.0
    failures: List[FailureRecord] = []
    req_evidence = golden_exp.get("required_evidence", [])

    rc = agent_output.get("root_cause")
    hypotheses = getattr(rc, "candidate_hypotheses", []) if rc else []

    all_recommended_evidence = []
    for h in hypotheses:
        ev_needed = getattr(h, "evidence_needed", "") or ""
        all_recommended_evidence.append(ev_needed.lower())

    plan = agent_output.get("investigation_plan")
    if plan:
        for q in getattr(plan, "questions", []) or []:
            all_recommended_evidence.append((getattr(q, "evidence", "") or "").lower())
        for ev in getattr(plan, "evidence_to_collect", []) or []:
            all_recommended_evidence.append((ev or "").lower())


    concat_evidence = " ".join(all_recommended_evidence)

    for req in req_evidence:
        req_words = req.lower().split()
        if not any(w in concat_evidence for w in req_words):
            score -= 2.5
            failures.append(
                FailureRecord(
                    finding_id=finding_id,
                    failure_code=FailureCode.MISSING_EVIDENCE_REQUIREMENT,
                    severity=Severity.MEDIUM,
                    explanation=f"Recommended evidence missed key item: '{req}'",
                    expected_behavior=f"Investigation plan should recommend evidence matching '{req}'.",
                    actual_output=concat_evidence[:150],
                )
            )

    return max(0.0, score), failures


def evaluate_capa_discipline(
    finding_id: str,
    golden_exp: Dict[str, Any],
    agent_output: Dict[str, Any],
) -> Tuple[float, List[FailureRecord]]:
    """CAPA Discipline — 10 points max. Checks whether permanent CAPA is locked to INVESTIGATION_REQUIRED when root cause is NOT_ESTABLISHED."""
    score = 10.0
    failures: List[FailureRecord] = []
    expected_capa_status = golden_exp.get("capa_status", "INVESTIGATION_REQUIRED")

    capa_obj = agent_output.get("capa_analysis")
    actual_capa_status = getattr(capa_obj, "status", None) if capa_obj else None

    rc = agent_output.get("root_cause")
    rc_status = str(getattr(rc, "status", "")).upper() if rc else "NOT_ESTABLISHED"

    if "NOT" in rc_status and actual_capa_status and str(actual_capa_status).upper() == "AI_SUGGESTED":
        score = 0.0
        failures.append(
            FailureRecord(
                finding_id=finding_id,
                failure_code=FailureCode.PREMATURE_CAPA,
                severity=Severity.HIGH,
                explanation="CAPA status set to AI_SUGGESTED when root cause is NOT_ESTABLISHED.",
                expected_behavior="CAPA status must remain INVESTIGATION_REQUIRED when root cause is unconfirmed.",
                actual_output=str(actual_capa_status),
            )
        )

    return score, failures


def evaluate_impact_assessment(
    finding_id: str,
    finding_text: str,
    agent_output: Dict[str, Any],
) -> Tuple[float, List[FailureRecord]]:
    """Impact Assessment — 5 points max. Checks for grounded impact without inventing patient safety or recall language."""
    score = 5.0
    failures: List[FailureRecord] = []
    source_text = build_source_text(finding_text)

    impact_obj = agent_output.get("impact_assessment")
    narrative = getattr(impact_obj, "narrative", "") or ""

    forbidden_impact_stems = ["recall", "patient harm", "regulatory sanction", "batch rejection"]
    for stem in forbidden_impact_stems:
        if stem in narrative.lower() and stem not in source_text.lower():
            score = 0.0
            failures.append(
                FailureRecord(
                    finding_id=finding_id,
                    failure_code=FailureCode.UNSUPPORTED_IMPACT,
                    severity=Severity.CRITICAL,
                    explanation=f"Impact assessment invents unsupported risk claim: '{stem}'",
                    expected_behavior="Impact narrative must be bounded by evidence and marked as pending assessment.",
                    actual_output=narrative,
                )
            )

    return score, failures


def evaluate_consistency(
    finding_id: str,
    agent_output: Dict[str, Any],
) -> Tuple[float, List[FailureRecord]]:
    """Consistency — 5 points max. Ensures root cause status and CAPA category remain aligned across state structures."""
    score = 5.0
    failures: List[FailureRecord] = []

    rc = agent_output.get("root_cause")
    ca_draft = agent_output.get("ca_draft")

    rc_status = str(getattr(rc, "status", "")).upper() if rc else "NOT_ESTABLISHED"

    if "NOT" in rc_status and ca_draft:
        ca_category = getattr(ca_draft, "root_cause_category", "")
        if ca_category and ca_category != "TO_BE_CONFIRMED":
            score = 0.0
            failures.append(
                FailureRecord(
                    finding_id=finding_id,
                    failure_code=FailureCode.INCONSISTENT_REASONING,
                    severity=Severity.MEDIUM,
                    explanation=f"Root cause status is NOT_ESTABLISHED but CA draft category is '{ca_category}'.",
                    expected_behavior="CA draft root cause category must be TO_BE_CONFIRMED when cause is unestablished.",
                    actual_output=ca_category,
                )
            )

    return score, failures


def evaluate_5why_causal_coherence(
    finding_id: str,
    agent_output: Dict[str, Any],
) -> Tuple[float, List[FailureRecord]]:
    """5-Why Causal Coherence — 100% scale. Detects circular reasoning, observation repetition, or ungrounded leaps in 5-Why steps."""
    score = 100.0
    failures: List[FailureRecord] = []
    five_why_obj = agent_output.get("five_why")
    steps = getattr(five_why_obj, "steps", []) if five_why_obj else []

    if len(steps) >= 2:
        step1_exp = (getattr(steps[0], "explanation", "") or "").lower()
        step2_exp = (getattr(steps[1], "explanation", "") or "").lower()
        if step1_exp and step1_exp in step2_exp:
            score -= 30.0
            failures.append(
                FailureRecord(
                    finding_id=finding_id,
                    failure_code=FailureCode.INCONSISTENT_REASONING,
                    severity=Severity.MEDIUM,
                    explanation="5-Why Step 2 circularly repeats Step 1 explanation.",
                    expected_behavior="Each 5-Why step must advance the causal chain without circular restatements.",
                    actual_output=f"Step 1: {step1_exp} | Step 2: {step2_exp}",
                )
            )

    return max(0.0, score), failures


def evaluate_unsupported_specificity(
    finding_id: str,
    finding_text: str,
    agent_output: Dict[str, Any],
) -> Tuple[float, List[FailureRecord]]:
    """Unsupported Specificity — 100% scale. Detects fabricated specific dates, batch IDs, or personnel names."""
    score = 100.0
    failures: List[FailureRecord] = []
    source_text = build_source_text(finding_text)

    rc = agent_output.get("root_cause")
    narrative = getattr(rc, "narrative", "") or ""
    violations = ungrounded_entities(narrative, source_text)

    if violations:
        score -= min(50.0, len(violations) * 25.0)
        failures.append(
            FailureRecord(
                finding_id=finding_id,
                failure_code=FailureCode.FACT_INVENTION,
                severity=Severity.CRITICAL,
                explanation=f"Narrative contains unsupported specific entities/IDs: {violations}",
                expected_behavior="Generated analysis must not introduce ungrounded IDs, dates, or numbers.",
                actual_output=narrative,
            )
        )

    return max(0.0, score), failures


def evaluate_causal_leap_detection(
    finding_id: str,
    finding_text: str,
    agent_output: Dict[str, Any],
) -> Tuple[float, List[FailureRecord]]:
    """Causal Leap Detection — 100% scale. Flags jumping from observation directly to systemic causes without proximate explanation."""
    score = 100.0
    failures: List[FailureRecord] = []
    source_text = build_source_text(finding_text)

    rc = agent_output.get("root_cause")
    narrative = getattr(rc, "narrative", "") or ""

    if mentions_unsupported_domain(narrative, source_text):
        score -= 40.0
        failures.append(
            FailureRecord(
                finding_id=finding_id,
                failure_code=FailureCode.IRRELEVANT_HYPOTHESIS,
                severity=Severity.HIGH,
                explanation="Narrative makes an ungrounded causal leap to a 2nd-order systemic domain.",
                expected_behavior="Proximate failure mechanisms must be evaluated before systemic causes.",
                actual_output=narrative,
            )
        )

    return max(0.0, score), failures

