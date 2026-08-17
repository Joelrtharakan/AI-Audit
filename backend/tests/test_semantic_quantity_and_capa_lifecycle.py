import pytest
from app.services.semantic_subject import (
    resolve_deviation,
    extract_date,
    topic_word,
    strip_quantity_prefix,
    extract_incidental_quantity,
    _clean_subject,
)
from app.agent.nodes.plan_investigation_fallback import (
    build_recurrence_investigation_questions,
    build_deterministic_investigation_plan,
)
from app.models.agent import (
    CanonicalFindingState,
    EvidenceItem,
    EvidenceStatus,
    CandidateHypothesis,
    PropositionType,
    CausalLevel,
    SupportLevel,
)
from app.agent.nodes.core_synthesis import _derive_deterministic_impact
from app.agent.nodes.five_why_fallback import build_deterministic_five_why
from app.agent.causal_guard import is_evidence_state_not_hypothesis
from app.agent.causal_graph import evaluate_root_cause_eligibility


def test_quantity_leakage_elimination_and_topic_derivation():
    finding_text = (
        "During an internal audit of the Quality Control Laboratory, three "
        "temperature-monitoring records for refrigerator QC-REF-02 were missing between 10 and 12 August."
    )
    # 1. Subject extraction: clean subject must not start with "three"
    resolved = resolve_deviation(finding_text, [])
    assert resolved.subject is not None
    assert not resolved.subject.lower().startswith("three")
    assert "temperature-monitoring records for refrigerator QC-REF-02" in resolved.subject

    # 2. Topic word must be "temperature", NEVER "three"
    topic = topic_word(resolved.subject)
    assert topic == "temperature"
    assert topic != "three"

    # 3. Quantity extraction
    qty = extract_incidental_quantity(finding_text)
    assert qty == 3

    # 4. Date extraction must extract "10–12 August" or "between 10 and 12 August", NOT None/UNKNOWN
    extracted_d = extract_date(finding_text)
    assert extracted_d is not None
    assert "10" in extracted_d and "12" in extracted_d and "August" in extracted_d


def test_process_at_risk_and_affected_object_no_quantity():
    finding_text = (
        "During an internal audit of the Quality Control Laboratory, three "
        "temperature-monitoring records for refrigerator QC-REF-02 were missing between 10 and 12 August."
    )
    resolved = resolve_deviation(finding_text, [])
    canonical = CanonicalFindingState(
        raw_finding=finding_text,
        finding_subject=resolved.subject,
        observed_deviation=resolved.deviation,
        deviation_condition="missing",
        affected_period="10–12 August",
        facts=[],
        reported_statements=[],
    )

    impact, clean_noun, topic, actor = _derive_deterministic_impact(
        finding_text, canonical, canonical.observed_deviation
    )

    assert impact.process_at_risk is not None
    assert not impact.process_at_risk.lower().startswith("three")
    assert "temperature monitoring and record control" in impact.process_at_risk.lower()
    assert impact.affected_period == "10–12 August"
    assert not impact.affected_object.lower().startswith("three")


def test_recurrence_investigation_questions_are_independent_propositions():
    subject = "temperature-monitoring records for refrigerator QC-REF-02"
    topic = "temperature"

    # Must produce 5 independent, single-proposition questions
    questions = build_recurrence_investigation_questions(subject, topic)
    assert len(questions) == 5
    for q in questions:
        assert q.target_proposition_id is not None
        assert q.decision_rule is not None
        assert q.resolves is not None
        assert q.evidence_required is not None
        assert "three" not in q.question.lower()

    q_texts = " ".join(q.question for q in questions)
    assert "fully implemented" in q_texts.lower()
    assert "effectiveness review" in q_texts.lower()
    assert "effectiveness criterion" in q_texts.lower()
    assert "scope" in q_texts.lower()
    assert "same causal mechanism" in q_texts.lower()


def test_evidence_state_not_promoted_to_hypothesis():
    statement = (
        "A previous corrective action was identified for similar temperature-related findings, "
        "but whether it was implemented, verified effective, and causally connected to this specific "
        "recurrence is unconfirmed."
    )
    name = "TEMPERATURE_PREVIOUS_CAPA_STATUS_UNCONFIRMED"

    # Must be identified as an evidence-state proposition
    assert is_evidence_state_not_hypothesis(statement, name)

    h = CandidateHypothesis(
        id="H2",
        name=name,
        statement=statement,
        status="POSSIBLE",
        evidence_needed="Previous CAPA records",
    )

    eligible, support_level, reason, missing, causal_level, promo = evaluate_root_cause_eligibility(
        h, source_text="finding text", evidence_items=[]
    )
    assert not eligible
    assert support_level == SupportLevel.REJECTED
    assert causal_level == CausalLevel.EVIDENCE_STATE
    assert not promo


def test_positive_causal_promotion_still_works():
    finding_text = "Audit log showed user deliberately bypassed workflow validation control bal-auth-01 on balance BAL-014."
    evidence = [
        EvidenceItem(
            claim="audit log showed user deliberately bypassed workflow validation control bal-auth-01",
            source="SYSTEM_AUDIT_TRAIL",
            status=EvidenceStatus.VERIFIED,
        )
    ]
    h = CandidateHypothesis(
        id="H1",
        name="WORKFLOW_VALIDATION_BYPASS",
        statement="The workflow validation control was bypassed, allowing the invalid record to be released.",
        status="SUPPORTED",
        evidence_needed="System audit logs",
    )

    eligible, support_level, reason, missing, causal_level, promo = evaluate_root_cause_eligibility(
        h, source_text=finding_text, evidence_items=evidence
    )
    assert eligible
    assert support_level in (SupportLevel.SUPPORTED, SupportLevel.ESTABLISHED, SupportLevel.VERIFIED)
    assert promo


def test_five_why_evidence_boundary_no_breakdown_framing():
    finding_text = (
        "During an internal audit of the Quality Control Laboratory, three "
        "temperature-monitoring records for refrigerator QC-REF-02 were missing between 10 and 12 August. "
        "The technician stated that they had not received retraining on SOP-LAB-021. "
        "The supervisor stated that all affected staff were informed of the revised requirements."
    )
    claims = [
        EvidenceItem(claim="three temperature-monitoring records for refrigerator QC-REF-02 were missing between 10 and 12 August", source="AUDITOR_FINDING", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="the technician stated they had not received retraining on SOP-LAB-021", source="REPORTED_STATEMENT", status=EvidenceStatus.REPORTED),
        EvidenceItem(claim="the supervisor stated all affected staff were informed of the revised requirements", source="REPORTED_STATEMENT", status=EvidenceStatus.REPORTED),
    ]
    resolved = resolve_deviation(finding_text, [c.claim for c in claims if c.status == EvidenceStatus.VERIFIED])
    canonical = CanonicalFindingState(
        raw_finding=finding_text,
        finding_subject=resolved.subject,
        observed_deviation=resolved.deviation,
        deviation_condition="missing",
        affected_period="10–12 August",
        facts=[claims[0].claim],
        reported_statements=[claims[1].claim, claims[2].claim],
    )

    five_why = build_deterministic_five_why(finding_text, claims, canonical_subject=canonical.finding_subject)
    assert not five_why.is_complete
    assert len(five_why.steps) == 1
    assert five_why.steps[0].status == "UNKNOWN"
    assert "breakdown" not in five_why.steps[0].question.lower()
    assert "does not establish whether the checks were not performed" in five_why.steps[0].answer.lower()


def test_investigation_plan_survives_zero_hypotheses():
    from app.agent.causal_guard import should_generate_investigation_plan
    from app.models.agent import RootCauseAnalysis, RootCauseStatus

    finding_text = (
        "During an internal audit of the Quality Control Laboratory, three "
        "temperature-monitoring records for refrigerator QC-REF-02 were missing between 10 and 12 August."
    )
    rc = RootCauseAnalysis(
        status=RootCauseStatus.NOT_ESTABLISHED,
        candidate_hypotheses=[],
        leading_hypothesis=None,
    )
    ledger = []

    # Must require investigation plan
    assert should_generate_investigation_plan(evidence_ledger=ledger, root_cause=rc, finding_text=finding_text)

    hyps, plan = build_deterministic_investigation_plan(finding_text, ledger)
    assert len(hyps) == 0  # 0 candidate hypotheses
    assert len(plan.questions) >= 5  # at least 5 targeted questions generated
    assert len(plan.areas) >= 3  # at least 3 investigation areas
    for q in plan.questions:
        assert q.id is not None
        assert q.target_proposition_id is not None
        assert q.resolves is not None
        assert q.evidence_required is not None
        assert q.decision_rule is not None


def test_conflicting_evidence_and_missing_document_generate_questions():
    finding_text = (
        "During an internal audit, the operator stated that retraining was not provided. "
        "The supervisor stated that all staff were informed. Calibration certificate was unavailable."
    )
    ledger = [
        EvidenceItem(claim="retraining was not provided", source="REPORTED_STATEMENT", status=EvidenceStatus.REPORTED),
        EvidenceItem(claim="all staff were informed", source="REPORTED_STATEMENT", status=EvidenceStatus.REPORTED),
    ]
    hyps, plan = build_deterministic_investigation_plan(finding_text, ledger)
    assert len(plan.questions) >= 3
    assert any("training" in q.question.lower() or "revision" in q.question.lower() or "procedure" in q.question.lower() for q in plan.questions)
    assert len(plan.areas) >= 2


def test_investigation_areas_aggregation_from_questions():
    from app.agent.analytical_validator import derive_investigation_areas
    from app.models.agent import InvestigationQuestion

    questions = [
        InvestigationQuestion(
            id="Q1", target_proposition_id="P_GOV", question="What procedure governs QC-REF-02?", resolves="Applicable requirement"
        ),
        InvestigationQuestion(
            id="Q2", target_proposition_id="P_TRN", question="Was retraining completed?", resolves="Retraining completion verification"
        ),
        InvestigationQuestion(
            id="Q3", target_proposition_id="P_REC_1", question="Was previous CAPA implemented?", resolves="Previous CAPA implementation verification"
        ),
        InvestigationQuestion(
            id="Q4", target_proposition_id="P_REC_2", question="Was previous CAPA effective?", resolves="Previous CAPA effectiveness verification"
        ),
    ]
    areas = derive_investigation_areas(candidate_hypotheses=[], questions=questions, canonical_subject="retraining")
    # Investigation areas are concise clusters, not one area per question
    # (Section 10: "Normally output 2-4 investigation areas") -- previous
    # CAPA implementation/effectiveness collapse into ONE area distinct from
    # the current-finding domain areas, never duplicated or split further.
    assert 2 <= len(areas) <= 4
    assert len(areas) == len(set(a.lower() for a in areas)), "no duplicate areas"
    area_text = " ".join(areas).lower()
    assert "capa implementation and effectiveness" in area_text
    assert "requirement" in area_text or "completion" in area_text or "retraining" in area_text


