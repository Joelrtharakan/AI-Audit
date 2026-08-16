"""Unit tests for the analytical validation firewall
(app/agent/analytical_validator.py) — the structural, deterministic layer
that runs after causal synthesis and before the final report is assembled.

These are pure-function tests (no finding-specific content, no domain
keywords) verifying the general invariants: leading-hypothesis selection,
root-cause certainty monotonicity, 5-Why mechanism-skip repair, contributing
factor established/potential classification, and CAPA causal linkage.
"""

from __future__ import annotations

from app.agent.analytical_validator import (
    classify_contributing_factors,
    classify_contributing_factors_full,
    classify_impact_field_basis,
    compute_analytical_quality,
    compute_impact_field_basis,
    conditional_action_has_causal_linkage,
    derive_investigation_questions,
    five_why_skips_available_mechanism,
    hypothesis_confidence,
    leading_hypothesis_confidence,
    leading_hypothesis_display,
    leading_hypothesis_status,
    repair_five_why_with_mechanism,
    select_leading_hypothesis,
    validate_capa_causal_linkage,
    validate_causal_graph,
    validate_investigation_question,
    validate_root_cause_state,
)
from app.agent.causal_guard import MechanismInfo
from app.models.agent import (
    CandidateHypothesis,
    CapaAnalysis,
    CapaStatus,
    ConditionalCapaAction,
    ContributingFactor,
    EvidenceItem,
    EvidenceStatus,
    FiveWhyStep,
    RootCauseAnalysis,
)


def _hyp(id_, status="POSSIBLE", rank="HIGH", statement=None, name="CAUSE"):
    return CandidateHypothesis(
        id=id_,
        name=name,
        statement=statement or f"Statement for {id_}",
        status=status,
        evidence_needed="Some record",
        relevance_rank=rank,
    )


# ---------------------------------------------------------------------------
# select_leading_hypothesis
# ---------------------------------------------------------------------------


def test_no_hypotheses_no_leader():
    assert select_leading_hypothesis([]) is None


def test_single_supported_is_leader():
    hyps = [_hyp("H1", status="SUPPORTED"), _hyp("H2", status="POSSIBLE", rank="LOW")]
    assert select_leading_hypothesis(hyps).startswith("H1")


def test_tied_supported_hypotheses_no_leader():
    """Two equally-strong SUPPORTED hypotheses are genuinely competing —
    forcing a pick would misrepresent the evidence."""
    hyps = [_hyp("H1", status="SUPPORTED", rank="HIGH"), _hyp("H2", status="SUPPORTED", rank="HIGH")]
    assert select_leading_hypothesis(hyps) is None


def test_tied_possible_hypotheses_no_leader():
    hyps = [_hyp("H1", status="POSSIBLE", rank="HIGH"), _hyp("H2", status="POSSIBLE", rank="HIGH")]
    assert select_leading_hypothesis(hyps) is None


def test_distinct_ranks_best_possible_is_leader():
    hyps = [_hyp("H1", status="POSSIBLE", rank="LOW"), _hyp("H2", status="POSSIBLE", rank="HIGH")]
    assert select_leading_hypothesis(hyps).startswith("H2")


def test_all_refuted_no_leader():
    hyps = [_hyp("H1", status="REFUTED"), _hyp("H2", status="REFUTED")]
    assert select_leading_hypothesis(hyps) is None


def test_leading_hypothesis_confidence_levels():
    hyps = [_hyp("H1", status="SUPPORTED")]
    assert leading_hypothesis_confidence(hyps, "H1 — x") == "MEDIUM"
    hyps2 = [_hyp("H1", status="POSSIBLE")]
    assert leading_hypothesis_confidence(hyps2, "H1 — x") == "LOW"
    assert leading_hypothesis_confidence([], None) == "LOW"


# ---------------------------------------------------------------------------
# leading_hypothesis_status / leading_hypothesis_display
#
# The bug these close: select_leading_hypothesis returning None for BOTH
# "no hypotheses" and "genuinely tied hypotheses" meant a report couldn't
# tell those apart -- both rendered as a blank/"no leading hypothesis"
# field, which reads as a weak/empty analysis even when the LLM actually
# produced two well-reasoned, evenly-matched hypotheses (Class C).
# ---------------------------------------------------------------------------


def test_leading_hypothesis_status_none_when_no_hypotheses():
    assert leading_hypothesis_status([]) == "NONE"
    assert leading_hypothesis_display([]) is None


def test_leading_hypothesis_status_selected_when_single_supported():
    hyps = [_hyp("H1", status="SUPPORTED"), _hyp("H2", status="POSSIBLE", rank="LOW")]
    assert leading_hypothesis_status(hyps) == "SELECTED"
    assert leading_hypothesis_display(hyps).startswith("H1")


def test_leading_hypothesis_status_tied_is_not_none_display():
    """Class C: two hypotheses tied at the same rank must render as an
    explicit TIED result, not disappear into the same blank the empty-list
    case produces."""
    hyps = [_hyp("H1", status="POSSIBLE", rank="HIGH"), _hyp("H2", status="POSSIBLE", rank="HIGH")]
    assert leading_hypothesis_status(hyps) == "TIED"
    assert select_leading_hypothesis(hyps) is None  # unchanged contract
    assert leading_hypothesis_display(hyps) == "NONE — COMPETING HYPOTHESES REMAIN TIED"


def test_leading_hypothesis_status_none_when_all_refuted():
    hyps = [_hyp("H1", status="REFUTED"), _hyp("H2", status="REFUTED")]
    assert leading_hypothesis_status(hyps) == "NONE"
    assert leading_hypothesis_display(hyps) is None


# ---------------------------------------------------------------------------
# hypothesis_confidence
# ---------------------------------------------------------------------------


def test_hypothesis_confidence_supported_high_rank_is_high():
    h = _hyp("H1", status="SUPPORTED", rank="HIGH")
    assert hypothesis_confidence(h) == "HIGH"


def test_hypothesis_confidence_supported_lower_rank_is_medium():
    h = _hyp("H1", status="SUPPORTED", rank="MEDIUM")
    assert hypothesis_confidence(h) == "MEDIUM"


def test_hypothesis_confidence_possible_without_discrimination_is_low():
    h = _hyp("H1", status="POSSIBLE", rank="HIGH")
    assert hypothesis_confidence(h) == "LOW"


def test_hypothesis_confidence_possible_high_rank_with_discrimination_is_medium():
    h = _hyp("H1", status="POSSIBLE", rank="HIGH")
    h.discrimination_evidence = "The calibration certificate would confirm or refute this."
    assert hypothesis_confidence(h) == "MEDIUM"


# ---------------------------------------------------------------------------
# validate_root_cause_state
# ---------------------------------------------------------------------------


def test_verified_root_cause_without_verified_mechanism_downgraded():
    """A REPORTED mechanism (someone's account) never justifies an
    ESTABLISHED-like root cause status, regardless of what else is VERIFIED
    in the ledger (Case A: the observation being VERIFIED doesn't verify
    WHY it happened)."""
    rc = RootCauseAnalysis(status="VERIFIED", narrative="x")
    mechanism = MechanismInfo(statement="someone reported it", status="REPORTED", polarity="non_performance")
    warnings = validate_root_cause_state(rc, mechanism)
    assert rc.status == "STATED_UNVERIFIED"
    assert warnings


def test_verified_root_cause_with_verified_mechanism_survives():
    """Case C: a VERIFIED fact that IS itself the causal mechanism (not
    just the observation) justifies the ESTABLISHED-like status."""
    rc = RootCauseAnalysis(status="VERIFIED", narrative="x")
    mechanism = MechanismInfo(statement="audit trail confirms the cause", status="VERIFIED", polarity="non_performance")
    warnings = validate_root_cause_state(rc, mechanism)
    assert rc.status == "VERIFIED"
    assert warnings == []


def test_verified_root_cause_with_no_mechanism_at_all_downgraded():
    rc = RootCauseAnalysis(status="VERIFIED", narrative="x")
    warnings = validate_root_cause_state(rc, None)
    assert rc.status == "STATED_UNVERIFIED"
    assert warnings


def test_not_established_never_downgraded_further():
    rc = RootCauseAnalysis(status="NOT_ESTABLISHED", narrative="x")
    warnings = validate_root_cause_state(rc, None)
    assert rc.status == "NOT_ESTABLISHED"
    assert warnings == []


# ---------------------------------------------------------------------------
# five_why_skips_available_mechanism / repair_five_why_with_mechanism
# ---------------------------------------------------------------------------


def test_skipped_mechanism_detected():
    mechanism = MechanismInfo(statement="the reviewer confirmed the check was missed", status="REPORTED", polarity="non_performance")
    steps = [FiveWhyStep(question="Why was the record incomplete?", answer="The record was incomplete.", status="VERIFIED")]
    assert five_why_skips_available_mechanism(steps, mechanism) is True


def test_mechanism_reflected_in_chain_not_flagged():
    mechanism = MechanismInfo(statement="the reviewer confirmed the check was missed", status="REPORTED", polarity="non_performance")
    steps = [
        FiveWhyStep(question="Why was the record incomplete?", answer="The record was incomplete.", status="VERIFIED"),
        FiveWhyStep(question="Why did that occur?", answer="The reviewer confirmed the check was missed.", status="REPORTED"),
    ]
    assert five_why_skips_available_mechanism(steps, mechanism) is False


def test_no_mechanism_never_flagged():
    assert five_why_skips_available_mechanism([], MechanismInfo()) is False


def test_repair_inserts_mechanism_step_after_observation():
    mechanism = MechanismInfo(statement="the reviewer confirmed the check was missed", status="REPORTED", polarity="non_performance")
    steps = [FiveWhyStep(question="Why was the record incomplete?", answer="The record was incomplete.", status="VERIFIED")]
    repaired = repair_five_why_with_mechanism(steps, mechanism, "the record was incomplete")
    assert len(repaired) == 2
    assert repaired[1].answer == mechanism.statement
    assert repaired[1].status == "REPORTED"


def test_repair_no_op_when_first_step_already_is_mechanism():
    mechanism = MechanismInfo(statement="the check was missed", status="VERIFIED", polarity="non_performance")
    steps = [FiveWhyStep(question="Why?", answer="the check was missed", status="VERIFIED")]
    repaired = repair_five_why_with_mechanism(steps, mechanism, "the check was missed")
    assert repaired == steps


# ---------------------------------------------------------------------------
# classify_contributing_factors
# ---------------------------------------------------------------------------


def test_contributing_factors_split_established_vs_potential():
    established_cf = ContributingFactor(description="A", evidence_status=EvidenceStatus.VERIFIED, status="VERIFIED")
    potential_cf = ContributingFactor(description="B", evidence_status=EvidenceStatus.INFERRED, status="POSSIBLE_UNCONFIRMED")
    established, potential = classify_contributing_factors([established_cf, potential_cf])
    assert established == [established_cf]
    assert potential == [potential_cf]


def test_empty_contributing_factors_both_empty():
    established, potential = classify_contributing_factors([])
    assert established == []
    assert potential == []


# ---------------------------------------------------------------------------
# CAPA causal linkage
# ---------------------------------------------------------------------------


def test_conditional_action_linked_by_id():
    hyps = [_hyp("H1", statement="task assignment failure")]
    assert conditional_action_has_causal_linkage("If H1 is confirmed", hyps) is True


def test_conditional_action_linked_by_word_overlap():
    hyps = [_hyp("H1", statement="a shift handover control weakness contributed")]
    assert conditional_action_has_causal_linkage("If the handover control weakness is confirmed", hyps) is True


def test_conditional_action_orphaned_dropped():
    hyps = [_hyp("H1", statement="a shift handover control weakness contributed")]
    assert conditional_action_has_causal_linkage("If an unrelated network outage is confirmed", hyps) is False


def test_no_hypotheses_means_no_linkage_required():
    assert conditional_action_has_causal_linkage("Anything at all", []) is True


def test_validate_capa_causal_linkage_drops_orphaned_action():
    hyps = [_hyp("H1", statement="a shift handover control weakness contributed")]
    capa = CapaAnalysis(
        status=CapaStatus.INVESTIGATION_REQUIRED,
        conditional_actions=[
            ConditionalCapaAction(if_cause_confirmed="If H1 is confirmed", recommended_action="Fix handover"),
            ConditionalCapaAction(if_cause_confirmed="If an unrelated satellite malfunction is confirmed", recommended_action="Replace satellite"),
        ],
    )
    warnings = validate_capa_causal_linkage(capa, hyps)
    assert len(capa.conditional_actions) == 1
    assert capa.conditional_actions[0].if_cause_confirmed == "If H1 is confirmed"
    assert warnings


# ---------------------------------------------------------------------------
# compute_analytical_quality — smoke test, all expected keys present
# ---------------------------------------------------------------------------


def test_analytical_quality_returns_all_expected_keys():
    rc = RootCauseAnalysis(status="NOT_ESTABLISHED", narrative="x", candidate_hypotheses=[_hyp("H1")])
    from app.models.agent import FiveWhyAnalysis
    fw = FiveWhyAnalysis(steps=[FiveWhyStep(question="Why?", answer="Because", status="UNKNOWN")])
    capa = CapaAnalysis(status=CapaStatus.INVESTIGATION_REQUIRED)
    mechanism = MechanismInfo(statement="x", status="REPORTED", polarity="non_performance")

    quality = compute_analytical_quality(rc, fw, [], capa, mechanism)
    expected_keys = {
        "mechanism_accuracy",
        "hypothesis_discrimination",
        "five_why_coherence",
        "contributing_factor_quality",
        "capa_linkage",
        "uncertainty_discipline",
    }
    assert expected_keys.issubset(quality.keys())
    for value in quality.values():
        assert value in ("LOW", "MEDIUM", "HIGH")


# ---------------------------------------------------------------------------
# Phase 1: impact field EXPLICIT/INFERRED/UNKNOWN classification
# ---------------------------------------------------------------------------


def test_impact_basis_explicit_when_verbatim_from_finding():
    finding = "The calibration certificate for the measuring instrument was expired."
    assert classify_impact_field_basis("the calibration certificate for the measuring instrument", finding) == "EXPLICIT"


def test_impact_basis_inferred_when_synthesized():
    finding = "The calibration certificate for the measuring instrument was expired."
    assert classify_impact_field_basis("equipment metrology control process", finding) == "INFERRED"


def test_impact_basis_unknown_when_empty_or_placeholder():
    finding = "The calibration certificate for the measuring instrument was expired."
    assert classify_impact_field_basis(None, finding) == "UNKNOWN"
    assert classify_impact_field_basis("", finding) == "UNKNOWN"
    assert classify_impact_field_basis("NOT ESTABLISHED", finding) == "UNKNOWN"
    assert classify_impact_field_basis("Unknown", finding) == "UNKNOWN"


def test_compute_impact_field_basis_covers_all_named_fields():
    from app.models.agent import ImpactAssessment, ImpactStatus

    finding = "The calibration certificate for the measuring instrument was expired."
    impact = ImpactAssessment(
        status=ImpactStatus.IMPACT_REQUIRES_ASSESSMENT,
        affected_object="the calibration certificate for the measuring instrument",
        process_at_risk="equipment metrology control process",
        potential_effect=None,
    )
    basis = compute_impact_field_basis(impact, finding)
    assert basis["affected_object"] == "EXPLICIT"
    assert basis["process_at_risk"] == "INFERRED"
    assert basis["potential_effect"] == "UNKNOWN"
    assert set(basis.keys()) == {"affected_object", "affected_period", "process_at_risk", "relevant_change", "potential_effect"}


def test_compute_impact_field_basis_handles_none_impact():
    basis = compute_impact_field_basis(None, "some finding text")
    assert all(v == "UNKNOWN" for v in basis.values())


# ---------------------------------------------------------------------------
# Phase 3: contributing factor established/potential/rejected split
# ---------------------------------------------------------------------------


def test_contributing_factor_contradicting_verified_completion_rejected():
    factor = ContributingFactor(
        description="A training deficiency may have contributed to the deviation.",
        evidence_status=EvidenceStatus.INFERRED,
        status="POSSIBLE_UNCONFIRMED",
    )
    mechanism = MechanismInfo()
    established, potential, rejected = classify_contributing_factors_full(
        [factor], mechanism, ["training was completed"]
    )
    assert established == []
    assert potential == []
    assert len(rejected) == 1
    assert rejected[0].status == "REJECTED"


def test_contributing_factor_not_contradicting_stays_potential():
    factor = ContributingFactor(
        description="A shift handover gap may have contributed to the deviation.",
        evidence_status=EvidenceStatus.INFERRED,
        status="POSSIBLE_UNCONFIRMED",
    )
    mechanism = MechanismInfo()
    established, potential, rejected = classify_contributing_factors_full(
        [factor], mechanism, ["training was completed"]
    )
    assert established == []
    assert potential == [factor]
    assert rejected == []


def test_contributing_factor_contradicting_mechanism_rejected():
    mechanism = MechanismInfo(statement="the check was missed", status="REPORTED", polarity="non_performance")
    factor = ContributingFactor(
        description="The check may have been performed but not documented in the record.",
        evidence_status=EvidenceStatus.INFERRED,
        status="POSSIBLE_UNCONFIRMED",
    )
    established, potential, rejected = classify_contributing_factors_full([factor], mechanism, [])
    assert len(rejected) == 1


# ---------------------------------------------------------------------------
# Phase 2: consolidated causal-graph audit
# ---------------------------------------------------------------------------


def test_causal_graph_audit_flags_hypothesis_without_rationale():
    from app.models.agent import FiveWhyAnalysis

    hyp = _hyp("H1")
    hyp.rationale = None
    hyp.evidence_needed = ""
    hyp.discrimination_evidence = None
    rc = RootCauseAnalysis(status="NOT_ESTABLISHED", narrative="x", candidate_hypotheses=[hyp])
    violations = validate_causal_graph(rc, FiveWhyAnalysis(), None, MechanismInfo())
    assert any("rationale" in v for v in violations)
    assert any("evidence_needed" in v for v in violations)


def test_causal_graph_audit_flags_systemic_action_without_verification():
    rc = RootCauseAnalysis(status="NOT_ESTABLISHED", narrative="x", candidate_hypotheses=[_hyp("H1")])
    capa = CapaAnalysis(
        status=CapaStatus.INVESTIGATION_REQUIRED,
        conditional_actions=[
            ConditionalCapaAction(
                if_cause_confirmed="If H1 is confirmed",
                recommended_action="Fix it",
                action_type="SYSTEMIC_ACTION",
                verification_method=None,
            )
        ],
    )
    from app.models.agent import FiveWhyAnalysis
    violations = validate_causal_graph(rc, FiveWhyAnalysis(), capa, MechanismInfo())
    assert any("verification_method" in v for v in violations)


def test_causal_graph_audit_clean_state_no_violations():
    from app.models.agent import FiveWhyAnalysis

    hyp = _hyp("H1")
    hyp.rationale = "Because the finding suggests it"
    hyp.evidence_needed = "Some record"
    rc = RootCauseAnalysis(status="NOT_ESTABLISHED", narrative="x", candidate_hypotheses=[hyp])
    capa = CapaAnalysis(
        status=CapaStatus.INVESTIGATION_REQUIRED,
        conditional_actions=[
            ConditionalCapaAction(
                if_cause_confirmed="If H1 is confirmed",
                recommended_action="Fix it",
                action_type="SYSTEMIC_ACTION",
                verification_method="Re-audit after 30 days",
            )
        ],
    )
    violations = validate_causal_graph(rc, FiveWhyAnalysis(), capa, MechanismInfo())
    assert violations == []


# ---------------------------------------------------------------------------
# derive_investigation_questions
#
# Closes the "investigation questions are empty even though hypotheses
# exist" bug: plan_investigation runs BEFORE core_synthesis produces
# hypotheses (and is usually fast-pathed to an empty plan in this
# deployment), so questions must be derivable from the hypotheses
# afterward, not only authored by the earlier planning step.
# ---------------------------------------------------------------------------


def test_derive_investigation_questions_one_per_live_hypothesis():
    hyps = [
        _hyp("H1", statement="The required post-calibration labeling step may not have been completed."),
        _hyp("H2", statement="Calibration record retrieval may have failed.", status="REFUTED"),
        _hyp("H3", statement="The label may have been removed after application."),
    ]
    hyps[0].evidence_needed = "Calibration completion and label application records"
    hyps[2].evidence_needed = "Equipment history and prior inspection records"

    questions = derive_investigation_questions(hyps)

    # REFUTED hypotheses don't need further discrimination.
    assert len(questions) == 2
    assert all(q.evidence for q in questions)
    assert any("labeling step" in q.question for q in questions)


def test_derive_investigation_questions_empty_when_no_hypotheses():
    assert derive_investigation_questions([]) == []


def test_derive_investigation_questions_are_grammatical_not_concatenated():
    """The exact regression from live output: "Does <evidence>. confirm or
    refute: <hypothesis>" is not a real question -- it's evidence text and
    hypothesis text glued together. The generated question must be a
    standalone, grammatically complete sentence that never contains that
    concatenation shape."""
    hyps = [_hyp("H1", statement="The calibration label was not applied due to human oversight or error.")]
    hyps[0].evidence_needed = "Record of label application or audit trail for EQ-104."

    questions = derive_investigation_questions(hyps)

    assert len(questions) == 1
    q = questions[0].question
    assert q.endswith("?")
    assert "confirm or refute" not in q.lower()
    assert not q.startswith("Does " + hyps[0].evidence_needed)


# ---------------------------------------------------------------------------
# validate_investigation_question
# ---------------------------------------------------------------------------


def test_validate_investigation_question_rejects_concatenation_pattern():
    assert not validate_investigation_question(
        "Does Record of label application or audit trail for EQ-104. confirm or refute: "
        "The calibration label was not applied due to human oversight or error."
    )


def test_validate_investigation_question_rejects_non_question():
    assert not validate_investigation_question("Review the relevant records.")
    assert not validate_investigation_question(None)
    assert not validate_investigation_question("Why?")


def test_validate_investigation_question_accepts_well_formed_question():
    assert validate_investigation_question(
        "Was the required equipment-status label applied after the calibration event?"
    )
