"""Tests for the structured causal-proposition model (app.agent.causal_model)
and its wiring into final_evidence_verification_node as the primary
hypothesis-eligibility enforcement layer.

Covers: Claim classification/provenance, computed (not LLM-declared)
support_level, the downstream-state-cannot-support-upstream-mechanism
invariant, hypothesis eligibility (Section 5 cases), root-cause status
derivation, investigation-area-without-hypothesis (Section 4/6), and
paraphrase-resistance (Section 32) -- the same recurring finding rejected
under five different vocabulary families without any new phrase list.
"""

import pytest

from app.agent.causal_model import (
    ClaimType,
    HypothesisEligibility,
    Provenance,
    SupportLevel,
    build_causal_proposition,
    claims_from_evidence_ledger,
    compute_support_level,
    derive_hypothesis_eligibility,
    derive_root_cause_status,
)
from app.models.agent import CandidateHypothesis, EvidenceItem, EvidenceStatus
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node


REVISION_FINDING = (
    "The daily equipment inspection checklist was not completed for three consecutive days. "
    "The operator stated that they were unaware that the checklist procedure had been revised."
)
REVISION_LEDGER = [
    EvidenceItem(
        claim="The daily equipment inspection checklist was not completed for three consecutive days.",
        source="finding_text", status=EvidenceStatus.VERIFIED,
    ),
    EvidenceItem(
        claim="The operator stated that they were unaware that the checklist procedure had been revised.",
        source="finding_text", status=EvidenceStatus.REPORTED,
    ),
]


# ---------------------------------------------------------------------------
# Claim classification / provenance
# ---------------------------------------------------------------------------

def test_verified_ledger_item_becomes_observed_fact_with_verified_provenance():
    claims = claims_from_evidence_ledger(REVISION_LEDGER)
    assert claims[0].claim_type == ClaimType.OBSERVED_FACT
    assert claims[0].provenance == Provenance.VERIFIED


def test_bare_reported_state_is_not_classified_as_causal_mechanism():
    """'operator was unaware' contains no causal connector ('because'/'due
    to'/etc.) -- it is a REPORTED_STATE, not a REPORTED_CAUSAL_MECHANISM.
    This distinction is the entire basis for why it cannot support a
    communication-failure proposition."""
    claims = claims_from_evidence_ledger(REVISION_LEDGER)
    assert claims[1].claim_type == ClaimType.REPORTED_STATE
    assert claims[1].provenance == Provenance.REPORTED


def test_reported_claim_with_causal_connector_is_a_reported_causal_mechanism():
    ledger = [EvidenceItem(
        claim="The technician stated the check was missed because they had not received retraining.",
        source="finding_text", status=EvidenceStatus.REPORTED,
    )]
    claims = claims_from_evidence_ledger(ledger)
    assert claims[0].claim_type == ClaimType.REPORTED_CAUSAL_MECHANISM


@pytest.mark.parametrize("status", [EvidenceStatus.INFERRED, EvidenceStatus.UNVERIFIED])
def test_inferred_or_unverified_claim_never_becomes_verified_provenance(status):
    """Non-negotiable invariant: an INFERRED claim must never silently
    become VERIFIED."""
    ledger = [EvidenceItem(claim="x", source="finding_text", status=status)]
    claims = claims_from_evidence_ledger(ledger)
    assert claims[0].provenance != Provenance.VERIFIED


# ---------------------------------------------------------------------------
# Downstream state cannot support upstream mechanism (Section 8) --
# formalized as a computed support_level, not a phrase list. Same recurring
# finding, five unrelated vocabulary families, zero new words added.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("statement", [
    "The revision was not communicated to the operator.",
    "The revision was not shared with the operator.",
    "The operator was not informed of the change.",
    "The change was not disseminated to staff.",
    "Staff were not notified of the revision.",
    "The checklist revision was not properly documented or accessible to the operator.",
])
def test_downstream_awareness_report_cannot_support_upstream_mechanism_any_vocabulary(statement):
    claims = claims_from_evidence_ledger(REVISION_LEDGER)
    from app.services.text_grounding import significant_words
    subject_words = significant_words("daily equipment inspection checklist")
    level = compute_support_level(statement, claims, REVISION_FINDING, subject_words=subject_words)
    assert level in (SupportLevel.NONE, SupportLevel.INDIRECT)
    assert derive_hypothesis_eligibility(level) == HypothesisEligibility.NOT_ELIGIBLE


def test_hedged_mechanism_with_no_reported_causal_claim_is_still_not_eligible():
    """A hedge word makes a HYPOTHESIS statement legitimate prose (not an
    over-claim the guards reject) but hedging is not evidence -- if no
    claim actually supports the mechanism, it stays NOT_ELIGIBLE regardless
    of how carefully it's hedged (HEDGING != EVIDENCE)."""
    claims = claims_from_evidence_ledger(REVISION_LEDGER)
    from app.services.text_grounding import significant_words
    subject_words = significant_words("daily equipment inspection checklist")
    statement = (
        "The revision affecting daily equipment inspection checklist may not have been "
        "effectively communicated to or acknowledged by the affected personnel."
    )
    level = compute_support_level(statement, claims, REVISION_FINDING, subject_words=subject_words)
    assert derive_hypothesis_eligibility(level) == HypothesisEligibility.NOT_ELIGIBLE


# ---------------------------------------------------------------------------
# Positive controls (Section 27): evidence-eligible mechanisms must still
# be allowed. Over-correcting to always reject is itself a defect.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("finding,statement", [
    ("Distribution log shows the operator was omitted from the checklist distribution.",
     "The revision may not have been distributed to the operator."),
    ("The training attendance record shows the operator did not complete required retraining.",
     "The operator did not complete required retraining on the revised procedure."),
    ("The controlled distribution record shows the operator was omitted from the "
     "revised-procedure distribution list.",
     "The operator was omitted from the required revision distribution."),
])
def test_verified_record_directly_supporting_mechanism_is_eligible(finding, statement):
    ledger = [EvidenceItem(claim=finding, source="finding_text", status=EvidenceStatus.VERIFIED)]
    claims = claims_from_evidence_ledger(ledger)
    level = compute_support_level(statement, claims, finding)
    assert level == SupportLevel.VERIFIED_SUPPORT
    assert derive_hypothesis_eligibility(level) == HypothesisEligibility.ELIGIBLE


def test_reported_causal_mechanism_claim_is_eligible():
    finding = (
        "The technician stated the check was missed because they had not received "
        "retraining on the new procedure."
    )
    ledger = [EvidenceItem(claim=finding, source="finding_text", status=EvidenceStatus.REPORTED)]
    claims = claims_from_evidence_ledger(ledger)
    statement = "The technician had not received retraining on the new procedure."
    level = compute_support_level(statement, claims, finding)
    assert level == SupportLevel.REPORTED_SUPPORT
    assert derive_hypothesis_eligibility(level) == HypothesisEligibility.ELIGIBLE


# ---------------------------------------------------------------------------
# Similarity != causal support: sharing the finding's own subject nouns
# must not, by itself, promote a hypothesis to VERIFIED_SUPPORT.
# ---------------------------------------------------------------------------

def test_shared_subject_nouns_alone_do_not_constitute_support():
    from app.services.text_grounding import significant_words
    claims = claims_from_evidence_ledger(REVISION_LEDGER)
    subject_words = significant_words("daily equipment inspection checklist")
    # This statement shares only the finding's subject phrase with the
    # VERIFIED claim -- no distinctive mechanism word overlaps.
    statement = "A gap in the daily equipment inspection checklist process caused the omission."
    level = compute_support_level(statement, claims, REVISION_FINDING, subject_words=subject_words)
    assert level != SupportLevel.VERIFIED_SUPPORT


# ---------------------------------------------------------------------------
# Root cause status derivation (Section 19), mapped onto the existing enum.
# ---------------------------------------------------------------------------

def test_root_cause_status_not_established_when_no_eligible_proposition():
    from app.models.agent import RootCauseStatus
    assert derive_root_cause_status([]) == RootCauseStatus.NOT_ESTABLISHED


def test_root_cause_status_supported_when_a_verified_proposition_exists():
    from app.models.agent import RootCauseStatus
    ledger = [EvidenceItem(
        claim="Distribution log shows the operator was omitted from the checklist distribution.",
        source="finding_text", status=EvidenceStatus.VERIFIED,
    )]
    claims = claims_from_evidence_ledger(ledger)
    h = CandidateHypothesis(
        id="H1", name="X",
        statement="The revision may not have been distributed to the operator.",
        evidence_needed="x",
    )
    prop = build_causal_proposition(h, claims, ledger[0].claim)
    assert derive_root_cause_status([prop]) == RootCauseStatus.SUPPORTED


# ---------------------------------------------------------------------------
# End-to-end: final_evidence_verification_node demotes ineligible
# hypotheses to investigation areas rather than dropping them silently,
# and forces NOT_ESTABLISHED when nothing remains eligible (Section 26
# acceptance test).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_acceptance_finding_produces_no_candidate_hypotheses_and_investigation_areas():
    mock_state = {
        "request": type("Request", (), {"finding_text": REVISION_FINDING})(),
        "evidence_ledger": REVISION_LEDGER,
        "root_cause": type("RC", (), {
            "narrative": "The deviation may stem from a communication gap regarding the checklist revision.",
            "statement": None,
            "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED",
            "candidate_hypotheses": [
                CandidateHypothesis(
                    id="H1", name="Procedure Change Oversight",
                    statement="The checklist revision was not properly documented or shared with the operator.",
                    status="POSSIBLE", evidence_needed="documented checklist revision process",
                ),
            ],
        })(),
        "investigation_plan": type("Inv", (), {"questions": [], "areas": []})(),
        "ca_draft": None,
        "trace": [],
        "errors": [],
    }
    res = await final_evidence_verification_node(mock_state)
    rc = res["root_cause"]
    assert rc.candidate_hypotheses == []
    assert rc.status in ("NOT_ESTABLISHED", None) or str(rc.status) == "RootCauseStatus.NOT_ESTABLISHED"


@pytest.mark.asyncio
async def test_eligible_hypothesis_survives_the_end_to_end_pipeline():
    finding = "Distribution log shows the operator was omitted from the checklist distribution."
    ledger = [EvidenceItem(claim=finding, source="finding_text", status=EvidenceStatus.VERIFIED)]
    mock_state = {
        "request": type("Request", (), {"finding_text": finding})(),
        "evidence_ledger": ledger,
        "root_cause": type("RC", (), {
            "narrative": None,
            "statement": None,
            "status": "STATED_UNVERIFIED",
            "category": "TO_BE_CONFIRMED",
            "candidate_hypotheses": [
                CandidateHypothesis(
                    id="H1", name="DISTRIBUTION_OMISSION",
                    statement="The revision may not have been distributed to the operator.",
                    status="POSSIBLE", evidence_needed="distribution record",
                ),
            ],
        })(),
        "investigation_plan": type("Inv", (), {"questions": [], "areas": []})(),
        "ca_draft": None,
        "trace": [],
        "errors": [],
    }
    res = await final_evidence_verification_node(mock_state)
    rc = res["root_cause"]
    assert len(rc.candidate_hypotheses) == 1
    assert rc.candidate_hypotheses[0].id == "H1"


# ---------------------------------------------------------------------------
# Training-conflict finding (this turn's regression target): a REPORTED
# claim that DIRECTLY reports the same proposition a hypothesis asserts
# (not merely a downstream state that could explain it) must be able to
# reach REPORTED_SUPPORT even when the finding's own extracted subject
# happens to name the mechanism's own topic word (e.g. "training for the
# revised procedure") -- excluding subject words from REPORTED-claim
# matching would make the hypothesis's own core mechanism unmatchable
# against a direct report of that same mechanism.
# ---------------------------------------------------------------------------

TRAINING_CONFLICT_FINDING = (
    "The operator stated that they had not received training on the revised procedure. "
    "The department supervisor stated that the operator completed the required training "
    "before the procedure became effective. "
    "No training attendance record was available during the audit."
)
TRAINING_CONFLICT_LEDGER = [
    EvidenceItem(
        claim="The operator stated that they had not received training on the revised procedure.",
        source="finding_text", status=EvidenceStatus.REPORTED,
    ),
    EvidenceItem(
        claim="The department supervisor stated that the operator completed the required "
        "training before the procedure became effective.",
        source="finding_text", status=EvidenceStatus.REPORTED,
    ),
    EvidenceItem(
        claim="No training attendance record was available during the audit.",
        source="finding_text", status=EvidenceStatus.VERIFIED,
    ),
]


def test_direct_reported_statement_of_hypothesis_proposition_is_eligible():
    from app.services.text_grounding import significant_words
    claims = claims_from_evidence_ledger(TRAINING_CONFLICT_LEDGER)
    subject_words = significant_words("training for the revised procedure")
    statement = (
        "Required training for the revised procedure may not have been completed before "
        "the procedure became effective."
    )
    level = compute_support_level(
        statement, claims, TRAINING_CONFLICT_FINDING, subject_words=subject_words
    )
    assert level == SupportLevel.REPORTED_SUPPORT
    assert derive_hypothesis_eligibility(level) == HypothesisEligibility.ELIGIBLE


@pytest.mark.asyncio
async def test_training_conflict_end_to_end_no_duplicate_capa_no_overclaim():
    from app.models.agent import CapaAnalysis, CapaStatus
    mock_state = {
        "request": type("Request", (), {"finding_text": TRAINING_CONFLICT_FINDING})(),
        "evidence_ledger": TRAINING_CONFLICT_LEDGER,
        "root_cause": type("RC", (), {
            "narrative": None, "statement": None, "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED", "candidate_hypotheses": [],
        })(),
        "investigation_plan": type("Inv", (), {"questions": [], "areas": [], "evidence_to_collect": []})(),
        "capa_analysis": CapaAnalysis(status=CapaStatus.INVESTIGATION_REQUIRED, conditional_actions=[]),
        "ca_draft": None,
        "trace": [],
        "errors": [],
    }
    res = await final_evidence_verification_node(mock_state)
    rc = res["root_cause"]
    capa = res["capa_analysis"]

    # Causal-level separation: record-availability ("record unavailable at
    # audit time") and record-control-process propositions are a different
    # causal level from the execution-level "was training completed?"
    # question -- they must never appear in candidate_hypotheses as
    # competing root causes. Only ONE causal hypothesis survives.
    hyp_by_name = {h.name: h for h in rc.candidate_hypotheses}
    assert "TRAINING_RECORD_UNAVAILABLE" not in hyp_by_name
    assert "TRAINING_RECORD_CONTROL_GAP" not in hyp_by_name
    assert "TRAINING_NOT_COMPLETED" in hyp_by_name
    assert len(rc.candidate_hypotheses) == 1

    h1 = hyp_by_name["TRAINING_NOT_COMPLETED"]
    assert h1.supporting_evidence
    assert h1.contradicting_evidence

    # The record-availability/record-control content still exists as
    # investigation questions, just never as competing hypotheses.
    inv = res["investigation_plan"]
    question_texts = " ".join(q.question for q in inv.questions).lower()
    assert "record" in question_texts and ("located" in question_texts or "retained" in question_texts)
    # Questions resolving evidence-state/systemic content are not falsely
    # bound to a hypothesis ID (they don't test a candidate hypothesis).
    for q in inv.questions:
        if q.hypothesis_tested:
            assert q.hypothesis_tested in {h.id for h in rc.candidate_hypotheses}

    # No two conditional actions targeting the same hypothesis are
    # near-identical restatements of each other.
    from app.services.text_grounding import significant_words
    by_condition: dict[str, list] = {}
    for a in capa.conditional_actions:
        by_condition.setdefault(a.if_cause_confirmed, []).append(a.recommended_action)
    for condition, texts in by_condition.items():
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                w1, w2 = significant_words(texts[i]), significant_words(texts[j])
                overlap = len(w1 & w2) / max(len(w1 | w2), 1)
                assert overlap < 0.55, f"Near-duplicate CAPA actions under {condition!r}: {texts[i]!r} / {texts[j]!r}"

    # No CAPA action leaked garbled/unresolved subject text (the finding's
    # own raw sentences dumped into the template because subject
    # extraction fell back to the whole finding text).
    for a in capa.conditional_actions:
        assert "operator stated" not in a.recommended_action.lower()

    assert rc.status in ("NOT_ESTABLISHED", None) or str(rc.status) == "RootCauseStatus.NOT_ESTABLISHED"
    assert rc.narrative  # Why field must never be left blank when NOT_ESTABLISHED


def test_potential_effect_invented_qualification_concept_is_rejected():
    from app.agent.causal_guard import impact_asserts_unestablished_concept
    text = (
        "If required training was not completed before the applicable effective date, the "
        "operator may have proceeded without confirmed qualification for the revised procedure."
    )
    assert impact_asserts_unestablished_concept(text, TRAINING_CONFLICT_FINDING, [])


def test_potential_effect_concept_licensed_when_evidence_establishes_it():
    from app.agent.causal_guard import impact_asserts_unestablished_concept
    finding = "The operator's qualification for the revised procedure was not verified before use."
    text = "The operator may have proceeded without confirmed qualification for the revised procedure."
    assert not impact_asserts_unestablished_concept(text, finding, [])


def test_record_retrieval_failure_overclaim_rejected_regardless_of_verb():
    from app.agent.causal_guard import detect_unsupported_causal_specificity
    finding = "No training attendance record was available during the audit."
    for statement in [
        "The required training record could not be retrieved during the audit.",
        "The required training record could not be located during the audit.",
        "The training record retrieval was unsuccessful during the audit.",
    ]:
        is_unsupported, _ = detect_unsupported_causal_specificity(statement, finding)
        assert is_unsupported, f"Expected rejection for: {statement!r}"


# ---------------------------------------------------------------------------
# Causal-level classification (Phase 1/7): a proposition's LEVEL is
# independent of its mechanism_type/support_level -- an evidence-eligible
# systemic claim must never be scored as a peer of an evidence-eligible
# execution-level claim.
# ---------------------------------------------------------------------------

def test_execution_level_statement_is_contributing_cause():
    from app.agent.causal_model import CausalLevel, derive_causal_level
    statement = "Required training for the revised procedure may not have been completed before the procedure became effective."
    assert derive_causal_level(statement) == CausalLevel.CONTRIBUTING_CAUSE


def test_systemic_statement_is_systemic_cause_level():
    from app.agent.causal_model import CausalLevel, derive_causal_level
    statement = "The training-record management process lacked adequate control."
    assert derive_causal_level(statement) == CausalLevel.SYSTEMIC_CAUSE


def test_causal_proposition_carries_its_causal_level():
    finding = "Distribution log shows the operator was omitted from the checklist distribution."
    ledger = [EvidenceItem(claim=finding, source="finding_text", status=EvidenceStatus.VERIFIED)]
    claims = claims_from_evidence_ledger(ledger)
    h = CandidateHypothesis(
        id="H1", name="X",
        statement="The revision may not have been distributed to the operator.",
        evidence_needed="x",
    )
    prop = build_causal_proposition(h, claims, finding)
    assert prop.causal_level is not None


# ---------------------------------------------------------------------------
# Property-based invariants (Phase 22) -- general statements that must hold
# across ANY input shape, not just the specific findings tested above.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hypotheses,expected_status", [
    ([], "NONE"),
    (
        [CandidateHypothesis(id="H1", name="X", statement="a", status="POSSIBLE", evidence_needed="x")],
        "TIED",
    ),
    (
        [
            CandidateHypothesis(id="H1", name="X", statement="a", status="POSSIBLE", evidence_needed="x"),
            CandidateHypothesis(id="H2", name="Y", statement="b", status="POSSIBLE", evidence_needed="x"),
        ],
        "TIED",
    ),
])
def test_invariant_leading_hypothesis_none_or_tied_never_silently_selected(hypotheses, expected_status):
    """IF leading_hypothesis == NONE (or the pool is tied) THEN no candidate
    may be rendered as leading -- a lone POSSIBLE candidate with no
    materially stronger evidence than (nonexistent or equal) alternatives
    must never be auto-promoted to SELECTED merely for being the only or
    first one in the list."""
    from app.agent.analytical_validator import leading_hypothesis_status, select_leading_hypothesis
    assert leading_hypothesis_status(hypotheses) == expected_status
    if expected_status != "SELECTED":
        assert select_leading_hypothesis(hypotheses) is None


def test_invariant_reported_claim_type_never_auto_promotes_to_verified():
    """IF claim.status == REPORTED THEN it cannot become VERIFIED without
    objective evidence -- classify_claim must preserve REPORTED provenance
    regardless of how confidently/specifically the claim reads."""
    ledger = [EvidenceItem(
        claim="The supervisor confirmed with absolute certainty that training was completed.",
        source="finding_text", status=EvidenceStatus.REPORTED,
    )]
    claims = claims_from_evidence_ledger(ledger)
    assert claims[0].provenance == Provenance.REPORTED


def test_invariant_record_unavailable_does_not_support_unrelated_activity_absence_claim():
    """IF evidence only states a record was unavailable THEN no proposition
    asserting the activity was not performed may reach VERIFIED_SUPPORT
    from that claim alone."""
    finding = "No training attendance record was available during the audit."
    ledger = [EvidenceItem(claim=finding, source="finding_text", status=EvidenceStatus.VERIFIED)]
    claims = claims_from_evidence_ledger(ledger)
    statement = "The training was never performed."
    level = compute_support_level(statement, claims, finding)
    assert level != SupportLevel.VERIFIED_SUPPORT


def test_invariant_unresolved_hypothesis_capa_actions_are_conditional():
    """IF hypothesis.status == UNRESOLVED/POSSIBLE THEN any generated CAPA
    action must be phrased as conditional ('IF ... confirmed'), never as an
    unconditional directive."""
    from app.agent.nodes.plan_investigation_fallback import build_conditional_capa_actions
    h = CandidateHypothesis(
        id="H1", name="TRAINING_NOT_COMPLETED", status="POSSIBLE",
        statement="Required training may not have been completed.", evidence_needed="x",
    )
    actions = build_conditional_capa_actions([h], "training", "training")
    assert actions
    for a in actions:
        assert a.if_cause_confirmed and "IF" in a.if_cause_confirmed.upper()
