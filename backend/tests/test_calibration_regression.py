"""Case A (calibration) regression test, required by the production-
hardening pass: a genuinely new domain (expiry-then-use temporal pattern)
this codebase had never been exercised against before. Verifies the
General/Unresolved deterministic branch no longer invents generic-bucket
hypotheses for an evidence-thin finding, that investigation content
survives to the final report, and that the expiry/use temporal
relationship produces a distinct, evidence-bounded impact narrative
instead of the generic "record missing" boilerplate.
"""

from __future__ import annotations

import pytest

from app.models.agent import CapaAnalysis, CapaStatus, CanonicalFindingState, EvidenceItem, EvidenceStatus

CALIBRATION_FINDING = (
    "The calibration certificate for balance BAL-014 showed an expiry date of "
    "10 August 2026. The balance was used for three production measurements on "
    "12 August 2026."
)
CALIBRATION_LEDGER = [
    EvidenceItem(
        claim="The calibration certificate for balance BAL-014 showed an expiry date of 10 August 2026.",
        source="finding_text", status=EvidenceStatus.VERIFIED,
    ),
    EvidenceItem(
        claim="The balance was used for three production measurements on 12 August 2026.",
        source="finding_text", status=EvidenceStatus.VERIFIED,
    ),
]


def test_deterministic_plan_generates_zero_invented_hypotheses():
    """No reported explanation, no conflict -- the deterministic generator
    must not invent a "compliance gap"/"control gap" hypothesis pair just
    to fill the slot (Section 21: zero hypotheses is a valid output)."""
    from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
    hyps, plan = build_deterministic_investigation_plan(CALIBRATION_FINDING, CALIBRATION_LEDGER)
    assert hyps == []
    assert "PROCESS_EXECUTION_COMPLIANCE_GAP" not in [h.name for h in hyps]
    assert "VERIFICATION_OR_RECONCILIATION_CONTROL_GAP" not in [h.name for h in hyps]


def test_deterministic_plan_generates_foundational_investigation_questions():
    """Zero hypotheses must not mean zero investigation (Section 10)."""
    from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
    _, plan = build_deterministic_investigation_plan(CALIBRATION_FINDING, CALIBRATION_LEDGER)
    assert len(plan.questions) >= 3
    joined = " ".join(q.question.lower() for q in plan.questions)
    # Questions investigate, they do not presuppose a specific failure mode.
    assert "did not" not in joined
    assert "failed" not in joined


def test_expiry_then_use_produces_distinct_impact_narrative():
    """The impact narrative for "used after stated expiry" must not be the
    generic "record missing" boilerplate (Section 18) -- and must not
    assert the measurement was invalid or that a product was affected."""
    from app.agent.nodes.core_synthesis import _detect_expiry_then_use, _derive_deterministic_impact
    assert _detect_expiry_then_use(CALIBRATION_FINDING) is True
    impact, _, _, _ = _derive_deterministic_impact(CALIBRATION_FINDING, None, "balance used after certificate expiry")
    assert "may require assessment" in impact.potential_effect
    assert "were invalid" not in impact.potential_effect.lower()
    assert "products are affected" not in impact.potential_effect.lower()
    assert "must be rejected" not in impact.potential_effect.lower()


def test_expiry_then_use_not_detected_without_both_dates():
    from app.agent.nodes.core_synthesis import _detect_expiry_then_use
    assert _detect_expiry_then_use("The balance was used for three measurements.") is False
    assert _detect_expiry_then_use("The certificate expired on 10 August 2026.") is False


@pytest.mark.asyncio
async def test_end_to_end_calibration_finding_not_established_with_real_investigation_content():
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
    mock_state = {
        "request": type("Request", (), {"finding_text": CALIBRATION_FINDING})(),
        "evidence_ledger": CALIBRATION_LEDGER,
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
    result = await final_evidence_verification_node(mock_state)
    rc = result["root_cause"]
    inv = result["investigation_plan"]

    assert str(rc.status) in ("NOT_ESTABLISHED", "RootCauseStatus.NOT_ESTABLISHED")
    assert rc.candidate_hypotheses == []
    # Never invented mechanisms for this finding.
    for forbidden in ("renewal", "reminder", "not renewed", "schedule was missed"):
        assert forbidden not in (rc.narrative or "").lower()
    # Investigation content must survive to the final report (Section 10).
    assert inv.questions
    assert inv.areas
    # Investigation areas must not read as an established cause (Section 11).
    for area in inv.areas:
        assert "compliance gap" not in area.lower()
        assert "control gap" not in area.lower()


def test_case_c_missing_records_treated_as_separate_from_noncompletion():
    """Case C: missing records must not be used to prove non-completion --
    both the VERIFIED noncompletion observation and the record-unavailable
    fact stay independent propositions."""
    from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
    finding = (
        "The auditor observed that the cleaning checklist was incomplete. "
        "The relevant checklist records were not available during the audit."
    )
    ledger = [
        EvidenceItem(claim="the cleaning checklist was incomplete", source="finding_text", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="the relevant checklist records were not available during the audit", source="finding_text", status=EvidenceStatus.VERIFIED),
    ]
    hyps, plan = build_deterministic_investigation_plan(finding, ledger)
    for h in hyps:
        assert "was not performed" not in h.statement.lower()
        assert "did not occur" not in h.statement.lower()
    assert plan.questions


def test_case_d_reported_workload_stays_reported_not_root_cause():
    """Case D: a reported workload explanation must never be silently
    promoted into a root cause or an unhedged causal hypothesis."""
    from app.agent.causal_model import claims_from_evidence_ledger, ClaimType
    finding = (
        "The operator did not record the equipment temperature. The operator "
        "stated that they forgot because the department was very busy that morning."
    )
    ledger = [
        EvidenceItem(claim="The operator did not record the equipment temperature.", source="finding_text", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="The operator stated that they forgot because the department was very busy that morning.", source="finding_text", status=EvidenceStatus.REPORTED),
    ]
    claims = claims_from_evidence_ledger(ledger)
    assert claims[0].claim_type == ClaimType.OBSERVED_FACT
    # Contains a causal connector ("because") -- correctly classified as a
    # REPORTED mechanism, never silently upgraded to VERIFIED.
    assert claims[1].claim_type == ClaimType.REPORTED_CAUSAL_MECHANISM
    from app.agent.causal_model import Provenance
    assert claims[1].provenance == Provenance.REPORTED


def test_case_e_subjective_supervisor_judgment_is_not_root_cause():
    """Case E: a supervisor's subjective attribution ("careless",
    "frequently ignored procedures") must never become a hypothesis or
    root cause -- it is evidence about an OPINION, not an objective
    mechanism."""
    from app.agent.causal_guard import hypothesis_attacks_statement_credibility
    from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
    finding = (
        "The required inspection was not completed. The supervisor stated that the "
        "employee was careless and frequently ignored procedures."
    )
    ledger = [
        EvidenceItem(claim="The required inspection was not completed.", source="finding_text", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="The supervisor stated that the employee was careless and frequently ignored procedures.", source="finding_text", status=EvidenceStatus.REPORTED),
    ]
    hyps, _ = build_deterministic_investigation_plan(finding, ledger)
    for h in hyps:
        assert "careless" not in h.statement.lower()
        assert "negligen" not in h.statement.lower()
        assert "irresponsible" not in h.statement.lower()


# ---------------------------------------------------------------------------
# Live-run-discovered defects (2/3 real LLM runs against this exact finding):
# a 5-Why answer fabricated "the certificate was not renewed" -- a distinct,
# stronger upstream-process claim the finding never establishes -- and
# mislabeled it MIXED despite no actual evidence conflict for this finding.
# ---------------------------------------------------------------------------

def test_renewal_mechanism_rejected_regardless_of_evidence_direction():
    from app.agent.causal_guard import detect_unsupported_causal_specificity
    for statement in [
        "The calibration certificate was not renewed before the expiry date.",
        "Recertification of BAL-014 did not occur before use.",
        "The balance was not recalibrated after the certificate expired.",
    ]:
        is_unsupported, _ = detect_unsupported_causal_specificity(statement, CALIBRATION_FINDING)
        assert is_unsupported, f"Expected rejection for: {statement!r}"


def test_renewal_mechanism_licensed_when_finding_independently_describes_it():
    from app.agent.causal_guard import detect_unsupported_causal_specificity
    finding = (
        "The calibration certificate for balance BAL-014 expired on 10 August 2026. "
        "The renewal request submitted on 8 August 2026 was not processed before the balance "
        "was used on 12 August 2026."
    )
    statement = "The calibration renewal request was not processed before the balance was used."
    is_unsupported, _ = detect_unsupported_causal_specificity(statement, finding)
    assert not is_unsupported


@pytest.mark.asyncio
async def test_five_why_mixed_status_downgraded_without_actual_conflict():
    """MIXED requires an actual detected conflict (Section 6/8) -- a step
    labeled MIXED for a finding with zero evidence_conflicts must be
    downgraded to UNKNOWN, never left as an unearned-sounding status."""
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
    from app.models.agent import FiveWhyAnalysis, FiveWhyStep

    mock_state = {
        "request": type("Request", (), {"finding_text": CALIBRATION_FINDING})(),
        "evidence_ledger": CALIBRATION_LEDGER,
        "canonical_finding_state": CanonicalFindingState(
            raw_finding=CALIBRATION_FINDING, observed_deviation="balance used after certificate expiry",
            evidence_conflicts=[],
        ),
        "root_cause": type("RC", (), {
            "narrative": None, "statement": None, "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED", "candidate_hypotheses": [],
        })(),
        "five_why": FiveWhyAnalysis(steps=[
            FiveWhyStep(
                question="Why was the balance used after its calibration certificate expired?",
                answer="Because the calibration certificate expiry date was 10 August 2026, and the "
                "balance was used on 12 August 2026.",
                status="MIXED",
            ),
        ]),
        "investigation_plan": type("Inv", (), {"questions": [], "areas": [], "evidence_to_collect": []})(),
        "capa_analysis": CapaAnalysis(status=CapaStatus.INVESTIGATION_REQUIRED, conditional_actions=[]),
        "ca_draft": None,
        "trace": [],
        "errors": [],
    }
    result = await final_evidence_verification_node(mock_state)
    fw = result["five_why"]
    assert fw.steps
    assert fw.steps[0].status != "MIXED"


@pytest.mark.asyncio
async def test_five_why_renewal_fabrication_replaced_with_evidence_boundary():
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
    from app.models.agent import FiveWhyAnalysis, FiveWhyStep

    mock_state = {
        "request": type("Request", (), {"finding_text": CALIBRATION_FINDING})(),
        "evidence_ledger": CALIBRATION_LEDGER,
        "canonical_finding_state": CanonicalFindingState(
            raw_finding=CALIBRATION_FINDING, observed_deviation="balance used after certificate expiry",
            evidence_conflicts=[],
        ),
        "root_cause": type("RC", (), {
            "narrative": None, "statement": None, "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED", "candidate_hypotheses": [],
        })(),
        "five_why": FiveWhyAnalysis(steps=[
            FiveWhyStep(
                question="Why was the balance used after its calibration certificate expired?",
                answer="Because the calibration certificate was not renewed before the expiry date.",
                status="MIXED",
            ),
        ]),
        "investigation_plan": type("Inv", (), {"questions": [], "areas": [], "evidence_to_collect": []})(),
        "capa_analysis": CapaAnalysis(status=CapaStatus.INVESTIGATION_REQUIRED, conditional_actions=[]),
        "ca_draft": None,
        "trace": [],
        "errors": [],
    }
    result = await final_evidence_verification_node(mock_state)
    fw = result["five_why"]
    assert fw.steps
    assert "renewed" not in fw.steps[0].answer.lower()
    assert fw.steps[0].status == "UNKNOWN"


# ---------------------------------------------------------------------------
# Affected-object/period extraction (Known Failures 6/8): structural fix in
# semantic_subject.py's entity-noun resolution, not a per-domain keyword list.
# ---------------------------------------------------------------------------

def test_affected_object_is_entity_pure_not_wrapped_in_generic_status_label():
    from app.services.semantic_subject import extract_semantic_subject
    r = extract_semantic_subject(CALIBRATION_FINDING)
    assert r.subject == "balance BAL-014"
    assert "equipment calibration status" not in r.subject.lower()


@pytest.mark.parametrize("finding,expected_object", [
    ("The temperature log for refrigerator QC-REF-02 was not completed for 12 August 2026.",
     "temperature log for refrigerator QC-REF-02"),
    ("The pipette P-200 failed visual inspection due to a cracked tip ejector.", "pipette P-200"),
])
def test_affected_object_extraction_generalizes_across_domains(finding, expected_object):
    from app.services.semantic_subject import extract_semantic_subject
    r = extract_semantic_subject(finding)
    assert r.subject == expected_object


def test_affected_period_is_use_date_not_unknown_for_expiry_then_use():
    """Known Failure 8: affected period must be the deviation/exposure
    period (the use date), never left UNKNOWN when the finding states it,
    and never the audit/discovery date."""
    from app.agent.nodes.core_synthesis import _derive_deterministic_impact
    impact, _, _, _ = _derive_deterministic_impact(CALIBRATION_FINDING, None, "balance used after certificate expiry")
    assert impact.affected_period == "12 August 2026"
    assert impact.affected_period != "UNKNOWN"


@pytest.mark.asyncio
async def test_end_to_end_calibration_affected_object_and_period():
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
    mock_state = {
        "request": type("Request", (), {"finding_text": CALIBRATION_FINDING})(),
        "evidence_ledger": CALIBRATION_LEDGER,
        "canonical_finding_state": CanonicalFindingState(
            raw_finding=CALIBRATION_FINDING, observed_deviation="balance used after certificate expiry",
        ),
        "root_cause": type("RC", (), {
            "narrative": None, "statement": None, "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED", "candidate_hypotheses": [],
        })(),
        "investigation_plan": type("Inv", (), {"questions": [], "areas": [], "evidence_to_collect": []})(),
        "capa_analysis": CapaAnalysis(status=CapaStatus.INVESTIGATION_REQUIRED, conditional_actions=[]),
        "impact_assessment": None,
        "ca_draft": None,
        "trace": [],
        "errors": [],
    }
    result = await final_evidence_verification_node(mock_state)
    impact = result.get("impact_assessment")
    if impact is not None:
        assert impact.affected_object == "Balance BAL-014"


# ---------------------------------------------------------------------------
# Semantic field ownership (single authoritative producer): the real defect
# was not any one function being wrong in isolation -- resolve_deviation()
# was already correct -- it was that THREE independent code paths could each
# propose a competing affected_object/process_at_risk string, and the LAST
# one to run (a "drift correction" that used a fundamentally different,
# entity-biased template) always won. These tests target each producer
# directly so a regression in any one of them is caught even if the others
# happen to compensate for it in a given finding.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_understand_finding_node_prefers_deterministic_subject_over_llm_guess():
    """understand_finding_node must not let extraction.deviation_subject
    (the LLM's own guess) override a deterministic resolver result that
    actually succeeded, merely because the LLM's guess shares vocabulary
    with the finding ("grounded") -- groundedness is not correctness."""
    from unittest.mock import AsyncMock, patch
    from app.agent.nodes.understanding import understand_finding_node
    from app.models.agent import InvestigateRequest

    state = {
        "request": InvestigateRequest(finding_text=CALIBRATION_FINDING),
        "trace": [], "errors": [], "iteration_count": 0,
    }
    # Simulate the LLM extraction proposing a garbled subject that
    # nonetheless shares vocabulary with the finding (would pass a naive
    # "groundedness" check).
    with patch("app.agent.nodes.understanding.get_llm_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = (
            '{"stated_facts": ["The calibration certificate for balance BAL-014 showed an '
            'expiry date of 10 August 2026.", "The balance was used for three production '
            'measurements on 12 August 2026."], "attributed_statements": [], '
            '"deviation_subject": "personnel calibration status for calibration certificate", '
            '"deviation_condition": "unconfirmed", "deviation_actor": null, "timeframe": null}'
        )
        mock_get.return_value = mock_client
        result = await understand_finding_node(state)

    canonical = result["canonical_finding_state"]
    assert canonical.finding_subject == "balance BAL-014"
    assert "personnel" not in canonical.finding_subject.lower()


def test_derive_deterministic_impact_consumes_canonical_subject_not_reresolved():
    """_derive_deterministic_impact must use canonical.finding_subject when
    available, never independently re-run resolve_deviation() and risk a
    different answer than the one already established upstream."""
    from app.agent.nodes.core_synthesis import _derive_deterministic_impact
    canonical = CanonicalFindingState(
        raw_finding=CALIBRATION_FINDING,
        observed_deviation="balance used after certificate expiry",
        finding_subject="balance BAL-014",
    )
    impact, clean_noun, _, _ = _derive_deterministic_impact(CALIBRATION_FINDING, canonical, "x")
    assert clean_noun == "balance BAL-014"
    assert impact.affected_object == "Balance BAL-014"


def test_build_affected_object_phrase_does_not_wrap_entity_shaped_subjects():
    """The 'actor + topic status for tail' template must only apply to
    activity/qualification-shaped subjects, never to already-clean entity
    subjects -- wrapping "balance BAL-014" in it produced "Personnel
    balance status for balance BAL-014"."""
    from app.services.semantic_subject import build_affected_object_phrase
    assert build_affected_object_phrase("balance BAL-014", None) == "Balance BAL-014"
    assert build_affected_object_phrase("temperature log for refrigerator QC-REF-02", None) == (
        "Temperature log for refrigerator QC-REF-02"
    )
    assert build_affected_object_phrase("training for the revised procedure", "the operator") == (
        "Operator training status for the revised procedure"
    )


@pytest.mark.asyncio
async def test_process_at_risk_drift_correction_does_not_reject_legitimate_domain_vocabulary():
    """process_at_risk names the CONTROL DOMAIN (e.g. "calibration"), which
    legitimately differs from the entity/affected-object vocabulary (e.g.
    "balance BAL-014") by design -- the drift-correction guard must not
    treat every entity/process vocabulary mismatch as drift, only genuinely
    unrelated substitutions."""
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
    from app.models.agent import ImpactAssessment, ImpactStatus

    mock_state = {
        "request": type("Request", (), {"finding_text": CALIBRATION_FINDING})(),
        "evidence_ledger": CALIBRATION_LEDGER,
        "canonical_finding_state": CanonicalFindingState(
            raw_finding=CALIBRATION_FINDING, observed_deviation="balance used after certificate expiry",
            finding_subject="balance BAL-014",
        ),
        "root_cause": type("RC", (), {
            "narrative": None, "statement": None, "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED", "candidate_hypotheses": [],
        })(),
        "investigation_plan": type("Inv", (), {"questions": [], "areas": [], "evidence_to_collect": []})(),
        "capa_analysis": CapaAnalysis(status=CapaStatus.INVESTIGATION_REQUIRED, conditional_actions=[]),
        "impact_assessment": ImpactAssessment(
            status=ImpactStatus.IMPACT_REQUIRES_ASSESSMENT,
            affected_object="Balance BAL-014",
            process_at_risk="Calibration and equipment-use control",
        ),
        "ca_draft": None,
        "trace": [],
        "errors": [],
    }
    result = await final_evidence_verification_node(mock_state)
    impact = result.get("impact_assessment")
    if impact is not None:
        assert impact.process_at_risk == "Calibration and equipment-use control"


def test_process_at_risk_domain_word_derived_from_deviation_not_entity():
    """Known Failure 7: process_at_risk must come from the deviation/
    control domain, not the entity resolver's own object noun."""
    from app.agent.nodes.core_synthesis import _extract_expiry_domain_word
    assert _extract_expiry_domain_word(CALIBRATION_FINDING) == "calibration"
