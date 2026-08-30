"""EVIDENCE ARTIFACT != FINDING SUBJECT — structural-role hardening.

An evidence-request / evidence-needed noun phrase must never be promoted to
the canonical affected subject merely because it is grammatically noun-like.
The SAME vocabulary ("investigation record", "maintenance records",
"procedure", "risk assessment") must be:

  * ACCEPTED as the subject when the record/document/control itself is the
    deficient object of the finding, and
  * REJECTED when it is functioning as requested evidence, evidence needed to
    establish a cause, or an evidence-adequacy artifact.

All detection is structural (grammatical request frame) — no phrase
blacklist. Entities below span unrelated domains as TEST DATA only.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app.services.semantic_subject import (
    looks_like_evidence_request,
    reject_subject_if_clause,
    resolve_deviation,
    validate_semantic_subject,
)

# ---------------------------------------------------------------------------
# 1. The predicate itself — role, not vocabulary.
# ---------------------------------------------------------------------------

_IS_EVIDENCE_REQUEST = [
    "Evidence needed: investigation record name",
    "Evidence required: maintenance history for pump P-4",
    "Evidence needed to establish the cause: vibration analysis reports",
    "evidence needed to establish an assignable laboratory cause",
    "evidence required to determine the root cause",
    "records required to verify completion of the calibration",
    "documentation required to support the batch disposition",
    "data needed to improve the estimate",
    "sufficient evidence to support invalidation of the result",
    "adequate documentation to justify the deviation closure",
    "objective evidence to confirm the training was delivered",
    "a basis to establish the assignable cause",
    "What records establish that the review was performed?",
    "Which evidence confirms the segregation of duties?",
]

_IS_NOT_EVIDENCE_REQUEST = [
    "investigation record",
    "the investigation record",
    "maintenance record",
    "maintenance records for chiller CH-2",
    "calibration certificate CC-7",
    "the applicable procedure",
    "risk assessment RA-9",
    "batch record BR-4471",
    "data integrity controls for the LIMS",
    "records retention schedule RS-4",
    "documentation of the validation study",
    "the second-person verification step",
    "supplier qualification file for vendor V-12",
]


@pytest.mark.parametrize("phrase", _IS_EVIDENCE_REQUEST)
def test_evidence_request_phrases_are_detected(phrase):
    assert looks_like_evidence_request(phrase) is True
    assert reject_subject_if_clause(phrase) is True
    assert validate_semantic_subject(phrase) is False


@pytest.mark.parametrize("phrase", _IS_NOT_EVIDENCE_REQUEST)
def test_substantive_noun_phrases_are_not_evidence_requests(phrase):
    assert looks_like_evidence_request(phrase) is False
    # (not asserting validate==True for all: a few may fail other rules; the
    # point here is the evidence-request predicate does not fire)


# ---------------------------------------------------------------------------
# 2. Structural distinctions from the spec (A–F) at the resolver level.
# ---------------------------------------------------------------------------

def _subj(text):
    d = resolve_deviation(text)
    return d.finding_subject, getattr(d, "condition", None), d.semantic_type


def test_A_record_itself_is_deficient_is_a_valid_subject():
    s, cond, st = _subj(
        "The investigation record did not document sufficient evidence to "
        "support invalidation of the result."
    )
    assert s == "investigation record"
    assert st == "RECORD"
    assert "document" in (cond or "").lower()
    # the evidence-adequacy artifact never becomes the subject
    assert "sufficient evidence" not in (s or "").lower()


def test_E_bare_deficient_record_stays_valid():
    s, cond, st = _subj("The maintenance record was incomplete.")
    assert s == "maintenance record"
    assert (cond or "").lower() == "incomplete"


def test_B_standalone_evidence_needed_line_has_no_subject():
    s, _, st = _subj("Evidence needed: Investigation record name")
    assert s is None and st == "NON_ACTIONABLE"


def test_C_evidence_request_object_does_not_become_subject():
    s, _, _ = _subj(
        "Evidence needed to establish the cause: maintenance records for pump P-4"
    )
    # never the evidence artifact
    assert s is None or "maintenance record" not in s.lower()


def test_D_evidence_source_reporting_a_proposition_is_not_the_subject():
    s, cond, _ = _subj(
        "Maintenance records show that temporary repairs were performed on compressor K-2."
    )
    assert s is not None
    assert "maintenance record" not in s.lower()
    assert "compressor k-2" in s.lower() or "compressor" in s.lower()


def test_trailing_evidence_needed_list_does_not_capture_the_subject():
    s, _, _ = _subj(
        "The out-of-specification result for assay batch B-231 was invalidated on "
        "the basis of an assignable laboratory error. Evidence needed: investigation "
        "record, maintenance records, applicable procedure, status/history records."
    )
    assert s is not None
    low = s.lower()
    assert "b-231" in low or "assay batch" in low
    for artefact in ("investigation record", "maintenance record", "status/history",
                     "applicable procedure"):
        assert artefact not in low


def test_negated_reporting_frame_with_real_proposition_unchanged():
    # regression: the records ARE a frame here (real inner proposition, just a
    # generic inner subject) -> unresolved, NOT "complaint records".
    s, _, st = _subj(
        "Complaint records did not demonstrate that recurring failures were evaluated."
    )
    assert s is None and st == "NON_ACTIONABLE"


# ---------------------------------------------------------------------------
# 3. Downstream consumers preserve the canonical subject / fail closed.
# ---------------------------------------------------------------------------

async def _pipeline(finding_text: str):
    from app.agent.nodes.core_synthesis import core_synthesis_node
    from app.agent.nodes.investigation_planner import plan_investigation_node
    from app.agent.nodes.understanding import understand_finding_node
    from app.models.agent import InvestigateRequest

    state = {
        "request": InvestigateRequest(finding_text=finding_text),
        "evidence_ledger": [], "iteration_count": 0, "tool_call_count": 0,
        "critic_iteration": 0, "trace": [], "errors": [],
    }
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        state = await understand_finding_node(state)
        state = await plan_investigation_node(state)
        state = await core_synthesis_node(state)
    return state


_ARTEFACT_TOKENS = ("relay test record", "thermographic survey", "maintenance history record",
                    "evidence needed", "evidence required", "records required", "record name")


def test_downstream_never_sees_the_evidence_artifact():
    finding = (
        "Repeated trips of breaker BRK-7 recurred during the quarter. The "
        "investigation did not establish a definitive electrical mechanism. "
        "Evidence needed to establish the cause: relay test records, thermographic "
        "survey reports, maintenance history records."
    )
    state = asyncio.run(_pipeline(finding))
    canon = state.get("canonical_finding_state")
    assert canon is not None
    for field in ("finding_subject", "affected_object", "deviation_condition",
                  "affected_process", "missing_record_activity"):
        val = (getattr(canon, field, "") or "").lower()
        for artefact in _ARTEFACT_TOKENS:
            assert artefact not in val, f"{field!r} leaked {artefact!r}"

    # investigation questions + 5-Why must not contain an evidence artifact either
    plan = state.get("investigation_plan")
    qs = []
    if plan is not None:
        qs += [getattr(q, "question", "") for q in getattr(plan, "questions", [])]
    for q in state.get("five_why_questions", []) or []:
        qs.append(q if isinstance(q, str) else getattr(q, "question", ""))
    for q in qs:
        low = q.lower()
        for artefact in _ARTEFACT_TOKENS:
            assert artefact not in low, f"question leaked {artefact!r}: {q!r}"


def test_remediation_does_not_target_an_evidence_artifact():
    from app.models.agent import CanonicalFindingState, EvidenceItem, EvidenceStatus
    from app.remediation.engine import estimate_remediation_cost

    class _RC:
        status = "NOT_ESTABLISHED"
        candidate_hypotheses: list = []

    class _Down:
        async def chat_completion(self, *a, **k):
            raise RuntimeError("unavailable")

    canon = CanonicalFindingState(
        raw_finding="f", finding_subject="investigation record",
        deviation_condition="did not document sufficient evidence to support invalidation",
        semantic_type="RECORD", affected_process="Deviation investigation control",
        observed_deviation="x", deviation="x",
    )
    res = asyncio.run(estimate_remediation_cost(
        finding_text="f",
        evidence_ledger=[EvidenceItem(claim="c", status=EvidenceStatus.VERIFIED, source="t")],
        root_cause=_RC(), canonical_state=canon, client=_Down(),
    ))
    blob = " ".join([res.remediation_strategy] + list(res.implementation_activities)
                    + list(res.evidence_improves_estimate)).lower()
    assert "evidence needed" not in blob
    # remediation is about the deficient record, not "update investigation record name"
    assert "record name" not in blob
