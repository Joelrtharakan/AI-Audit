"""Referenced-Evidence Boundary: REFERENCED EVIDENCE != INSPECTED EVIDENCE.

A document (report/certificate/record/log) the finding merely mentions,
cites, references, or attaches -- but which was NOT available for actual
inspection -- may establish only that it was referenced and unavailable. It
must never contaminate the finding's subject/affected_object/process_at_risk,
and its type/name (e.g. "calibration report") must never seed a causal
hypothesis, root cause, or 5-Why answer about its contents.

Detection is structural (a reference-verb clause co-occurring with an
unavailability clause), never a per-document-type keyword list -- see
app.services.semantic_subject.detect_referenced_unavailable_documents and
app.agent.causal_guard.hypothesis_asserts_referenced_document_content.
"""

from __future__ import annotations

import pytest

from app.models.agent import (
    CandidateHypothesis,
    CanonicalFindingState,
    CapaAnalysis,
    CapaStatus,
    EvidenceItem,
    EvidenceStatus,
    ReferencedDocumentInfo,
)

MAIN_DEVIATION = (
    "The audit observation states that the equipment was operated outside its "
    "validated range. "
)

# (label, clause, expected document_type substring)
ADVERSARIAL_CASES = [
    ("A", "The auditor cited a calibration certificate, but it was unavailable.", "calibration certificate"),
    ("B", "The finding refers to a validation report that the AI could not access.", "validation report"),
    ("C", "The auditor referenced the maintenance record, which was not attached.", "maintenance record"),
    ("D", "The inspector cited a training record that could not be retrieved.", "training record"),
    ("E", "The auditor referred to an equipment qualification document that was unavailable.", "qualification document"),
    ("F", "The attached certificate was mentioned but could not be reviewed.", "certificate"),
]

_FORBIDDEN_SUBJECT_WORDS = {
    "calibration", "validation", "maintenance", "training", "qualification",
    "certificate", "certification", "report", "document",
}


@pytest.mark.parametrize("label,clause,expected_doc_type", ADVERSARIAL_CASES)
def test_referenced_unavailable_document_detected_and_subject_stays_clean(label, clause, expected_doc_type):
    from app.services.semantic_subject import resolve_deviation, resolve_referenced_documents

    finding = MAIN_DEVIATION + clause
    refs = resolve_referenced_documents(finding)
    assert refs, f"Case {label}: expected a referenced-unavailable document to be detected"
    assert expected_doc_type in refs[0]["document_type"].lower()

    resolved = resolve_deviation(finding)
    assert resolved.subject, f"Case {label}: subject must still resolve"
    subject_words = set(resolved.subject.lower().split())
    contaminated = subject_words & _FORBIDDEN_SUBJECT_WORDS
    assert not contaminated, (
        f"Case {label}: subject {resolved.subject!r} was contaminated by referenced-document "
        f"vocabulary {contaminated}"
    )


def test_referenced_document_does_not_fire_when_document_is_actually_available():
    from app.services.semantic_subject import resolve_referenced_documents

    finding = (
        "The attached calibration report shows that the calibration certificate "
        "expired on 10 August 2026."
    )
    assert resolve_referenced_documents(finding) == []


def test_referenced_document_does_not_fire_for_available_training_record():
    from app.services.semantic_subject import resolve_referenced_documents

    finding = "The attached training record shows that the operator completed training on 5 August."
    assert resolve_referenced_documents(finding) == []


def test_classify_finding_specificity_recognizes_structural_deviation_condition():
    """A finding with no entity ID, date, or reported statement but a
    clear, structurally-captured deviation condition ("operated outside its
    validated range") must not be classified LOW -- Section 9: this finding
    is NOT low-specificity merely because it lacks an entity/date/quote."""
    from app.services.semantic_subject import classify_finding_specificity

    assert classify_finding_specificity(
        MAIN_DEVIATION, [], None, "operated outside its validated range",
    ) != "LOW"
    # Still correctly LOW when there is truly no condition captured either.
    assert classify_finding_specificity(
        "The department is not following the required procedure correctly.", [], None, None,
    ) == "LOW"


def test_hypothesis_asserting_referenced_document_content_is_flagged():
    from app.agent.causal_guard import hypothesis_asserts_referenced_document_content

    docs = [ReferencedDocumentInfo(document_type="calibration report", raw_span="x")]
    assert hypothesis_asserts_referenced_document_content(
        "The calibration report shows the calibration certificate had expired.", docs,
    )


def test_hypothesis_unrelated_to_referenced_document_is_not_flagged():
    from app.agent.causal_guard import hypothesis_asserts_referenced_document_content

    docs = [ReferencedDocumentInfo(document_type="maintenance record", raw_span="x")]
    assert not hypothesis_asserts_referenced_document_content(
        "The equipment was operated outside its validated range.", docs,
    )


def test_hypothesis_merely_restating_reference_unavailability_is_not_flagged():
    from app.agent.causal_guard import hypothesis_asserts_referenced_document_content

    docs = [ReferencedDocumentInfo(document_type="calibration report", raw_span="x")]
    assert not hypothesis_asserts_referenced_document_content(
        "The calibration report was referenced but was not available for inspection.", docs,
    )


@pytest.mark.asyncio
async def test_final_evidence_verification_rejects_hypothesis_asserting_document_content():
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node

    finding = MAIN_DEVIATION + (
        "The auditor referenced an attached calibration report, but the report was "
        "not available to the AI agent."
    )
    ledger = [
        EvidenceItem(claim="The equipment was operated outside its validated range.",
                     source="finding_text", status=EvidenceStatus.VERIFIED),
    ]
    canonical = CanonicalFindingState(
        raw_finding=finding,
        observed_deviation="equipment — operated outside its validated range",
        finding_subject="equipment",
        referenced_documents=[
            ReferencedDocumentInfo(document_type="calibration report", raw_span="x"),
        ],
    )
    bad_hyp = CandidateHypothesis(
        id="H1", name="CALIBRATION_EXPIRED",
        statement="The calibration report shows the calibration certificate had expired before use.",
        status="POSSIBLE", evidence_needed="x",
    )
    mock_state = {
        "request": type("Request", (), {"finding_text": finding})(),
        "evidence_ledger": ledger,
        "canonical_finding_state": canonical,
        "root_cause": type("RC", (), {
            "narrative": None, "statement": None, "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED", "candidate_hypotheses": [bad_hyp],
        })(),
        "investigation_plan": type("Inv", (), {"questions": [], "areas": [], "evidence_to_collect": []})(),
        "capa_analysis": CapaAnalysis(status=CapaStatus.INVESTIGATION_REQUIRED, conditional_actions=[]),
        "ca_draft": None,
        "trace": [],
        "errors": [],
    }
    result = await final_evidence_verification_node(mock_state)
    assert "CALIBRATION_EXPIRED" not in [h.name for h in result["root_cause"].candidate_hypotheses]
