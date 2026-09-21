"""Phase 9.3 -- final realization firewall (finding-agnostic).

  1. No structured enum/status token (UNKNOWN / NOT_ESTABLISHED / VERIFIED / ...)
     leaks into auditor-facing prose, across EVERY 5-Why evidence-boundary
     branch -- including the canonical-context branch (which previously emitted
     "Root cause is NOT_ESTABLISHED." verbatim).
  2. An investigation gap / process / fact is not asserted to be an "Evidence
     Artifact".  (frontend label -- covered by a source assertion here.)
  3. The generic immediate action is a condition-aware verification action with
     no invented operational restriction.

All synthetic, domain-varied. No benchmark text. No raw-finding-text keywords.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
from unittest.mock import patch

from app.agent.nodes.five_why_fallback import build_deterministic_five_why
from app.models.agent import EvidenceItem, EvidenceStatus, InvestigateRequest
from app.services.canonical_semantic_models import CanonicalFindingContext

_ENUM = re.compile(
    r"(?<![A-Za-z_])(UNKNOWN|NOT_ESTABLISHED|TO_BE_CONFIRMED|POSSIBLE|VERIFIED|SUPPORTED|"
    r"UNVERIFIED|NOT_ASSESSABLE|MIXED|REPORTED|EVIDENCE_BOUNDARY|STATED_UNVERIFIED|CONTRADICTED)"
    r"(?![A-Za-z])")
_GLUE = re.compile(r"[a-z0-9)\"']\.?(UNKNOWN|NOT_ESTABLISHED|VERIFIED|POSSIBLE|SUPPORTED|NOT_ASSESSABLE|REPORTED)\b")


async def _state(finding_text: str):
    from app.agent.nodes.core_synthesis import core_synthesis_node
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
    from app.agent.nodes.investigation_planner import plan_investigation_node
    from app.agent.nodes.report_generator import generate_report_node
    from app.agent.nodes.understanding import understand_finding_node

    st = {"request": InvestigateRequest(finding_text=finding_text), "evidence_ledger": [],
          "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0, "trace": [], "errors": []}
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        for n in (understand_finding_node, plan_investigation_node, core_synthesis_node,
                  generate_report_node, final_evidence_verification_node):
            st = await n(st)
    return st


_DOMAINS = [
    "The purge cycle on reactor R-2 was not logged for three batches.",
    "Contractor competency verification was not completed before high-risk work began.",
    "The dispensing scale in room 214 drifted outside tolerance and no adjustment was recorded.",
    "Two invoices were approved above the delegated authority limit without escalation.",
    "The building evacuation drill for the second quarter was not conducted.",
    "Machine guarding on the stamping press was found removed during operation.",
]


def _all_prose(report):
    out = []
    for s in report.five_why.steps:
        out += [s.question or "", s.answer or ""]
    for q in report.investigation.questions:
        out += [q.question or "", getattr(q, "objective", "") or ""]
    out += [report.root_cause.narrative or "", report.root_cause.root_cause_basis or "",
            report.impact_assessment.process_at_risk or "",
            str(getattr(report.impact_assessment, "potential_effect", "") or "")]
    return [t for t in out if t and t.strip()]


# --------------------------------------------------------------------------- #
# 1. enum-to-prose firewall -- full pipeline, deterministic floor
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("finding", _DOMAINS)
def test_no_enum_token_leaks_into_any_prose_field(finding):
    report = asyncio.run(_state(finding))["report"]
    for t in _all_prose(report):
        assert not _ENUM.search(t), f"enum token in prose: {t!r}"
        assert not _GLUE.search(t), f"glued status token: {t!r}"


# --------------------------------------------------------------------------- #
# 1b. every 5-Why evidence-boundary branch, incl. the CANONICAL-CONTEXT branch
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("obs", [
    "the purge cycle was not logged for three batches",
    "competency verification was not completed",
    "the dispensing scale drifted outside tolerance",
    "invoices were approved above the delegated limit",
    "the guarding was found removed during operation",
])
def test_canonical_context_five_why_boundary_is_clean(obs):
    ctx = CanonicalFindingContext.model_validate({
        "primary_deviation": obs, "observed_condition": obs,
        "root_cause_status": "NOT_ESTABLISHED",
        "information_gaps": ["the applicable governing requirement"],
        "candidate_hypotheses": [],
    })
    fw = build_deterministic_five_why(
        "x", [EvidenceItem(claim=obs, status=EvidenceStatus.VERIFIED, source="t")],
        semantic_context=ctx)
    assert fw.is_complete is False
    for s in fw.steps:
        for text in (s.question or "", s.answer or ""):
            assert not _ENUM.search(text), f"enum leaked: {text!r}"
            assert not _GLUE.search(text), f"glued token: {text!r}"
            if text.strip():
                assert text.strip().endswith((".", "?")), f"fragment: {text!r}"
        assert s.status in ("UNKNOWN", "REPORTED", "MIXED")  # never VERIFIED at the boundary
    joined = " ".join(s.answer or "" for s in fw.steps).lower()
    assert "not been established" in joined or "does not establish" in joined
    # no causal assertion
    assert not re.search(r"\b(caused by|because of|due to|resulted from)\b", joined)


# --------------------------------------------------------------------------- #
# 1c. the canonical-context branch realizes the QUESTION from the canonical
# subject/condition (finding_subject / observed_condition) via the same
# grammar machinery used everywhere else, instead of a generic "observed
# condition affecting X" wrapper -- reproduces and closes a real runtime
# defect (production-hardening / runtime-trace charter, defect 5).
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("subject,condition", [
    ("the purge cycle on reactor R-2", "was not logged for three batches"),
    ("the vendor's audit certificate", "had expired at the time of the shipment"),
    ("the dispensing scale in room 214", "drifted outside tolerance"),
    ("the delegated authority limit for invoices", "was exceeded"),
])
def test_canonical_context_five_why_question_uses_canonical_subject(subject, condition):
    obs = f"{subject} {condition}"
    ctx = CanonicalFindingContext.model_validate({
        "primary_deviation": obs, "observed_condition": condition, "finding_subject": subject,
        "root_cause_status": "NOT_ESTABLISHED", "information_gaps": [], "candidate_hypotheses": [],
    })
    fw = build_deterministic_five_why(
        "x", [EvidenceItem(claim=obs, status=EvidenceStatus.VERIFIED, source="t")],
        semantic_context=ctx)
    q = fw.steps[0].question
    assert "observed condition affecting" not in q.lower(), f"generic wrapper not replaced: {q!r}"
    assert q.startswith("Why")
    assert not _ENUM.search(q)


def test_canonical_context_five_why_question_falls_back_when_no_canonical_subject():
    # No finding_subject on the context -> the honest generic fallback still
    # fires (never a fabricated subject).
    obs = "the record was incomplete"
    ctx = CanonicalFindingContext.model_validate({
        "primary_deviation": obs, "observed_condition": obs,
        "root_cause_status": "NOT_ESTABLISHED", "information_gaps": [], "candidate_hypotheses": [],
    })
    fw = build_deterministic_five_why(
        "x", [EvidenceItem(claim=obs, status=EvidenceStatus.VERIFIED, source="t")],
        semantic_context=ctx)
    assert "observed condition affecting" in fw.steps[0].question.lower()


def test_canonical_context_branch_does_not_say_the_literal_enum():
    ctx = CanonicalFindingContext.model_validate({
        "primary_deviation": "the record was incomplete", "observed_condition": "the record was incomplete",
        "root_cause_status": "NOT_ESTABLISHED", "information_gaps": [], "candidate_hypotheses": [],
    })
    fw = build_deterministic_five_why(
        "x", [EvidenceItem(claim="the record was incomplete", status=EvidenceStatus.VERIFIED, source="t")],
        semantic_context=ctx)
    ans = " ".join(s.answer or "" for s in fw.steps)
    assert "NOT_ESTABLISHED" not in ans
    assert "the root cause has not been established" in ans.lower()


# --------------------------------------------------------------------------- #
# 2. investigation gap != evidence artifact (frontend label)
# --------------------------------------------------------------------------- #

def test_frontend_does_not_label_the_mixed_list_as_evidence_artifacts():
    js = (Path(__file__).resolve().parent.parent.parent / "frontend" / "assets" / "js" / "lqms_ai.js").read_text()
    assert "Additional Evidence Artifacts to Collect" not in js
    assert "Additional evidence and open points to resolve" in js


def test_frontend_five_why_answer_and_status_badge_are_text_separated():
    # Production-hardening / runtime-trace charter Issue 1: the 5-Why answer
    # <span> and the structured status-badge <span> sit side by side in a
    # flex row -- CSS flex spacing is visual-only and does not exist for
    # copy/paste or any programmatic text extraction, so without a literal
    # space between the two concatenated strings a sentence and a raw status
    # token glue together with no separator the moment the HTML is read as
    # plain text (observed: "...before a causal conclusion can be
    # drawn.UNKNOWN"). Asserts the source literally separates them.
    js = (Path(__file__).resolve().parent.parent.parent / "frontend" / "assets" / "js" / "lqms_ai.js").read_text()
    idx = js.index("Requires verification")
    window = js[idx:idx + 800]
    concat_idx = window.index("html += ", window.index("</span>"))
    # the next string literal concatenated after the answer's closing
    # </span> must itself start with a literal space, not immediately "<span"
    next_string_start = window.index('"', concat_idx)
    assert window[next_string_start:next_string_start + 2] == '" '


# --------------------------------------------------------------------------- #
# 3. immediate action -- condition-aware, no invented restriction
# --------------------------------------------------------------------------- #

_BANNED_ACTION = re.compile(
    r"\b(quarantine|recall|shut ?down|release hold|stop production|cease operations|"
    r"before permitting independent execution or release)\b", re.IGNORECASE)


@pytest.mark.parametrize("finding", _DOMAINS)
def test_immediate_action_is_grounded_verification_without_invented_restriction(finding):
    st = asyncio.run(_state(finding))
    ia = getattr(st.get("ca_draft"), "immediate_action", "") or ""
    assert ia
    assert not _BANNED_ACTION.search(ia), f"invented restriction: {ia!r}"
    assert re.search(r"\b(verif|review|assess|confirm|reconcile|determine)\w*\b", ia.lower())
    assert not _ENUM.search(ia)
    assert ia.strip().endswith(".")


# --------------------------------------------------------------------------- #
# 4. metamorphic
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("actor", ["the day-shift operator", "a contractor", "the QA reviewer", "an auditor"])
def test_actor_name_does_not_change_enum_firewall_or_rca(actor):
    r = asyncio.run(_state(
        f"{actor.capitalize()} identified that the flow-test record for line 7 was missing."))["report"]
    assert str(r.root_cause.status).endswith("NOT_ESTABLISHED")
    for t in _all_prose(r):
        assert not _ENUM.search(t)


def test_irrelevant_narrative_does_not_create_an_artifact_or_leak_an_enum():
    a = asyncio.run(_state("The calibration certificate for gauge G-4 had expired."))["report"]
    b = asyncio.run(_state(
        "During the second day of the audit, while walking bay 7 with the shift lead, it was noted "
        "that the calibration certificate for gauge G-4 had expired; this was also raised last month."))["report"]
    assert str(a.root_cause.status) == str(b.root_cause.status)
    for t in _all_prose(a) + _all_prose(b):
        assert not _ENUM.search(t)
