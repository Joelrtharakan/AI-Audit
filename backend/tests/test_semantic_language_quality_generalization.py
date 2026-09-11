"""Phase 6 -- auditor-grade natural-language realization and relationship
reasoning, tested by SEMANTIC PROPERTY across arbitrary domains (no benchmark
phrases, no keyword rules).

All findings here are synthetic and finding-agnostic. The pipeline runs on the
deterministic floor (get_llm_client -> None) so the assertions exercise the
DETERMINISTIC realization layer, which is where label-to-prose bugs live.
"""
from __future__ import annotations

import asyncio
import re

import pytest
from unittest.mock import patch

from app.models.agent import InvestigateRequest
from app.services.semantic_subject import subject_is_plural, was_were


async def _report(finding_text: str):
    from app.agent.nodes.core_synthesis import core_synthesis_node
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
    from app.agent.nodes.investigation_planner import plan_investigation_node
    from app.agent.nodes.report_generator import generate_report_node
    from app.agent.nodes.understanding import understand_finding_node

    state = {"request": InvestigateRequest(finding_text=finding_text), "evidence_ledger": [],
             "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0, "trace": [], "errors": []}
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        state = await understand_finding_node(state)
        state = await plan_investigation_node(state)
        state = await core_synthesis_node(state)
        state = await generate_report_node(state)
        state = await final_evidence_verification_node(state)
    return state["report"]


# Diverse domains, diverse grammatical number, diverse structure.
_FINDINGS = [
    "The access-control logs for server room B were incomplete for March.",
    "Two warehouse forklifts lacked current safety-inspection records.",
    "The analytical method used for batch release was not revalidated after the instrument change.",
    "Several purchase orders in the procurement system were approved without a second signature.",
    "The fire-suppression system in the east wing had not been serviced within the required interval.",
    "A contractor's site-induction record could not be located during the audit.",
    "Le registre d'etalonnage de la balance BAL-7 etait incomplet.",
    "The patient-transport handover checklists on ward 9 were missing entries for two shifts.",
]


def _all_prose(report) -> list[str]:
    out = []
    for s in report.five_why.steps:
        out += [s.question or "", s.answer or ""]
    for q in report.investigation.questions:
        out += [q.question or "", getattr(q, "objective", "") or ""]
    out += [report.root_cause.narrative or "", report.root_cause.root_cause_basis or ""]
    out += [report.impact_assessment.process_at_risk or "",
            str(getattr(report.impact_assessment, "potential_effect", "") or "")]
    return [t for t in out if t.strip()]


# --------------------------------------------------------------------------- #
# grammatical quality
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("finding", _FINDINGS)
def test_no_subject_verb_number_disagreement_in_why_questions(finding):
    report = asyncio.run(_report(finding))
    for step in report.five_why.steps:
        q = step.question or ""
        # "Why was the <plural NP> ..." is a number error; "Why were the <singular NP>" too
        m = re.match(r"^Why (was|were) (the .+?) (?:incomplete|missing|not\b|unconfirmed|overdue|"
                     r"expired|absent|inadequate|invalid|unavailable|performed\b)", q)
        if m:
            aux, np_ = m.group(1), m.group(2)
            assert aux == was_were(np_), f"number disagreement: {q!r} (np={np_!r})"


@pytest.mark.parametrize("finding", _FINDINGS)
def test_generated_prose_has_no_label_concatenation_artifacts(finding):
    report = asyncio.run(_report(finding))
    for t in _all_prose(finding_report := report) and _all_prose(report):
        low = t.lower()
        # bracketed structured labels leaking into prose
        assert not re.search(r"\[[A-Z_]{3,}\]", t), t
        # em-dash-joined label fragments ("checklist — not completed")
        assert " — not " not in t and " -- not " not in t, t
        # "because <UPPERCASE_LABEL>" / "Why ... <UPPERCASE_LABEL> ..."
        assert not re.search(r"\bbecause [A-Z][A-Z_]{4,}\b", t), t
        # doubled articles / verbs
        assert not re.search(r"\bthe the\b|\bwas was\b|\bwere were\b|\ba a\b", low), t


@pytest.mark.parametrize("finding", _FINDINGS)
def test_five_why_first_step_is_not_circular(finding):
    report = asyncio.run(_report(finding))
    if not report.five_why.steps:
        return
    s0 = report.five_why.steps[0]
    q = re.sub(r"[^a-z ]", " ", (s0.question or "").lower())
    a = re.sub(r"[^a-z ]", " ", (s0.answer or "").lower())
    q_content = {w for w in q.split() if w not in {
        "why", "was", "were", "did", "the", "a", "an", "of", "for", "in", "on", "not", "this", "that"}}
    a_content = {w for w in a.split() if len(w) > 3}
    # the answer must not be a bare restatement of the question's content words
    if q_content and a_content:
        assert not q_content.issubset(a_content) or len(a_content) > len(q_content) + 3, \
            f"circular: Q={s0.question!r} A={s0.answer!r}"


def test_ambiguous_subject_yields_neutral_construction_not_a_broken_noun_phrase():
    # a finding whose subject is generic but which IS actionable: the prose must
    # use a neutral construction, never a broken/empty noun phrase.
    report = asyncio.run(_report(
        "The condition observed in the packing area did not meet the applicable requirement."))
    prose = _all_prose(report)
    for t in prose:
        assert "the the" not in t.lower()
        assert not re.search(r"\b(?:affecting|for|in|of)\s*[.?!]", t), t  # dangling preposition
        assert not re.search(r"\bthe\s+$|\bWhy (?:was|were) the\?\s*$", t), t
    joined = " ".join(prose).lower()
    assert any(p in joined for p in (
        "the observed condition", "the affected process", "the identified", "the deviation",
        "the finding", "the observed deviation", "the condition", "the applicable requirement",
        "the packing area"))


# --------------------------------------------------------------------------- #
# relationship reasoning
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("finding", _FINDINGS)
def test_hypotheses_are_causal_propositions_not_bare_fragments(finding):
    report = asyncio.run(_report(finding))
    for h in report.root_cause.candidate_hypotheses:
        stmt = (h.statement or "").strip()
        assert len(stmt.split()) >= 4, f"hypothesis too thin to be a proposition: {stmt!r}"
        # it predicates something (a verb / relation), not just a noun phrase
        assert re.search(r"\b(?:caused|led|resulted|because|due to|failed|was not|were not|"
                         r"did not|absen|lack|inadequ|insufficient|not (?:performed|implemented|"
                         r"followed|available|established)|may have|could have)\b", stmt.lower()), \
            f"hypothesis is not a causal proposition: {stmt!r}"
        assert h.status in ("POSSIBLE", "SUPPORTED", "REJECTED", "TIED")
        assert h.status != "VERIFIED"


@pytest.mark.parametrize("finding", _FINDINGS)
def test_supporting_and_discriminating_evidence_are_functionally_distinct(finding):
    report = asyncio.run(_report(finding))
    for h in report.root_cause.candidate_hypotheses:
        disc = (getattr(h, "discrimination_evidence", None) or "").strip()
        supp = " ".join(getattr(h, "supporting_evidence", None) or [])
        if disc and supp:
            assert disc.lower() != supp.lower(), h.name
            # a discrimination criterion names something that would CHANGE the
            # relative plausibility -- it is not a bare "this hypothesis exists"
            assert re.search(r"\b(?:distinguish|differ|versus|vs\.?|whether|rather than|"
                             r"instead of|rule out|confirm|refute|common to|specific to|"
                             r"present|absent|show|indicate)\b", disc.lower()), disc


@pytest.mark.parametrize("finding", _FINDINGS)
def test_investigation_questions_target_a_specific_information_gap(finding):
    report = asyncio.run(_report(finding))
    qs = report.investigation.questions
    if not qs:
        pytest.skip("deterministic floor produced a minimal plan for this shape")
    generic = sum(1 for q in qs if (q.question or "").strip().lower() in (
        "what caused this?", "why did this happen?", "what is the root cause?"))
    assert generic == 0, "bare generic cause question present"
    for q in qs:
        assert (q.question or "").strip().endswith("?")
        assert getattr(q, "evidence_required", None)


# --------------------------------------------------------------------------- #
# metamorphic
# --------------------------------------------------------------------------- #

def test_irrelevant_narrative_does_not_change_rca_status_or_review():
    base = "The calibration certificate for gauge G-4 had expired."
    padded = ("During the second day of the scheduled surveillance audit, while reviewing the "
              "metrology cabinet in bay 7 with the day-shift lead present, it was noted that "
              "the calibration certificate for gauge G-4 had expired. This was also discussed "
              "briefly at the previous month's quality meeting.")
    a, b = asyncio.run(_report(base)), asyncio.run(_report(padded))
    assert str(a.root_cause.status) == str(b.root_cause.status)
    assert a.review.required == b.review.required is True
    assert a.root_cause.risk_of_recurrence == b.root_cause.risk_of_recurrence


def test_paraphrase_of_the_same_deficiency_yields_the_same_rca_status():
    for ft in ("The maintenance log for chiller CH-2 was not completed for week 14.",
               "Week 14 maintenance-log entries for chiller CH-2 are missing.",
               "No maintenance record exists for chiller CH-2 covering week 14."):
        r = asyncio.run(_report(ft))
        assert str(r.root_cause.status).endswith("NOT_ESTABLISHED")
        assert r.review.required is True
