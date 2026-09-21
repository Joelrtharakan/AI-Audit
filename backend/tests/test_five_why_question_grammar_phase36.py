"""FINAL OUTPUT-QUALITY HARDENING, Part 6/7: the graph-grounded 5-Why
question builder (app.agent.causal_graph_traversal._graph_node_why_question)
used to interpolate a node's raw label verbatim into a stiff "Why did the
following occur: <label>?" template. Reproduced as a real defect via a live
Ollama transcript: for a finance-domain finding, the rendered question was
"Why did the following occur: journal entry reversing a Q3 revenue accrual
— was posted by a user outside the finance department..." -- leaking the
internal dash-joined subject/condition separator (an artifact of
understanding_node's canonical `observed_deviation` field) directly into
auditor-facing text.

Fixed by reusing the SAME existing declarative_to_why_question/
format_deviation_why_question machinery already used by
analytical_validator.repair_five_why_with_mechanism for this exact
grammatical-repair problem, instead of a second parallel template -- and by
hardening declarative_to_why_question's leading-word lowercasing to not
corrupt a short all-caps subject token (an acronym or an abstract synthetic
ID like "X"/"H1"), while still correctly lowercasing the genuine English
words "A"/"I" when they start a sentence.

Uses abstract, domain-general fixtures -- no finding-specific vocabulary.
"""
from __future__ import annotations

from app.agent.causal_graph_traversal import _graph_node_why_question
from app.models.agent import CausalGraphNode, CausalGraphNodeType
from app.services.semantic_subject import declarative_to_why_question


def _node(label, **kw):
    return CausalGraphNode(node_id="N1", node_type=CausalGraphNodeType.OBSERVED_DEVIATION, label=label, **kw)


def test_dash_joined_deviation_label_never_leaks_separator_into_question():
    label = "OBJECT_A reversal entry — was posted by an actor outside the authorized role without documented approval"
    q = _graph_node_why_question(_node(label))
    assert "—" not in q
    assert "the following occur" not in q.lower()
    assert q.startswith("Why")


def test_plain_declarative_hypothesis_label_does_not_use_stiff_template():
    label = "CONTROL_B was disabled prior to the event."
    q = _graph_node_why_question(_node(label))
    assert "the following occur" not in q.lower()
    assert q == "Why was CONTROL_B disabled prior to the event?"


def test_short_synthetic_placeholder_subject_preserves_case():
    """A bare single-letter/ID-like subject (this session's convention:
    OBJECT_A, H1, X) must not be corrupted by the mid-sentence lowercasing
    step -- distinguishing it from the genuine English article "A"/"I"."""
    assert declarative_to_why_question("X occurred") == "Why X occurred?"
    assert declarative_to_why_question("H1 was refuted") == "Why was H1 refuted?"


def test_genuine_leading_article_still_lowercases():
    assert declarative_to_why_question("A deviation occurred in the process") == \
        "Why did a deviation occur in the process?"
    assert declarative_to_why_question("An actor performed an activity outside the authorized role") == \
        "Why did an actor perform an activity outside the authorized role?"


def test_empty_label_falls_back_to_generic_question_not_empty_string():
    q = _graph_node_why_question(_node(""))
    assert q == "Why did this occur?"


# --------------------------------------------------------------------------- #
# subject-verb number agreement -- a singular head noun that happens to end in
# a common record-type word ("certificate", "record", "report", "result",
# "log") must take "was", not "were". Regression: a tail-word regex matched
# both singular and plural of those nouns and forced "were" unconditionally.
# Domain-general morphology test -- no finding vocabulary, no domain rules.
# --------------------------------------------------------------------------- #

import pytest
from app.services.semantic_subject import format_deviation_why_question, subject_is_plural


@pytest.mark.parametrize("subject", [
    "the calibration record for balance B-12",
    "the vendor's audit certificate",
    "the deviation report",
    "the final inspection result",
    "the equipment maintenance log",
    "the training completion check",
])
def test_singular_record_type_subject_takes_was(subject):
    assert subject_is_plural(subject) is False
    q = format_deviation_why_question(subject, "incomplete")
    assert "Why was " in q and "Why were " not in q


@pytest.mark.parametrize("subject", [
    "the calibration records",
    "the audit certificates",
    "the deviation reports",
    "the inspection results",
    "the maintenance logs",
])
def test_plural_record_type_subject_takes_were(subject):
    assert subject_is_plural(subject) is True
    q = format_deviation_why_question(subject, "incomplete")
    assert "Why were " in q


# --------------------------------------------------------------------------- #
# modal auxiliaries carry their own tense/modality and must never receive a
# second, incompatible auxiliary in front of them (production-hardening /
# runtime-trace charter, Issue 2/3: "Why was X could promote..." is invalid
# and silently strengthens a possibility into a flat assertion). Generalized
# over the full closed set of English modal auxiliaries and multiple
# unrelated domains/subjects -- no finding-specific wording.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("modal", ["could", "can", "may", "might", "would", "should", "must", "will", "shall"])
@pytest.mark.parametrize("subject,rest", [
    ("the purified-water system", "promote biofilm formation"),
    ("the vendor onboarding process", "not meet the specification"),
    ("the control", "have been effective"),
    ("the reconciliation", "not resolve the variance"),
])
def test_modal_auxiliary_is_never_preceded_by_a_second_auxiliary(modal, subject, rest):
    q = format_deviation_why_question(subject, f"{modal} {rest}")
    assert q.startswith(f"Why {modal} ")
    assert " was " not in q.lower() and " did " not in q.lower() and " does " not in q.lower()
    assert rest in q


def test_modal_auxiliary_preserves_the_canonical_modality_word():
    # "could" must remain "could" -- never collapse to a flat past-tense
    # assertion ("promoted") or an unrelated modal.
    q = format_deviation_why_question("the system", "could promote biofilm formation")
    assert "could" in q.lower()
    assert "promoted" not in q.lower()


def test_non_modal_conditions_are_unaffected_by_the_modal_branch():
    assert format_deviation_why_question("the checklist", "was incomplete") == \
        "Why was the checklist incomplete?"
    assert format_deviation_why_question("the SOP", "not include a step") == \
        "Why did the SOP not include a step?"


# --------------------------------------------------------------------------- #
# a subject noun phrase must never be echoed twice within one mechanically
# assembled Why question (production-hardening Phase 9.4 Defect B: "Why were
# the preventive-maintenance controls inadequate preventive-maintenance
# controls?"). Pure string-level deduplication of two spans already
# extracted from the same sentence -- generalized across unrelated subjects.
# --------------------------------------------------------------------------- #

from app.services.semantic_subject import declarative_to_why_question as _d2q


@pytest.mark.parametrize("subject,tail_condition", [
    ("the preventive-maintenance controls", "inadequate preventive-maintenance controls"),
    ("the calibration program", "insufficient calibration program"),
    ("the vendor qualification process", "incomplete vendor qualification process"),
])
def test_subject_is_never_echoed_twice_in_a_why_question(subject, tail_condition):
    q = format_deviation_why_question(subject, tail_condition)
    subj_words = subject.replace("the ", "").lower()
    assert q.lower().count(subj_words) == 1, q


def test_declarative_to_why_question_drops_a_trailing_subject_echo():
    q = _d2q("The preventive-maintenance controls were found to be inadequate preventive-maintenance controls.")
    assert q.lower().count("preventive-maintenance controls") == 1
    assert q == "Why were the preventive-maintenance controls found to be inadequate?"


def test_declarative_to_why_question_unaffected_when_no_echo_present():
    assert _d2q("The preventive-maintenance controls were inadequate.") == \
        "Why were the preventive-maintenance controls inadequate?"
