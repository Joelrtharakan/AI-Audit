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
