"""Regression suite for _reportedly_clause's verb-phrase classification.

Locks in the fix in app.agent.nodes.core_synthesis._reportedly_clause: its
final fallback branch used to treat ANY leading lowercase word as an
omitted-action verb infinitive ("X reportedly failed to <cond>"), which also
matched an already-inflected past-tense predicate describing something that
DID happen -- producing tense-mismatched, semantically-inverted text like
"the payment gateway reportedly failed to processed the transaction twice".

Now reuses the same closed infinitive-verb whitelist
(app.services.semantic_subject._TRANSITIVE_FAILED_TO_VERBS) that
format_deviation_why_question already uses for the identical classification
problem, so an unrecognized/past-tense condition renders as a neutral dash
clause instead. Scenarios deliberately span unrelated domains (finance,
lab records, logistics) to confirm this is a structural/morphological fix,
not a domain-specific patch.
"""

from __future__ import annotations

from app.agent.nodes.core_synthesis import _reportedly_clause


def test_past_tense_event_condition_not_rendered_as_failed_to():
    result = _reportedly_clause(
        "The payment gateway", "processed the transaction twice due to a retry-queue duplication bug",
    )
    assert "reportedly failed to processed" not in result
    assert "reportedly failed to" not in result
    assert "processed the transaction twice" in result


def test_past_tense_condition_across_lab_domain():
    result = _reportedly_clause("The analyzer", "recorded an incorrect calibration value")
    assert "reportedly failed to recorded" not in result
    assert "reportedly failed to" not in result


def test_past_tense_condition_across_logistics_domain():
    result = _reportedly_clause("The shipment", "routed to the wrong distribution center")
    assert "reportedly failed to routed" not in result
    assert "reportedly failed to" not in result


def test_genuine_omitted_action_verb_phrase_still_uses_failed_to():
    """The original, correctly-handled shape (captured from a 'failed to
    <verb> <object>' source pattern) must still render exactly as before."""
    result = _reportedly_clause("The document-control system", "distribute the revised SOP")
    assert result == "The document-control system reportedly failed to distribute the revised SOP."


def test_adjective_predicate_condition_unaffected():
    result = _reportedly_clause("The checklist", "incomplete")
    assert result == "The checklist was reportedly incomplete."


def test_negated_participle_condition_unaffected():
    result = _reportedly_clause("The checklist", "not completed")
    assert result == "The checklist was reportedly not completed."


def test_quantity_descriptor_condition_unaffected():
    result = _reportedly_clause("The batch", "approximately 12 units short")
    assert result == "The batch was reportedly associated with approximately 12 units short."


def test_unknown_condition_unaffected():
    result = _reportedly_clause("The record", None)
    assert result == "The record was reportedly in a condition that has not been verified against applicable requirements."
