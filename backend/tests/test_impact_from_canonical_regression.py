"""Pass 27c -- when the canonical LLM semantic interpretation is engaged, the
impact section is built from ITS fields only. A field the interpretation did
not establish is reported "NOT ESTABLISHED", never synthesised from a
deterministic `semantic_type` template or the finding's own discovery phrase
(spec §1/§8/§13).
"""

from __future__ import annotations

from app.agent.nodes.core_synthesis import _derive_deterministic_impact
from app.services.canonical_semantic_models import CanonicalFindingContext


class _Canon:
    """Stand-in for the deterministic CanonicalFindingState -- deliberately
    carries the kind of guesses the resolver makes so the test proves they are
    NOT used when the semantic context is present."""
    finding_subject = "safety guards"
    affected_object = "safety guards"
    affected_process = "safety guard operational process"   # resolver guess
    affected_period = "inspection of the production area"    # discovery phrase
    relevant_change = None
    control_at_risk = None
    actor = None
    semantic_type = "RECURRENCE"                             # resolver mis-classification
    occurrence_population = "2 damaged safety guards"
    deviation_condition = "were found with damaged safety guards"


FINDING = (
    "During inspection of the production area, two machines were found with damaged "
    "safety guards. Engineering determined that both guards require replacement."
)


def test_impact_uses_canonical_fields_not_deterministic_templates():
    sc = CanonicalFindingContext(
        finding_subject="safety guards",
        observed_condition="damaged safety guards",
        root_cause_status="NOT_ESTABLISHED",
        affected_process="safety guard maintenance",   # LLM established this one
        scope="two machines",
        # affected_period deliberately NOT set by the LLM
    )
    impact, clean_noun, _topic, _actor = _derive_deterministic_impact(
        FINDING, _Canon(), "damaged safety guards", semantic_context=sc,
    )

    assert impact.process_at_risk == "safety guard maintenance"        # LLM value
    assert impact.affected_period == "NOT ESTABLISHED"                 # not the discovery phrase
    assert "operational process" not in (impact.process_at_risk or "")
    assert impact.relevant_change == "NOT ESTABLISHED"
    assert impact.potential_effect == "NOT ESTABLISHED"                # no template
    assert "across" not in (impact.potential_effect or "")
    assert impact.impact_observed == "damaged safety guards"          # clean, no "were found with"
    # affected_object must never be a full clause or the raw deviation fragment
    assert "were found with" not in (impact.affected_object or "")


def test_impact_period_not_established_when_neither_layer_has_one():
    sc = CanonicalFindingContext(
        finding_subject="safety guards",
        observed_condition="damaged safety guards",
        root_cause_status="NOT_ESTABLISHED",
    )
    impact, _n, _t, _a = _derive_deterministic_impact(
        FINDING, _Canon(), "damaged safety guards", semantic_context=sc,
    )
    assert impact.affected_period == "NOT ESTABLISHED"
    assert impact.process_at_risk == "NOT ESTABLISHED"
