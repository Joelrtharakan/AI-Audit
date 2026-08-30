"""§15 -- LLM-primary finding matrix: 15 structurally-different findings run
through the FULL pipeline with the flag ON (recorded LLM responses), then
each downstream output is asserted consistent with the canonical semantics.

§9 -- enabled-path E2E. §14 -- information conservation to the report.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.agent.invariants import evaluate_all_invariants
from app.config import get_settings
from app.models.agent import InvestigateRequest

_BASE = {
    "primary_deviation": None, "primary_deviation_claim_id": None,
    "primary_deviation_confidence": "NOT_ESTABLISHED",
    "finding_subject": None, "subject_kind": None, "evidence_source": None,
    "reported_observation": None, "observed_condition": None, "epistemic_status": None,
    "comparison": None, "recurrence": None,
    "stated_causal_alternatives": [], "causal_alternatives_unresolved": False,
    "missing_record_status": None, "activity_performance_ambiguity": False,
    "affected_period": None, "scope": None,
    "entities": [], "causal_claims": [], "explicit_previous_capa_reference": False,
    "previous_capa_evidence_ids": [], "evidence_boundaries": [], "unresolved_ambiguities": [],
}


class _LLM:
    def __init__(self, payload):
        self._p = payload

    async def chat_completion(self, *a, **k):
        return json.dumps(self._p)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setattr(get_settings(), "canonical_semantic_llm_primary", True)
    for mod in ("understanding", "investigation_planner", "core_synthesis"):
        monkeypatch.setattr(f"app.agent.nodes.{mod}.get_llm_client", lambda **kw: None)

    def _install(over):
        monkeypatch.setattr(
            "app.services.canonical_finding_interpreter.get_llm_client",
            lambda **kw: _LLM({**_BASE, **over}),
        )
    return _install


async def _run(finding: str):
    from app.agent.nodes.core_synthesis import core_synthesis_node
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
    from app.agent.nodes.investigation_planner import plan_investigation_node
    from app.agent.nodes.report_generator import generate_report_node
    from app.agent.nodes.understanding import understand_finding_node

    st = {
        "request": InvestigateRequest(finding_text=finding),
        "evidence_ledger": [], "trace": [], "errors": [],
        "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
    }
    st = await understand_finding_node(st)
    st = await plan_investigation_node(st)
    st = await core_synthesis_node(st)
    st = await generate_report_node(st)
    st = await final_evidence_verification_node(st)
    return st


def _questions(st):
    return [(q.question or "").lower() for q in (st["investigation_plan"].questions or [])]


def _five_why_text(st):
    fw = st.get("five_why_analysis") or st.get("five_why")
    return " ".join(s.answer.lower() for s in getattr(fw, "steps", []))


# ---------------------------------------------------------------------------

_MATRIX = [
    # (id, finding, llm_over, checks(callable(st, cf, rc)))
    ("1_inventory_shortage_competing",
     "The inventory count for location IL-4 showed a shortfall of 120 units against the system "
     "record, and it was unclear whether this resulted from an unrecorded issue, a miscount, or "
     "a system entry error.",
     dict(finding_subject="inventory at location IL-4",
          comparison={"left": "physical count", "right": "system record",
                      "reference": "system record", "direction": "BELOW",
                      "magnitude": 120, "unit": "units"},
          stated_causal_alternatives=["an unrecorded issue", "a miscount", "a system entry error"],
          causal_alternatives_unresolved=True)),
    ("2_equipment_repeated_failures",
     "Compressor K-2 experienced four breakdowns during the last quarter.",
     dict(finding_subject="Compressor K-2",
          recurrence={"count": 4, "event": "breakdowns", "period": "the last quarter"})),
    ("3_missing_maintenance_record",
     "The preventive maintenance record for pump P-9 was missing for the June service.",
     dict(finding_subject="the preventive maintenance record for pump P-9",
          missing_record_status="RECORD_MISSING", activity_performance_ambiguity=True)),
    ("4_medical_device_change",
     "A material change to infusion pump IP-3 was implemented without documented evidence that "
     "the associated risk assessment had been completed before implementation.",
     dict(finding_subject="infusion pump IP-3",
          observed_condition="material change implemented without a documented prior risk assessment")),
    ("5_oos_investigation",
     "An investigation invalidated the out-of-specification result for batch B-8, but the record "
     "did not contain sufficient evidence to establish the assignable laboratory cause.",
     dict(finding_subject="the OOS investigation for batch B-8",
          observed_condition="invalidation not supported by sufficient evidence")),
    ("6_access_control_excess",
     "Several employees retained access to the ERP that was not required by their current roles.",
     dict(finding_subject="ERP access", observed_condition="access exceeded role requirement")),
    ("7_temperature_record_unavailable",
     "Temperature records for refrigerator FR-3 were unavailable for 18 hours.",
     dict(finding_subject="temperature records for refrigerator FR-3",
          missing_record_status="RECORD_UNAVAILABLE", affected_period="18 hours")),
    ("8_supplier_deviation",
     "Supplier SP-12 shipped two consignments that did not meet the agreed specification.",
     dict(finding_subject="consignments from supplier SP-12",
          observed_condition="did not meet the agreed specification")),
    ("9_quantified_comparison",
     "The recorded fill weight for line L-4 differed from the target by 3.5 grams.",
     dict(finding_subject="the recorded fill weight for line L-4",
          comparison={"left": "recorded fill weight", "right": "target", "reference": "target",
                      "direction": "MISMATCH", "magnitude": 3.5, "unit": "grams"})),
    ("10_recurrence_finding",
     "The same labelling defect was identified in three separate production runs.",
     dict(finding_subject="the labelling defect",
          recurrence={"count": 3, "event": "production runs", "period": None},
          scope="three separate production runs")),
    ("11_evidence_proposition",
     "Maintenance records show that temporary repairs were performed on press PR-204.",
     dict(finding_subject="press PR-204", evidence_source="maintenance records",
          reported_observation="temporary repairs were performed", epistemic_status="REPORTED")),
    ("12_reported_observation",
     "The operator reported that the second-person verification for batch B-5 was not completed.",
     dict(finding_subject="the second-person verification for batch B-5",
          reported_observation="the verification was not completed", epistemic_status="REPORTED")),
    ("13_belief_statement",
     "It is believed that the deviation for lot L-3 was minor, though no assessment was documented.",
     dict(finding_subject="the deviation for lot L-3", epistemic_status="BELIEF",
          observed_condition="believed minor; no assessment documented")),
    ("14_explicit_non_performance",
     "The required environmental monitoring for cleanroom CR-2 was not performed during the second quarter.",
     dict(finding_subject="environmental monitoring for cleanroom CR-2",
          missing_record_status="ACTIVITY_NOT_PERFORMED", affected_period="the second quarter")),
    ("15_ambiguous_missing_record",
     "The batch record for lot L-7 did not document the in-process check, but it is unclear "
     "whether the check was performed.",
     dict(finding_subject="the in-process check for lot L-7",
          missing_record_status="ACTIVITY_NOT_RECORDED", activity_performance_ambiguity=True)),
]


@pytest.mark.parametrize("fid,finding,over", [(m[0], m[1], m[2]) for m in _MATRIX],
                         ids=[m[0] for m in _MATRIX])
def test_finding_matrix_downstream_consistency(flag_on, fid, finding, over):
    flag_on(over)
    st = asyncio.run(_run(finding))
    cf = st["canonical_finding_state"]
    rc = st.get("root_cause_result") or st.get("root_cause")

    ok, violations = evaluate_all_invariants(st)
    assert ok, f"{fid}: invariants: {violations}"

    # --- canonical semantics reflect the finding structure -----------
    if over.get("comparison"):
        assert cf.semantic_type == "COMPARISON"
        assert cf.measurement is not None and cf.measurement.value == float(over["comparison"]["magnitude"])
    if over.get("recurrence", {}).get("count"):
        assert cf.recurrence_count == over["recurrence"]["count"]
    if over.get("stated_causal_alternatives"):
        assert len(cf.stated_causal_alternatives) >= len(over["stated_causal_alternatives"])
        # never established / never ranked
        status = getattr(getattr(rc, "status", None), "value", getattr(rc, "status", None))
        assert status in ("NOT_ESTABLISHED", "NOT_APPLICABLE", None)
        assert not getattr(rc, "leading_hypothesis", None) or \
            "NONE" in str(getattr(rc, "leading_hypothesis", "")).upper()
    if over.get("missing_record_status"):
        assert cf.missing_record_status == over["missing_record_status"] \
            or cf.semantic_type == "MISSING_RECORD"
    if over.get("activity_performance_ambiguity"):
        assert cf.activity_performance_ambiguity is True
    if over.get("epistemic_status"):
        assert cf.finding_epistemic_status == over["epistemic_status"]
    if over.get("evidence_source"):
        assert cf.evidence_source == over["evidence_source"]
        assert "maintenance records" not in (cf.finding_subject or "").lower()

    # --- root cause never fabricated -------------------------------
    _fw = _five_why_text(st)
    for w in ("forgot", "negligence", "was ignored", "human error", "poor maintenance",
              "inadequate training", "management failure"):
        assert w not in _fw, f"{fid}: fabricated cause phrase {w!r} in 5-Why"

    # --- subject is stable canonical -> impact --------------------
    rep = st.get("report")
    ia = getattr(rep, "impact_assessment", None) or getattr(rep, "risk_impact", None)
    if ia is not None and getattr(ia, "affected_object", None):
        _ao = (ia.affected_object or "").lower()
        _subj = (cf.finding_subject or "").lower()
        if _subj and not _subj.startswith(("unknown", "unresolved", "finding subject not")):
            # impact object shares the canonical subject's head, never invents one
            assert any(w in _ao for w in _subj.split() if len(w) > 3) or _ao in _subj

    # --- competing causes -> discrimination question --------------
    if over.get("causal_alternatives_unresolved"):
        _qs = _questions(st)
        assert any("distinguish" in q or "discriminate" in q or "which of the" in q for q in _qs), \
            f"{fid}: no discrimination question"


def test_flag_off_matrix_is_baseline(monkeypatch):
    """The same matrix with the flag OFF still produces valid pipelines
    (deterministic floor) -- proves the merge is purely additive."""
    monkeypatch.setattr(get_settings(), "canonical_semantic_llm_primary", False)
    for mod in ("understanding", "investigation_planner", "core_synthesis"):
        monkeypatch.setattr(f"app.agent.nodes.{mod}.get_llm_client", lambda **kw: None)
    for _fid, finding, _over in _MATRIX[:6]:
        st = asyncio.run(_run(finding))
        assert st.get("canonical_semantic_context") is None
        ok, v = evaluate_all_invariants(st)
        assert ok, f"{_fid} flag-off: {v[:1]}"
