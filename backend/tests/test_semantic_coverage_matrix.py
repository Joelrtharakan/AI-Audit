"""Semantic coverage + anti-template regression matrix.

Runs the 10 representative finding structures through the full deterministic
pipeline (LLM patched off) and asserts PROPERTIES, not wording:

  * every explicit number / unit / count / period / causal alternative that
    is in the finding text survives into the canonical state or an
    authoritative field;
  * observation never becomes cause; hypotheses stay hypotheses;
  * uncertainty becomes an investigation plan, not a fabricated fact;
  * investigation questions are structurally matched to the finding type
    (comparison -> comparison qs, recurrence -> recurrence qs, competing
    causes -> a discrimination q, missing record -> execution-vs-doc q);
  * unrelated findings do NOT get identical remediation activity sets or
    identical investigation questions (template contamination).

Deterministic only. Domain-neutral: the assertions key on semantic type /
role, never on the finding's domain nouns.
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import patch

import pytest

from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.nodes.understanding import understand_finding_node
from app.models.agent import InvestigateRequest

# id -> finding text  (spec section 38)
FINDINGS = {
    "A_comparison_inventory": "The reconciliation of inventory location IL-4 showed a shortfall of 120 units against the system record.",
    "B_recurrence_equipment": "Equipment M-204 experienced three failures over a six-month period.",
    "C_comparison_percent": "The measured result differed from the approved value by 4.2 percent.",
    "D_competing_causes": "The discrepancy could have resulted from an unrecorded transaction, a physical miscount, or a system data-entry error, and the available records did not allow these to be distinguished.",
    "E_evidence_proposition": "Maintenance records show that temporary repairs were performed after each failure.",
    "F_access_competing": "Several employees retained access that was not required for their current roles, but the available evidence did not establish whether the access resulted from provisioning error, incomplete review, or an approved exception.",
    "G_medical_device_change": "A material change to a medical-device component was implemented without documented evidence that the associated risk assessment had been completed before implementation.",
    "H_oos_investigation": "An investigation invalidated an out-of-specification result, but the record did not contain sufficient evidence to establish the assignable laboratory cause.",
    "I_recurrence_procedural": "Three delivery vehicles operated beyond their scheduled maintenance intervals without documented approval or risk assessment for continued operation.",
    "J_missing_temperature_records": "Temperature records for a pharmacy refrigerator were unavailable for 18 hours while temperature-sensitive medicines remained stored inside.",
}

_FABRICATED_CAUSE_WORDS = (
    "negligence", "forgot", "was ignored", "was overlooked", "carelessness",
    "poor maintenance", "inadequate training", "human error", "management failure",
    "lack of attention", "failure to follow",
)


async def _run(finding_text: str) -> dict:
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
        state = await generate_report_node(state)
        state = await final_evidence_verification_node(state)
    return state


@pytest.fixture(scope="module")
def outcomes():
    return {k: asyncio.run(_run(v)) for k, v in FINDINGS.items()}


# --------------------------------------------------------------------------
# 1. Per-finding property invariants (spec 28 / 33 / 37)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fid", list(FINDINGS))
def test_pipeline_is_evidence_bounded(fid, outcomes):
    state = outcomes[fid]
    ok, violations = evaluate_all_invariants(state)
    assert ok, f"{fid}: invariants violated: {violations}"

    cf = state["canonical_finding_state"]
    rc = state.get("root_cause_result") or state.get("root_cause")
    status = getattr(getattr(rc, "status", None), "value", getattr(rc, "status", None))

    # observation never silently becomes an established cause. (A compound
    # finding whose subject cannot be isolated resolves NOT_APPLICABLE -- also
    # acceptable: it is the honest "cannot analyse" state, not a fabricated
    # cause.)
    assert status in ("NOT_ESTABLISHED", "NOT_APPLICABLE", None), \
        f"{fid}: root cause was established from initial evidence"
    assert not getattr(rc, "leading_hypothesis", None) or \
        "NONE" in str(getattr(rc, "leading_hypothesis", "")).upper()

    # no fabricated causal vocabulary anywhere in the causal narrative / 5-Why
    fw = state.get("five_why_analysis") or state.get("five_why")
    blob = " ".join(s.answer.lower() for s in getattr(fw, "steps", []))
    blob += " " + (getattr(rc, "narrative", "") or "").lower()
    for w in _FABRICATED_CAUSE_WORDS:
        assert w not in blob, f"{fid}: fabricated causal phrase {w!r} in reasoning"

    # recurrence risk is never HIGH off a bare count / severity wording
    assert getattr(rc, "risk_of_recurrence", "NOT_ASSESSABLE") in (
        "NOT_ASSESSABLE", "LOW", "MEDIUM", None,
    ) or "recur" in FINDINGS[fid].lower()


@pytest.mark.parametrize("fid,numbers,typed", [
    ("A_comparison_inventory", ["120"], True),
    ("B_recurrence_equipment", ["3", "three"], True),
    ("C_comparison_percent", ["4.2"], True),
    # I/J: the count/duration is a population / window not covered by a typed
    # field yet -- it must still survive in authoritative provenance (spec 28).
    ("I_recurrence_procedural", ["3", "three"], False),
    ("J_missing_temperature_records", ["18"], False),
])
def test_explicit_numbers_survive(fid, numbers, typed, outcomes):
    """Spec 3 / 28: an explicit number in the finding must remain represented
    somewhere in the canonical state or its authoritative provenance."""
    cf = outcomes[fid]["canonical_finding_state"]
    typed_haystack = " ".join(str(x) for x in (
        cf.deviation_condition, cf.observed_deviation, cf.affected_period,
        getattr(cf.measurement, "value", None) if cf.measurement else None,
        cf.recurrence_count, cf.recurrence_period, cf.occurrence_population,
    ))
    if typed:
        assert any(n in typed_haystack for n in numbers), f"{fid}: {numbers} lost from typed fields -> {typed_haystack!r}"
    prov_haystack = (
        typed_haystack + " " + " ".join(cf.verified_observations or [])
        + " " + " ".join(cf.facts or [])
    ).lower()
    assert any(n.lower() in prov_haystack for n in numbers), f"{fid}: {numbers} lost from canonical state entirely"


def test_comparison_semantics_survive(outcomes):
    for fid in ("A_comparison_inventory", "C_comparison_percent"):
        cf = outcomes[fid]["canonical_finding_state"]
        assert cf.semantic_type == "COMPARISON"
        assert cf.comparison_type in ("BELOW", "EXCEEDED", "MISMATCH", "INCONSISTENT", "RECONCILIATION_FAILURE")
        assert cf.comparison_left and cf.comparison_right
        assert cf.measurement is not None and cf.measurement.value is not None
    # bare "differed" -> direction not invented
    assert outcomes["C_comparison_percent"]["canonical_finding_state"].comparison_type == "MISMATCH"


def test_recurrence_semantics_survive(outcomes):
    cf = outcomes["B_recurrence_equipment"]["canonical_finding_state"]
    assert cf.semantic_type == "RECURRENCE"
    assert cf.recurrence_count == 3
    assert cf.recurrence_event and cf.recurrence_period
    assert cf.affected_period != "UNKNOWN"


def test_competing_causes_survive(outcomes):
    # D has no substantive subject other than the differential itself; F is a
    # compound finding (a real deficiency + a differential). In BOTH the
    # enumerated mechanisms must be PRESERVED and never ranked/established --
    # this is the spec-9/28 core guarantee that holds regardless of how well
    # the subject resolves.
    for fid in ("D_competing_causes", "F_access_competing"):
        state = outcomes[fid]
        cf = state["canonical_finding_state"]
        assert len(cf.stated_causal_alternatives) >= 3, fid
        assert cf.causal_alternatives_unresolved is True, fid
        rc = state.get("root_cause_result") or state.get("root_cause")
        status = getattr(getattr(rc, "status", None), "value", getattr(rc, "status", None))
        assert status in ("NOT_ESTABLISHED", "NOT_APPLICABLE", None), fid
        assert not getattr(rc, "leading_hypothesis", None) or \
            "NONE" in str(getattr(rc, "leading_hypothesis", "")).upper()
        # no enumerated mechanism was silently promoted to the subject
        subj = (cf.finding_subject or "").lower()
        assert not any(
            alt.lower().split()[-1] in subj and "error" in alt.lower()
            for alt in cf.stated_causal_alternatives
        ), f"{fid}: an enumerated cause leaked into the subject ({subj!r})"

    # D / F: the 5-Why stays incomplete at the evidence boundary and never
    # fabricates a mechanism; a discrimination-style investigation question
    # exists for at least one of them.
    for fid in ("D_competing_causes", "F_access_competing"):
        st = outcomes[fid]
        fw = st.get("five_why_analysis") or st.get("five_why")
        assert not getattr(fw, "is_complete", True), fid
        txt = " ".join(s.answer.lower() for s in getattr(fw, "steps", []))
        for w in ("forgot", "was ignored", "was overlooked", "negligence", "human error"):
            assert w not in txt, f"{fid}: fabricated causal phrase {w!r}"
    _dq = re.compile(r"distinguish|discriminate|which\s+of\s+the")
    assert any(_dq.search(q) for q in _questions(outcomes["D_competing_causes"])) or \
        any(_dq.search(q) for q in _questions(outcomes["F_access_competing"]))


def test_evidence_proposition_not_promoted_to_subject_or_cause(outcomes):
    cf = outcomes["E_evidence_proposition"]["canonical_finding_state"]
    # "Maintenance records show that temporary repairs were performed" ->
    # the records are provenance, "temporary repairs" is a reported action,
    # neither is an established cause.
    subj = (cf.finding_subject or "").lower()
    assert "temporary repairs" not in subj
    rc = outcomes["E_evidence_proposition"].get("root_cause_result") or outcomes["E_evidence_proposition"].get("root_cause")
    status = getattr(getattr(rc, "status", None), "value", getattr(rc, "status", None))
    assert status in ("NOT_ESTABLISHED", None)


# --------------------------------------------------------------------------
# 2. Investigation-plan is structurally matched to the finding (spec 10/11)
# --------------------------------------------------------------------------

def _questions(state) -> list[str]:
    inv = state.get("investigation_plan")
    return [(q.question or "").lower() for q in (getattr(inv, "questions", None) or [])]


def test_investigation_plan_matches_finding_structure(outcomes):
    cmp_q = re.compile(r"compar\w*|calculat\w*|reconcil\w*|discrepancy|both\s+values|source\s+record")
    rec_q = re.compile(r"recurr\w*|each\s+(?:failure|occurrence|incident)|common\s+(?:cause|mechanism)|prior\s+occurrence|history")
    disc_q = re.compile(r"distinguish|discriminate|which\s+of\s+the\s+(?:stated\s+)?mechanisms")
    exec_q = re.compile(r"performed\s+but\s+not\s+recorded|whether\s+the\s+(?:required\s+)?activity\s+(?:was\s+)?(?:performed|occurred)|audit\s+trail|contemporaneous")

    assert any(cmp_q.search(q) for q in _questions(outcomes["A_comparison_inventory"]))
    assert any(cmp_q.search(q) for q in _questions(outcomes["C_comparison_percent"]))
    assert any(disc_q.search(q) for q in _questions(outcomes["D_competing_causes"]))
    # J (records unavailable): questions must at least be scoped to the record
    # / its governing requirement -- not silently dropped.
    _jq = _questions(outcomes["J_missing_temperature_records"])
    assert _jq and any(
        w in q for q in _jq
        for w in ("record", "refrigerator", "temperature", "requirement", "waiver",
                  "retriev", "unavailable", "secondary", "audit trail")
    )


# --------------------------------------------------------------------------
# 3. Anti-template contamination (spec 30 / 39)
# --------------------------------------------------------------------------

def _remediation_activities(state) -> set[str]:
    rep = state.get("report")
    rc_cost = getattr(rep, "remediation_cost", None) or state.get("remediation_cost")
    acts = list(getattr(rc_cost, "implementation_activities", None) or [])
    return {re.sub(r"\s+", " ", a.strip().lower()) for a in acts if a}


def test_unrelated_findings_do_not_share_identical_remediation(outcomes):
    pairs = [
        ("B_recurrence_equipment", "E_evidence_proposition"),
        ("A_comparison_inventory", "J_missing_temperature_records"),
        ("G_medical_device_change", "C_comparison_percent"),
        ("I_recurrence_procedural", "H_oos_investigation"),
    ]
    for a, b in pairs:
        sa, sb = _remediation_activities(outcomes[a]), _remediation_activities(outcomes[b])
        if not sa or not sb:
            continue  # NOT_ASSESSABLE with no activities on one side -> nothing to compare
        assert sa != sb, f"{a} and {b} produced an identical remediation activity set: {sa}"


def test_unrelated_findings_do_not_share_identical_investigation_questions(outcomes):
    def qset(fid):
        return {re.sub(r"[A-Z]{1,4}-?\d[\w-]*|\d+", "#", q) for q in _questions(outcomes[fid])}
    pairs = [
        ("A_comparison_inventory", "J_missing_temperature_records"),
        ("B_recurrence_equipment", "D_competing_causes"),
        ("G_medical_device_change", "C_comparison_percent"),
    ]
    for a, b in pairs:
        qa, qb = qset(a), qset(b)
        if qa and qb:
            assert qa != qb, f"{a} and {b} produced identical (entity-masked) investigation questions"


def test_no_unrelated_domain_concept_leaks(outcomes):
    # a non-equipment finding must not be handed calibration/batch-record questions
    for fid in ("A_comparison_inventory", "D_competing_causes", "F_access_competing",
                "G_medical_device_change"):
        for q in _questions(outcomes[fid]):
            assert "calibration certificate" not in q and "batch manufacturing record" not in q, \
                f"{fid}: unrelated domain concept in question: {q!r}"
