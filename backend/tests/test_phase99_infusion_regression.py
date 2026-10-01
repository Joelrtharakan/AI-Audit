"""Phase 9.9 targeted regression (generated report with duplicated gaps, a
malformed 5-Why, a leaked status, wrong evidence labels, mislabelled cost scope).

Every test proves a GENERAL invariant across several unrelated domains; none
recognises the original finding's wording. No live LLM.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent.nodes.plan_investigation_fallback import _plan_from_canonical_structure
from app.agent.nodes.report_generator import _finalize_report_consistency
from app.models.agent import (
    EvidenceItem, EvidenceStatus, ImpactAssessment, ImpactStatus, InvestigationPlan, InvestigationQuestion,
)
from app.remediation.calculator import assemble_estimate
from app.remediation.engine import _aggregate_scope_status, derive_component_scope_roles
from app.remediation.semantic_models import ImplementationActivity, RemediationCostComponent
from app.services import canonical_consistency_review as ccr
from app.services.canonical_context_validator import validate_canonical_context
from app.services.canonical_semantic_models import (
    CanonicalFindingContext, SemImpact, SemInvestigationStep, SemRemediationAction,
)
from app.services.evidence_ids import build_registry
from app.services.semantic_subject import format_deviation_why_question

_ROOT = Path(__file__).resolve().parent.parent.parent
_JS = _ROOT / "frontend" / "assets" / "js" / "lqms_ai.js"


def _ledger(*pairs):
    return [EvidenceItem(claim=c, status=EvidenceStatus.VERIFIED, source=f"S{i}") for i, c in enumerate(pairs)]


# =========================================================================== #
# 1. investigation gap identity (no duplication across sections)
# =========================================================================== #

_DOMAIN_STEPS = [
    ("mfg", "the governing torque specification", "Approved torque specification"),
    ("health", "the approved sterilisation schedule", "Approved sterilisation schedule"),
    ("software", "the access-review policy that applies", "Access review policy"),
    ("food", "the validated cold-chain limit", "Validated cold-chain limit"),
]


def _ctx_with_steps(unknowns_and_evidence):
    return CanonicalFindingContext(
        finding_subject="the subject", observed_condition="a condition",
        investigation_plan=[SemInvestigationStep(unknown=u, evidence_that_would_resolve=e)
                            for u, e in unknowns_and_evidence],
        information_gaps=[u for u, _ in unknowns_and_evidence],
    )


@pytest.mark.parametrize("_d,unknown,evidence", _DOMAIN_STEPS)
def test_every_gap_gets_a_stable_unique_id(_d, unknown, evidence):
    out = validate_canonical_context(_ctx_with_steps([(unknown, evidence), (unknown + " v2", evidence + " v2")]),
                                     _ledger("x"), "f")
    ids = [s.gap_id for s in out.investigation_plan]
    assert all(ids) and len(set(ids)) == len(ids)


def test_duplicate_gap_ids_are_reidentified_but_reported():
    ctx = _ctx_with_steps([("a", "ea"), ("b", "eb")])
    for s in ctx.investigation_plan:
        s.gap_id = "G1"
    out = validate_canonical_context(ctx, _ledger("x"), "f")
    assert len({s.gap_id for s in out.investigation_plan}) == 2
    assert "GAP_ID_DUPLICATE:G1" in out.consistency_issues and out.review_required


def test_two_gaps_resolved_by_identical_evidence_are_flagged_not_merged_by_wording():
    out = validate_canonical_context(_ctx_with_steps([("gap one", "Approved   Specification"),
                                                      ("gap two", "approved specification")]), _ledger("x"), "f")
    assert any(i.startswith("GAP_DUPLICATE_EVIDENCE") for i in out.consistency_issues)
    assert len(out.investigation_plan) == 2  # nothing dropped: the model owns merging


def test_gaps_list_diverging_from_plan_is_reported_not_rewritten():
    ctx = _ctx_with_steps([("alpha unknown", "ev")])
    ctx.information_gaps = ["a differently worded copy"]
    out = validate_canonical_context(ctx, _ledger("x"), "f")
    assert "GAPS_LIST_DIVERGES_FROM_PLAN" in out.consistency_issues
    assert out.information_gaps == ["a differently worded copy"]  # preserved


@pytest.mark.parametrize("_d,unknown,evidence", _DOMAIN_STEPS)
def test_plan_questions_carry_gap_identity_and_gaps_are_not_repeated_as_evidence(_d, unknown, evidence):
    ctx = validate_canonical_context(_ctx_with_steps([(unknown, evidence)]), _ledger("x"), "f")
    _hyps, plan = _plan_from_canonical_structure(ctx, "the subject")
    assert [q.gap_id for q in plan.questions] == [ctx.investigation_plan[0].gap_id]
    # the gap description is rendered ONCE (as the question), never appended to the evidence list
    assert unknown not in plan.evidence_to_collect
    assert evidence in plan.evidence_to_collect


def test_unlinked_evidence_is_by_identity_not_position():
    q = InvestigationQuestion(question="q?", evidence="Approved Spec", evidence_required=None)
    plan = InvestigationPlan(questions=[q], evidence_to_collect=["approved   spec", "Calibration record", "calibration record"])
    assert plan.unlinked_evidence == ["Calibration record"]
    assert "unlinked_evidence" in plan.model_dump()


def test_frontend_uses_server_side_unlinked_evidence():
    js = _JS.read_text()
    assert "inv.unlinked_evidence" in js and "invEvidence.slice(invQuestions.length).join" not in js


# =========================================================================== #
# 2. 5-Why grammar: realised from grammatical SHAPE, never from finding words
# =========================================================================== #

# (subject, condition, expected FORM). Wording is irrelevant to the assertions.
_FORMS = [
    # active past verb phrase with a direct object -> clause frame, never "was/were <verb> <object>"
    ("infusion pumps", "missed scheduled preventive maintenance", "CLAUSE"),
    ("the forklift fleet", "skipped the weekly inspection", "CLAUSE"),
    ("night-shift staff", "omitted the second verification", "CLAUSE"),
    ("the supplier", "delivered the wrong batch", "CLAUSE"),
    ("access reviews", "overlooked terminated accounts", "CLAUSE"),
    # copular states -> was/were with number agreement
    ("calibration records", "incomplete", "COPULAR_PLURAL"),
    ("the filter", "overdue by 5 weeks", "COPULAR_SINGULAR"),
    ("shipments", "damaged during transit", "COPULAR_PLURAL"),
    ("the audit", "not completed", "COPULAR_SINGULAR"),
    ("reagent lots", "expired on 3 March", "COPULAR_PLURAL"),
    ("the checklist", "was missed", "COPULAR_SINGULAR"),
    # finite auxiliary already in the condition -> inversion
    ("the log", "did not record the entry", "DID"),
    ("the pumps", "had not been serviced", "HAD"),
    ("the valve", "could have failed", "MODAL"),
]


@pytest.mark.parametrize("subject,cond,form", _FORMS)
def test_question_shape_follows_grammatical_form(subject, cond, form):
    q = format_deviation_why_question(subject, cond)
    assert q.startswith("Why ") and q.endswith("?")
    assert "  " not in q and "_" not in q
    if form == "CLAUSE":
        assert q.startswith("Why is it that ")
        assert not re.match(r"^Why (?:was|were) ", q)
    elif form == "COPULAR_PLURAL":
        assert q.startswith("Why were ")
    elif form == "COPULAR_SINGULAR":
        assert q.startswith("Why was ")
    elif form == "DID":
        assert q.startswith("Why did ") and " did not " not in q
    elif form == "HAD":
        assert q.startswith("Why had ")
    elif form == "MODAL":
        assert q.startswith("Why could ")


@pytest.mark.parametrize("subject,cond,_f", _FORMS)
def test_question_never_duplicates_the_subject_or_the_aux(subject, cond, _f):
    q = format_deviation_why_question(subject, cond).lower()
    core = re.sub(r"^the ", "", subject.lower())
    assert q.count(core) == 1
    assert not re.search(r"\b(was|were|did|had)\s+(was|were|did|had)\b", q)


# =========================================================================== #
# 3. no internal enum may reach natural-language rendering
# =========================================================================== #

_ENUMS = ["UNKNOWN", "NOT_ESTABLISHED", "VERIFIED", "REPORTED_UNVERIFIED", "BELIEF", "ESTIMATED", "PROPOSED",
          "AUTHORIZED", "COMPLETED", "REQUIRES_ASSESSMENT", "NOT_ASSESSABLE", "SOME_FUTURE_STATUS_VALUE"]


def _humanizer_outputs():
    src = _JS.read_text()
    a = src.index("var STATUS_LABELS")
    b = src.index("function escapeHtml")
    js = src[a:b] + "\nconsole.log(JSON.stringify(" + json.dumps(_ENUMS) + ".map(statusLabel)));"
    return json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_status_humanizer_never_emits_a_raw_enum():
    for raw, shown in zip(_ENUMS, _humanizer_outputs()):
        assert shown and shown != raw and "_" not in shown and not shown.isupper(), (raw, shown)


def test_five_why_status_is_a_separate_labelled_block():
    js = _JS.read_text()
    w = js[js.index("Requires verification"):][:900]
    assert "Evidence status: " in w and "statusLabel(stStatus)" in w and "escapeHtml(stStatus)" not in w


def test_confidence_values_render_as_separate_labelled_rows():
    js = _JS.read_text()
    w = js[js.index("Observation Confidence: ") - 300:][:1400]
    assert w.count("statusLabel(") >= 3 and "Obs: " not in w
    # the underlying values are passed through untouched
    assert "report.observation_confidence" in w and "report.root_cause_confidence" in w and "report.overall_confidence" in w


# =========================================================================== #
# 4. evidence provenance: own labels, never remapped
# =========================================================================== #

def _items(*claims):
    return [SimpleNamespace(claim=c) for c in claims]


def test_direct_label_wins_over_positional_guess():
    reg = build_registry(_items("C2: second claim stated first", "C1: first claim stated second"))
    assert reg.resolve("C2") == "E0" and reg.resolve("C1") == "E1"   # positional guess would invert these
    assert reg.display("E0") == "C2" and reg.display("E1") == "C1"


def test_cost_driver_shows_the_evidence_own_label_not_the_internal_index():
    reg = build_registry(_items("C1: scope", "C2: labour 2.5 h at 950", "C3: quarterly contract 8400"))
    assert [reg.display(reg.resolve(x)) for x in ("C2", "e1", "E2")] == ["C2", "C2", "C3"]


def test_ambiguous_label_resolves_to_nothing():
    reg = build_registry(_items("C1: a", "C1: b"))
    assert reg.resolve("C1") is None


@pytest.mark.parametrize("ref", ["C9", "E99", "", None, "zzz", "R1"])
def test_unresolvable_reference_is_never_invented(ref):
    reg = build_registry(_items("C1: a", "C2: b"))
    assert reg.resolve(ref) is None


def test_positional_alias_never_claims_a_label_another_item_states():
    reg = build_registry(_items("R7: a", "C1: b"))
    assert reg.resolve("C1") == "E1"            # stated label
    assert reg.resolve("C2") is None            # explicit namespace -> no positional guessing


def test_finding_text_labels_align_only_when_counts_match():
    led = _items("alpha", "beta", "gamma")
    assert build_registry(led, "C1: x. C2: y. C3: z.").resolve("C3") == "E2"
    assert build_registry(led, "C1: x. C2: y.").resolve("C2") is not None  # falls to positional only (no labels stated)
    assert build_registry(_items("alpha", "beta"), "C1: x. C5: y. C9: z.").display("E0") == "E0"


def test_frontend_maps_evidence_through_the_registry_labels():
    js = _JS.read_text()
    assert js.count("evidence_labels") >= 3


# =========================================================================== #
# 5-7. cost scope / recurring / labels
# =========================================================================== #

def _comp(cid, amount, **kw):
    base = dict(component_id=cid, description="x", unit_cost=amount, unit_cost_basis="ESTIMATED", currency="INR",
                amount_type="COMPONENT")
    base.update(kw)
    return RemediationCostComponent(**base)


def test_arithmetic_is_correct_and_independent_of_scope():
    labour = _comp("L", 950, amount_type="PER_HOUR", quantity=7 * 2.5, quantity_unit="hour", quantity_basis="EVIDENCED")
    kit = _comp("K", 4200, amount_type="PER_UNIT", quantity=7, quantity_unit="kit", quantity_basis="EVIDENCED")
    for established in (True, False):
        est = assemble_estimate([labour, kit], [], [], scope_established=established)
        assert est.one_time_cost == 16625 + 29400 == 46025


def test_required_wording_only_when_scope_established():
    comps = [_comp("A", 100), _comp("B", 50)]
    assert "required implementation" in assemble_estimate(comps, [], [], scope_established=True).estimation_method
    unscoped = assemble_estimate(comps, [], [], scope_established=False).estimation_method
    assert "required" not in unscoped and "priced activity" in unscoped


@pytest.mark.parametrize("roles,expected", [
    ([], "NOT_ESTABLISHED"),
    (["ESTABLISHED_REMEDIATION", "ESTABLISHED_REMEDIATION"], "ESTABLISHED"),
    (["ESTABLISHED_REMEDIATION", "NOT_ESTABLISHED"], "NOT_ESTABLISHED"),
    (["PROPOSED_REMEDIATION"], "PROPOSED"),
    (["PREVENTIVE", "MONITORING"], "PREVENTIVE_OR_MONITORING"),
    (["OPTIONAL"], "OPTIONAL"),
    (["ESTABLISHED_REMEDIATION", "OPTIONAL"], "MIXED"),
])
def test_scope_rollup_is_a_pure_function_of_declared_roles(roles, expected):
    assert _aggregate_scope_status(roles) == expected


def test_scope_derived_only_from_llm_declared_activity_flags_never_from_amounts():
    acts = [ImplementationActivity(activity_id="A1", description="d", disposition="IMMEDIATE_CORRECTION"),
            ImplementationActivity(activity_id="A2", description="d", disposition="CONDITIONAL_SYSTEMIC"),
            ImplementationActivity(activity_id="A3", description="d", disposition="IMMEDIATE_CORRECTION", depends_on_root_cause=True),
            ImplementationActivity(activity_id="A4", description="d", disposition="EFFECTIVENESS_CHECK")]
    comps = [_comp("c1", 999999, activity_ids=["A1"]), _comp("c2", 1, activity_ids=["A2"]),
             _comp("c3", 5, activity_ids=["A3"]), _comp("c4", 5, activity_ids=["A4"]),
             _comp("c5", 5), _comp("c6", 5, activity_ids=["A2"], scope_role="OPTIONAL")]
    derive_component_scope_roles(comps, acts)
    assert [c.scope_role for c in comps] == ["ESTABLISHED_REMEDIATION", "PROPOSED_REMEDIATION",
                                             "PROPOSED_REMEDIATION", "MONITORING", "NOT_ESTABLISHED", "OPTIONAL"]


def test_recurring_amount_is_kept_separate_with_its_own_scope_and_no_invented_horizon():
    one = _comp("O", 46025)
    rec = _comp("R", 8400, recurrence="RECURRING", recurring_period="quarter", unit_cost_basis="VERIFIED",
                scope_role="NOT_ESTABLISHED")
    est = assemble_estimate([one, rec], [], [])
    assert est.one_time_cost == 46025 and est.recurring_cost == 8400 and est.recurring_period == "quarter"
    assert est.recurring_horizon_total is None
    recurring_result = [r for r in est.component_results if r.recurrence == "RECURRING"][0]
    assert recurring_result.scope_role == "NOT_ESTABLISHED"  # relationship preserved, not assumed


def test_frontend_headline_depends_on_semantic_scope_and_facets_are_separate():
    js = _JS.read_text()
    assert 'rc.scope_status === "ESTABLISHED"' in js and "Priced Activity Estimate" in js
    for facet in ("Arithmetic: ", "Scope: ", "Confidence: ", "Classification: ", "Evidence Basis: "):
        assert facet in js
    assert "Recurring priced activity:" in js and "Relationship to remediation:" in js


# =========================================================================== #
# 8-9. canonical action status and impact are consumed
# =========================================================================== #

def _report_stub(impact_status, rc_status="NOT_ESTABLISHED", inv_required="YES"):
    return SimpleNamespace(
        impact_assessment=ImpactAssessment(status=impact_status), remediation_cost=None,
        root_cause=SimpleNamespace(status=rc_status, leading_hypothesis=None),
        semantic_consistency_issues=[], investigation_required=inv_required,
        immediate_action_items=[], canonical_impact=None)


def _sc(**kw):
    base = dict(finding_subject="s", observed_condition="c")
    base.update(kw)
    return CanonicalFindingContext(**base)


def test_report_consumes_canonical_action_status_after_provenance_validation():
    ctx = _sc(remediation_activities=[
        SemRemediationAction(action_id="A1", activity="do x", disposition="IMMEDIATE_CORRECTION", action_status="COMPLETED"),
        SemRemediationAction(action_id="A2", activity="do y", disposition="CONTAINMENT", action_status="AUTHORIZED",
                             action_status_evidence_ids=["E0"]),
        SemRemediationAction(action_id="A3", activity="later", disposition="CONDITIONAL_SYSTEMIC")])
    validated = validate_canonical_context(ctx, _ledger("approval record"), "f")
    rep = _report_stub(ImpactStatus.IMPACT_REQUIRES_ASSESSMENT)
    _finalize_report_consistency(rep, None, validated)
    got = {i.action_id: i.action_status for i in rep.immediate_action_items}
    assert got == {"A1": "PROPOSED", "A2": "AUTHORIZED"}   # uncited COMPLETED demoted; conditional not "immediate"
    assert rep.immediate_action_items[1].evidence_ids == ["E0"]


@pytest.mark.parametrize("canon,expected", [("POTENTIAL", ImpactStatus.IMPACT_POSSIBLE),
                                            ("NOT_ESTABLISHED", ImpactStatus.IMPACT_REQUIRES_ASSESSMENT),
                                            ("REQUIRES_ASSESSMENT", ImpactStatus.IMPACT_REQUIRES_ASSESSMENT)])
def test_report_impact_never_exceeds_canonical_impact_and_cap_is_recorded(canon, expected):
    rep = _report_stub(ImpactStatus.IMPACT_VERIFIED)
    _finalize_report_consistency(rep, None, _sc(impact=SemImpact(status=canon)))
    assert rep.impact_assessment.status == expected
    assert any(i.startswith("IMPACT_STATUS_EXCEEDS_CANONICAL") for i in rep.semantic_consistency_issues)
    assert rep.canonical_impact.status == canon


def test_observed_canonical_impact_is_allowed_to_stand():
    rep = _report_stub(ImpactStatus.IMPACT_VERIFIED)
    ctx = validate_canonical_context(_sc(impact=SemImpact(status="OBSERVED", evidence_ids=["E0"])), _ledger("e"), "f")
    _finalize_report_consistency(rep, None, ctx)
    assert rep.impact_assessment.status == ImpactStatus.IMPACT_VERIFIED and not rep.semantic_consistency_issues


# =========================================================================== #
# 10. object / activity / process separation
# =========================================================================== #

@pytest.mark.parametrize("copy_field", ["affected_activity", "finding_subject", "observed_condition"])
def test_process_copied_from_another_field_is_not_established_and_is_reported(copy_field):
    kw = dict(affected_process="Scheduled   Task", affected_activity="other", finding_subject="subj", observed_condition="cond")
    kw[copy_field] = "scheduled task"
    out = validate_canonical_context(_sc(**kw), _ledger("e"), "f")
    assert out.affected_process is None
    assert any(i.startswith("PROCESS_COPIES_") for i in out.consistency_issues)


def test_distinct_process_and_activity_are_both_preserved():
    out = validate_canonical_context(_sc(affected_process="equipment maintenance programme",
                                         affected_activity="a scheduled service visit",
                                         affected_requirement="service interval"), _ledger("e"), "f")
    assert (out.affected_process, out.affected_activity, out.affected_requirement) == (
        "equipment maintenance programme", "a scheduled service visit", "service interval")
    assert not out.consistency_issues


# =========================================================================== #
# 12. cross-section consistency under root cause NOT_ESTABLISHED
# =========================================================================== #

def test_investigation_cannot_be_closed_while_root_cause_not_established():
    rep = _report_stub(ImpactStatus.IMPACT_REQUIRES_ASSESSMENT, inv_required="NO")
    _finalize_report_consistency(rep, None, None)
    assert rep.investigation_required == "YES"
    assert "INVESTIGATION_CLOSED_WITH_ROOT_CAUSE_NOT_ESTABLISHED" in rep.semantic_consistency_issues


def test_corrective_action_and_leading_hypothesis_flagged_without_established_cause():
    ctx = _sc(leading_hypothesis_id="H1", root_cause_status="NOT_ESTABLISHED",
              remediation_activities=[SemRemediationAction(action_id="A1", activity="a", disposition="CORRECTIVE_ACTION")])
    codes = ccr.review_canonical_consistency(ctx, 1)
    assert "LEADING_HYPOTHESIS_WITHOUT_ESTABLISHED_CAUSE" in codes
    assert "CORRECTIVE_ACTION_WITHOUT_ESTABLISHED_CAUSE:A1" in codes


# =========================================================================== #
# 1b. a valid unknown is never deleted by a deterministic wording veto
# =========================================================================== #

@pytest.mark.parametrize("unknown", [
    "the reason for identical invoices", "the mechanism behind the missed reading",
    "the basis for the schedule interval", "the cause of the delay",
])
def test_gap_and_plan_step_survive_validation_whatever_causal_noun_they_contain(unknown):
    out = validate_canonical_context(_ctx_with_steps([(unknown, "records that would settle it")]), _ledger("x"), "f")
    assert out.information_gaps == [unknown]
    assert [s.unknown for s in out.investigation_plan] == [unknown]


# =========================================================================== #
# provenance cap on finding-level epistemic status
# =========================================================================== #

def _ledger_with(status):
    return [EvidenceItem(claim="c", status=status, source="s")]


@pytest.mark.parametrize("status,ceiling", [(EvidenceStatus.REPORTED, "REPORTED"), (EvidenceStatus.BELIEF, "BELIEF"),
                                            (EvidenceStatus.UNVERIFIED, "UNKNOWN")])
def test_verified_status_is_capped_by_the_evidence_held_and_recorded(status, ceiling):
    out = validate_canonical_context(_sc(epistemic_status="VERIFIED"), _ledger_with(status), "f")
    assert out.epistemic_status == ceiling
    assert "EPISTEMIC_STATUS_EXCEEDS_EVIDENCE" in out.consistency_issues and out.review_required


def test_verified_status_stands_with_a_verified_item_or_no_ledger():
    assert validate_canonical_context(_sc(epistemic_status="VERIFIED"), _ledger_with(EvidenceStatus.VERIFIED), "f").epistemic_status == "VERIFIED"
    assert validate_canonical_context(_sc(epistemic_status="VERIFIED"), [], "f").epistemic_status == "VERIFIED"
    mixed = _ledger_with(EvidenceStatus.REPORTED) + _ledger_with(EvidenceStatus.VERIFIED)
    assert validate_canonical_context(_sc(epistemic_status="VERIFIED"), mixed, "f").epistemic_status == "VERIFIED"
