"""Model- and provider-independent AUTONOMOUS CAPABILITY FRAMEWORK.

Tests the governance layer added in `app.models.autonomy` / `app.agent.autonomy`
around the EXISTING architecture. Domain-agnostic and model-agnostic by
construction: every test operates on synthetic `ExecutionConfigIdentity` /
`ModelCapabilityCertificate` objects, never on finding text, and never on a
literal model name beyond "does identity equality behave correctly".

Covers the required test matrix (spec §29) items A-J, O, P, Q, R and the
metamorphic properties (spec §30) 6, 7, 8, 9 that this layer can express
without a live LLM.
"""
from __future__ import annotations

import pytest

from app.agent.autonomy import CertificateStore, evaluate_autonomy, required_capabilities_for
from app.models.autonomy import (
    TASK_CAPABILITY_REQUIREMENTS,
    AutonomyBlockingReason,
    AutonomyStatus,
    CapabilityDimension as CD,
    CapabilityResult as CR,
    ExecutionConfigIdentity,
    ModelCapabilityCertificate,
)


def _identity(**overrides) -> ExecutionConfigIdentity:
    base = dict(
        provider="ollama", model="synthetic-model:8b", model_version="digestA",
        semantic_schema_version="schemaA", contract_version="contractA",
        inference_config_hash="cfgA", benchmark_hash="benchA",
    )
    base.update(overrides)
    return ExecutionConfigIdentity(**base)


def _cert(identity: ExecutionConfigIdentity, *, capability_results: dict | None = None,
          safety="PASS", autonomous="PASS", sme=0, caught=0) -> ModelCapabilityCertificate:
    return ModelCapabilityCertificate(
        certificate_id=ModelCapabilityCertificate.make_id(identity, 1.0),
        identity=identity, generated_at=1.0,
        correct_count=20, safe_abstention_count=2, caught_error_count=caught,
        silent_material_error_count=sme, run_error_count=0, total_cases=22,
        capability_results=capability_results or {},
        semantic_safety_gate=safety, autonomous_capability_gate=autonomous,
        human_reviewed_production_gate="PASS (CONDITIONAL)" if safety == "PASS" else "FAIL",
    )


def _all_pass_results() -> dict:
    return {dim.value: CR.PASS.value for dim in CD}


@pytest.fixture()
def store(tmp_path):
    return CertificateStore(tmp_path)


# --------------------------------------------------------------------------- #
# A. certified capable model -> autonomy eligible
# --------------------------------------------------------------------------- #

def test_a_certified_capable_model_is_eligible_for_its_certified_task(store):
    identity = _identity()
    store.save(_cert(identity, capability_results=_all_pass_results()))
    decision = evaluate_autonomy(identity, ["RCA"], store=store)
    assert decision.eligible is True
    assert decision.status == AutonomyStatus.ELIGIBLE.value
    assert decision.blocking_reasons == []
    assert decision.review_required is True  # autonomy != auditor approval


# --------------------------------------------------------------------------- #
# B. uncertified model -> human review, fail-closed default
# --------------------------------------------------------------------------- #

def test_b_uncertified_model_defaults_to_not_certified(store):
    decision = evaluate_autonomy(_identity(model="never-certified:1b"), ["RCA"], store=store)
    assert decision.eligible is False
    assert decision.status == AutonomyStatus.NOT_CERTIFIED.value
    assert AutonomyBlockingReason.MODEL_NOT_CERTIFIED.value in decision.blocking_reasons
    assert decision.review_required is True


# --------------------------------------------------------------------------- #
# C. certified model with an unsupported TASK capability -> human review for
#    that task only (task-level eligibility, spec §9)
# --------------------------------------------------------------------------- #

def test_c_certified_model_ineligible_for_task_missing_a_required_capability(store):
    identity = _identity()
    results = _all_pass_results()
    results[CD.RECURRENCE_REASONING.value] = CR.FAIL.value
    store.save(_cert(identity, capability_results=results))

    rca_decision = evaluate_autonomy(identity, ["RCA"], store=store)
    assert rca_decision.eligible is True  # RCA does not require RECURRENCE_REASONING

    pricing_decision = evaluate_autonomy(identity, ["REMEDIATION_COSTING"], store=store)
    assert pricing_decision.eligible is False
    assert AutonomyBlockingReason.REQUIRED_CAPABILITY_NOT_CERTIFIED.value in pricing_decision.blocking_reasons


# --------------------------------------------------------------------------- #
# D. silent material error on the certificate -> autonomy prohibited globally
# --------------------------------------------------------------------------- #

def test_d_silent_material_error_blocks_autonomy_even_with_passing_capabilities(store):
    identity = _identity()
    store.save(_cert(identity, capability_results=_all_pass_results(), safety="FAIL", sme=1))
    decision = evaluate_autonomy(identity, ["RCA"], store=store)
    assert decision.eligible is False
    assert AutonomyBlockingReason.SEMANTIC_SAFETY_FAILED.value in decision.blocking_reasons


# --------------------------------------------------------------------------- #
# E. CAUGHT_ERROR on a required capability -> autonomy prohibited for tasks
#    needing it (spec §32 -- not automatically treated as harmless)
# --------------------------------------------------------------------------- #

def test_e_caught_error_capability_blocks_only_tasks_that_require_it(store):
    identity = _identity()
    results = _all_pass_results()
    results[CD.CAUSAL_REASONING.value] = CR.FAIL.value  # e.g. a CAUGHT_ERROR case
    store.save(_cert(identity, capability_results=results, caught=1))

    rca = evaluate_autonomy(identity, ["RCA"], store=store)
    assert rca.eligible is False
    five_why = evaluate_autonomy(identity, ["FIVE_WHY"], store=store)
    assert five_why.eligible is False
    remediation = evaluate_autonomy(identity, ["REMEDIATION_COSTING"], store=store)
    assert remediation.eligible is True  # does not require CAUSAL_REASONING


# --------------------------------------------------------------------------- #
# F. SAFE_ABSTENTION is not penalized as a capability failure
# --------------------------------------------------------------------------- #

def test_f_safe_abstention_capability_can_still_pass(store):
    identity = _identity()
    # A certificate whose only non-CORRECT cases were SAFE_ABSTENTION still
    # reports PASS for the dimensions they cover -- this module trusts
    # whatever compute_capability_results already decided; it does not
    # second-guess PASS/FAIL, it only reads them.
    store.save(_cert(identity, capability_results=_all_pass_results()))
    decision = evaluate_autonomy(identity, ["INVESTIGATION"], store=store)
    assert decision.eligible is True


# --------------------------------------------------------------------------- #
# G. provider change -> previous certificate not reused
# --------------------------------------------------------------------------- #

def test_g_provider_change_does_not_reuse_certificate(store):
    identity = _identity(provider="ollama")
    store.save(_cert(identity, capability_results=_all_pass_results()))
    other = _identity(provider="groq")
    decision = evaluate_autonomy(other, ["RCA"], store=store)
    assert decision.status == AutonomyStatus.NOT_CERTIFIED.value


# --------------------------------------------------------------------------- #
# H. model version change -> previous certificate invalidated
# --------------------------------------------------------------------------- #

def test_h_model_version_change_invalidates_certificate(store):
    identity = _identity(model_version="digestA")
    store.save(_cert(identity, capability_results=_all_pass_results()))
    upgraded = _identity(model_version="digestB")
    decision = evaluate_autonomy(upgraded, ["RCA"], store=store)
    assert decision.status == AutonomyStatus.NOT_CERTIFIED.value


# --------------------------------------------------------------------------- #
# I. semantic schema change -> certificate invalidated
# --------------------------------------------------------------------------- #

def test_i_semantic_schema_change_invalidates_certificate(store):
    identity = _identity(semantic_schema_version="schemaA")
    store.save(_cert(identity, capability_results=_all_pass_results()))
    changed = _identity(semantic_schema_version="schemaB")
    decision = evaluate_autonomy(changed, ["RCA"], store=store)
    assert decision.status == AutonomyStatus.NOT_CERTIFIED.value


# --------------------------------------------------------------------------- #
# J. prompt/contract version change -> certificate invalidated when material
# --------------------------------------------------------------------------- #

def test_j_contract_version_change_invalidates_certificate(store):
    identity = _identity(contract_version="contractA")
    store.save(_cert(identity, capability_results=_all_pass_results()))
    changed = _identity(contract_version="contractB")
    decision = evaluate_autonomy(changed, ["RCA"], store=store)
    assert decision.status == AutonomyStatus.NOT_CERTIFIED.value


def test_j_benchmark_hash_change_invalidates_certificate(store):
    identity = _identity(benchmark_hash="benchA")
    store.save(_cert(identity, capability_results=_all_pass_results()))
    changed = _identity(benchmark_hash="benchB")
    decision = evaluate_autonomy(changed, ["RCA"], store=store)
    assert decision.status == AutonomyStatus.NOT_CERTIFIED.value


# --------------------------------------------------------------------------- #
# runtime signals (independent of certificate) -- spec §11, §12
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("signal,reason", [
    ("evidence_sufficient", AutonomyBlockingReason.EVIDENCE_INSUFFICIENT.value),
    ("causal_consistent", AutonomyBlockingReason.CAUSAL_EVIDENCE_INSUFFICIENT.value),
    ("epistemic_consistent", AutonomyBlockingReason.EPISTEMIC_CONSISTENCY_FAILED.value),
    ("structural_safety_ok", AutonomyBlockingReason.STRUCTURED_OUTPUT_INVALID.value),
    ("provenance_ok", AutonomyBlockingReason.STRUCTURED_OUTPUT_INVALID.value),
])
def test_runtime_signal_failure_blocks_autonomy_even_when_certified(store, signal, reason):
    identity = _identity()
    store.save(_cert(identity, capability_results=_all_pass_results()))
    kwargs = {"evidence_sufficient": True, "causal_consistent": True,
              "epistemic_consistent": True, "structural_safety_ok": True, "provenance_ok": True}
    kwargs[signal] = False
    decision = evaluate_autonomy(identity, ["RCA"], store=store, **kwargs)
    assert decision.eligible is False
    assert reason in decision.blocking_reasons


def test_confidence_is_not_a_controller_input():
    # spec §12: the controller signature has no confidence/score parameter at
    # all -- a model's self-reported confidence cannot influence autonomy
    # because there is nowhere to pass it in.
    import inspect
    sig = inspect.signature(evaluate_autonomy)
    assert not any("confidence" in p.lower() or "score" in p.lower() for p in sig.parameters)


# --------------------------------------------------------------------------- #
# O. serialization round trip -> no autonomy-state or epistemic drift
# --------------------------------------------------------------------------- #

def test_o_certificate_serialization_roundtrip_preserves_identity_and_gates(store):
    identity = _identity()
    original = _cert(identity, capability_results=_all_pass_results())
    store.save(original)
    reloaded = store.load(identity)
    assert reloaded is not None
    assert reloaded.identity == original.identity
    assert reloaded.semantic_safety_gate == original.semantic_safety_gate
    assert reloaded.autonomous_capability_gate == original.autonomous_capability_gate
    assert reloaded.capability_results == original.capability_results
    assert reloaded.certificate_id == original.certificate_id


# --------------------------------------------------------------------------- #
# P. provenance round trip -> certificate identity preserved
# --------------------------------------------------------------------------- #

def test_p_decision_carries_exact_certificate_identity(store):
    identity = _identity()
    cert = _cert(identity, capability_results=_all_pass_results())
    store.save(cert)
    decision = evaluate_autonomy(identity, ["RCA"], store=store)
    assert decision.certificate_identity == identity
    assert decision.certificate_id == cert.certificate_id


# --------------------------------------------------------------------------- #
# Q. task-level capability failure blocks only the affected task
# R. global safety failure blocks EVERY task
# --------------------------------------------------------------------------- #

def test_q_task_level_failure_is_isolated(store):
    identity = _identity()
    results = _all_pass_results()
    results[CD.COMPARISON_REASONING.value] = CR.FAIL.value
    store.save(_cert(identity, capability_results=results))
    # no existing task requires COMPARISON_REASONING directly -- every
    # defined task remains eligible.
    for task in TASK_CAPABILITY_REQUIREMENTS:
        assert evaluate_autonomy(identity, [task], store=store).eligible is True


def test_r_global_safety_failure_blocks_every_task(store):
    identity = _identity()
    store.save(_cert(identity, capability_results=_all_pass_results(), safety="FAIL", sme=1))
    for task in TASK_CAPABILITY_REQUIREMENTS:
        decision = evaluate_autonomy(identity, [task], store=store)
        assert decision.eligible is False


# --------------------------------------------------------------------------- #
# metamorphic property 6/7: removing/changing certification removes eligibility
# --------------------------------------------------------------------------- #

def test_property_6_removing_certification_removes_eligibility(store):
    identity = _identity()
    store.save(_cert(identity, capability_results=_all_pass_results()))
    assert evaluate_autonomy(identity, ["RCA"], store=store).eligible is True
    (store.directory / f"{identity.identity_key()}.json").unlink()
    assert evaluate_autonomy(identity, ["RCA"], store=store).eligible is False


@pytest.mark.parametrize("field", ["provider", "model", "model_version", "semantic_schema_version",
                                    "contract_version", "inference_config_hash", "benchmark_hash"])
def test_property_7_any_identity_field_change_loses_eligibility(store, field):
    identity = _identity()
    store.save(_cert(identity, capability_results=_all_pass_results()))
    changed = _identity(**{field: "CHANGED_VALUE"})
    assert evaluate_autonomy(changed, ["RCA"], store=store).eligible is False


# --------------------------------------------------------------------------- #
# property 8: identity/decision structural stability across a JSON round trip
# --------------------------------------------------------------------------- #

def test_property_8_decision_is_stable_across_serialization(store):
    identity = _identity()
    store.save(_cert(identity, capability_results=_all_pass_results()))
    d1 = evaluate_autonomy(identity, ["RCA"], store=store)
    d2 = evaluate_autonomy(ExecutionConfigIdentity.model_validate(identity.model_dump()), ["RCA"], store=store)
    assert d1.eligible == d2.eligible
    assert d1.status == d2.status
    assert d1.blocking_reasons == d2.blocking_reasons


# --------------------------------------------------------------------------- #
# structural-only: no finding text anywhere in the controller's surface
# --------------------------------------------------------------------------- #

def test_controller_never_touches_finding_text():
    import inspect
    src = inspect.getsource(evaluate_autonomy)
    for banned in ("finding_text", "raw_finding", ".lower()", "re.search", "re.match"):
        assert banned not in src


def test_required_capabilities_are_task_table_lookups_only():
    assert required_capabilities_for(["RCA"]) == TASK_CAPABILITY_REQUIREMENTS["RCA"]
    assert required_capabilities_for(["UNKNOWN_TASK"]) == []
    assert required_capabilities_for(["RCA", "FIVE_WHY"])[:1] == [CD.CAUSAL_REASONING]
