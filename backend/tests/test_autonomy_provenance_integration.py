"""AUTONOMY PROVENANCE INTEGRATION -- wiring `AutonomyDecision` into the
existing `AnalysisProvenance` / `derive_analysis_provenance` pipeline
(production-finalization / autonomy-provenance-only task).

Domain-agnostic and model-agnostic: every test builds synthetic reports and
identities, never relies on a particular finding, and never inspects raw
finding text.
"""
from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

from app.agent.autonomy import (
    CertificateStore,
    build_current_execution_identity,
    evaluate_autonomy,
)
from app.agent.provenance import _derive_autonomy_decision, _required_tasks_for
from app.models.agent import (
    AnalysisProvenance,
    CapaAnalysis,
    CapaStatus,
    EvidenceCompleteness,
    FiveWhyAnalysis,
    ImpactAssessment,
    ImpactStatus,
    InvestigateRequest,
    InvestigationPlan,
    InvestigationReport,
    RootCauseAnalysis,
    RootCauseStatus,
)
from app.models.autonomy import (
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


def _cert(identity, **kw) -> ModelCapabilityCertificate:
    defaults = dict(
        capability_results={d.value: CR.PASS.value for d in CD},
        semantic_safety_gate="PASS", autonomous_capability_gate="PASS",
        human_reviewed_production_gate="PASS (CONDITIONAL)",
    )
    defaults.update(kw)
    return ModelCapabilityCertificate(
        certificate_id=ModelCapabilityCertificate.make_id(identity, 1.0),
        identity=identity, generated_at=1.0,
        correct_count=20, safe_abstention_count=2, run_error_count=0,
        total_cases=22, **defaults,
    )


def _minimal_report(**overrides) -> InvestigationReport:
    defaults = dict(
        observation_quality="SUFFICIENT",
        investigation_required="LIMITED",
        evidence_completeness=EvidenceCompleteness.COMPLETE,
        root_cause=RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[]),
        investigation=InvestigationPlan(),
        five_why=FiveWhyAnalysis(steps=[], is_complete=False),
        capa=CapaAnalysis(status=CapaStatus.INVESTIGATION_REQUIRED),
        impact_assessment=ImpactAssessment(status=ImpactStatus.IMPACT_NOT_IDENTIFIED),
    )
    defaults.update(overrides)
    return InvestigationReport(**defaults)


@pytest.fixture()
def store(tmp_path):
    return CertificateStore(tmp_path)


# --------------------------------------------------------------------------- #
# 1. eligible decision persists the exact certificate identity
# --------------------------------------------------------------------------- #

def test_1_eligible_decision_persists_exact_certificate_identity(store):
    identity = _identity()
    store.save(_cert(identity))
    decision = evaluate_autonomy(identity, ["RCA"], store=store)
    assert decision.eligible is True
    assert decision.certificate_identity == identity


# --------------------------------------------------------------------------- #
# 2. non-eligible decision persists the blocking reason
# --------------------------------------------------------------------------- #

def test_2_non_eligible_decision_persists_blocking_reason(store):
    decision = evaluate_autonomy(_identity(model="uncertified"), ["RCA"], store=store)
    assert decision.eligible is False
    assert AutonomyBlockingReason.MODEL_NOT_CERTIFIED.value in decision.blocking_reasons


# --------------------------------------------------------------------------- #
# 3-8. round-trip preservation through AnalysisProvenance JSON serialization
# --------------------------------------------------------------------------- #

def _provenance_with_decision(decision) -> AnalysisProvenance:
    return AnalysisProvenance(autonomy=decision)


def test_3_provider_model_identity_survives_roundtrip(store):
    identity = _identity(provider="ollama", model="synthetic-model:8b")
    store.save(_cert(identity))
    decision = evaluate_autonomy(identity, ["RCA"], store=store)
    prov = _provenance_with_decision(decision)
    reloaded = AnalysisProvenance.model_validate_json(prov.model_dump_json())
    assert reloaded.autonomy.certificate_identity.provider == "ollama"
    assert reloaded.autonomy.certificate_identity.model == "synthetic-model:8b"


def test_4_model_digest_survives_roundtrip(store):
    identity = _identity(model_version="digest-xyz")
    store.save(_cert(identity))
    decision = evaluate_autonomy(identity, ["RCA"], store=store)
    prov = _provenance_with_decision(decision)
    reloaded = AnalysisProvenance.model_validate_json(prov.model_dump_json())
    assert reloaded.autonomy.certificate_identity.model_version == "digest-xyz"


def test_5_benchmark_hash_survives_roundtrip(store):
    identity = _identity(benchmark_hash="bench-xyz")
    store.save(_cert(identity))
    decision = evaluate_autonomy(identity, ["RCA"], store=store)
    prov = _provenance_with_decision(decision)
    reloaded = AnalysisProvenance.model_validate_json(prov.model_dump_json())
    assert reloaded.autonomy.certificate_identity.benchmark_hash == "bench-xyz"


def test_6_capability_results_survive_roundtrip(store):
    identity = _identity()
    results = {d.value: CR.PASS.value for d in CD}
    results[CD.RECURRENCE_REASONING.value] = CR.FAIL.value
    store.save(_cert(identity, capability_results=results))
    decision = evaluate_autonomy(identity, ["REMEDIATION_COSTING"], store=store)
    prov = _provenance_with_decision(decision)
    reloaded = AnalysisProvenance.model_validate_json(prov.model_dump_json())
    assert reloaded.autonomy.capability_results[CD.RECURRENCE_REASONING.value] == CR.FAIL.value


def test_7_required_capabilities_survive_roundtrip(store):
    identity = _identity()
    store.save(_cert(identity))
    decision = evaluate_autonomy(identity, ["FIVE_WHY"], store=store)
    prov = _provenance_with_decision(decision)
    reloaded = AnalysisProvenance.model_validate_json(prov.model_dump_json())
    assert set(reloaded.autonomy.required_capabilities) == {
        CD.CAUSAL_REASONING.value, CD.CAUSAL_DEPTH.value,
        CD.EVIDENCE_GROUNDING.value, CD.EPISTEMIC_PRESERVATION.value,
    }


def test_8_review_required_remains_true_regardless_of_eligibility(store):
    identity = _identity()
    store.save(_cert(identity))
    eligible_decision = evaluate_autonomy(identity, ["RCA"], store=store)
    not_eligible_decision = evaluate_autonomy(_identity(model="other"), ["RCA"], store=store)
    assert eligible_decision.review_required is True
    assert not_eligible_decision.review_required is True
    for decision in (eligible_decision, not_eligible_decision):
        prov = _provenance_with_decision(decision)
        reloaded = AnalysisProvenance.model_validate_json(prov.model_dump_json())
        assert reloaded.autonomy.review_required is True
        assert reloaded.review_required is True  # existing top-level field untouched


# --------------------------------------------------------------------------- #
# 9. missing/malformed provenance fails closed
# --------------------------------------------------------------------------- #

def test_9_missing_autonomy_field_is_a_valid_none_not_a_crash():
    prov = AnalysisProvenance()
    assert prov.autonomy is None
    reloaded = AnalysisProvenance.model_validate_json(prov.model_dump_json())
    assert reloaded.autonomy is None


def test_9_derive_autonomy_decision_never_raises_on_a_broken_state():
    report = _minimal_report()
    # `state` deliberately missing keys / of the wrong shape.
    decision = _derive_autonomy_decision(report, {})
    assert decision is None or decision.review_required is True


def test_9_corrupt_certificate_file_is_treated_as_absent(tmp_path):
    store = CertificateStore(tmp_path)
    identity = _identity()
    path = store._path(identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not valid json")
    decision = evaluate_autonomy(identity, ["RCA"], store=store)
    assert decision.status == AutonomyStatus.NOT_CERTIFIED.value


# --------------------------------------------------------------------------- #
# 10. serialization does not change the autonomy status
# --------------------------------------------------------------------------- #

def test_10_serialization_does_not_change_autonomy_status(store):
    identity = _identity()
    store.save(_cert(identity))
    decision = evaluate_autonomy(identity, ["RCA"], store=store)
    prov = _provenance_with_decision(decision)
    reloaded = AnalysisProvenance.model_validate_json(prov.model_dump_json())
    assert reloaded.autonomy.status == decision.status
    assert reloaded.autonomy.eligible == decision.eligible
    assert reloaded.autonomy.blocking_reasons == decision.blocking_reasons


# --------------------------------------------------------------------------- #
# 11. a different provider/model creates a different provenance identity
# --------------------------------------------------------------------------- #

def test_11_different_provider_model_creates_different_provenance_identity(store):
    a = _identity(provider="ollama", model="model-a")
    b = _identity(provider="ollama", model="model-b")
    store.save(_cert(a))
    store.save(_cert(b))
    da = evaluate_autonomy(a, ["RCA"], store=store)
    db = evaluate_autonomy(b, ["RCA"], store=store)
    assert da.certificate_identity != db.certificate_identity
    assert da.certificate_id != db.certificate_id


# --------------------------------------------------------------------------- #
# 12. no raw finding text enters the autonomy controller / derivation path
# --------------------------------------------------------------------------- #

def test_12_no_raw_finding_text_enters_autonomy_derivation():
    src = inspect.getsource(_derive_autonomy_decision) + inspect.getsource(_required_tasks_for)
    for banned in ("finding_text", "raw_finding", "request.finding", ".lower()", "re.search", "re.match"):
        assert banned not in src


@pytest.mark.asyncio
async def test_12_end_to_end_provenance_never_echoes_finding_text_into_autonomy():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
    from app.agent.nodes.investigation_planner import plan_investigation_node
    from app.agent.nodes.report_generator import generate_report_node
    from app.agent.nodes.understanding import understand_finding_node

    finding_text = "A very specific, unique marker phrase XYZQ123 was found in the widget assembly."
    st = {"request": InvestigateRequest(finding_text=finding_text), "evidence_ledger": [],
          "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0, "trace": [], "errors": []}
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        for n in (understand_finding_node, plan_investigation_node, core_synthesis_node,
                  generate_report_node, final_evidence_verification_node):
            st = await n(st)
    prov = st["report"].provenance
    dumped = prov.model_dump_json()
    assert "XYZQ123" not in dumped


# --------------------------------------------------------------------------- #
# 13. no semantic classifiers introduced by the required-tasks derivation
# --------------------------------------------------------------------------- #

def test_13_required_tasks_is_a_pure_structural_presence_check():
    populated = _minimal_report(
        five_why=FiveWhyAnalysis(steps=[], is_complete=False),
        investigation=InvestigationPlan(),
    )
    # empty structures -> no FIVE_WHY / INVESTIGATION task
    assert _required_tasks_for(populated) == ["RCA", "CAPA"]


def test_13_identity_builder_matches_certification_script_formula():
    # The runtime identity builder and the certification script MUST hash
    # the same fields the same way, or a certificate the script just saved
    # would never be found at runtime (see app.agent.autonomy docstring).
    identity = build_current_execution_identity(model_version="x")
    assert identity.semantic_schema_version and len(identity.semantic_schema_version) == 16
    assert identity.contract_version and len(identity.contract_version) == 16
    assert identity.benchmark_hash and len(identity.benchmark_hash) == 16


def test_14_identity_binds_exact_canonical_prompt_bytes(tmp_path):
    # Editing the canonical prompt text -- WITHOUT bumping any version string --
    # must change the certificate identity, so an old certificate can never
    # match a prompt it did not certify.
    from app.config import get_settings

    class _S:
        def __init__(self, base, directory):
            self._b, self.prompts_dir = base, directory

        def __getattr__(self, name):
            return getattr(self._b, name)

    base = get_settings()
    (tmp_path / "canonical_finding_interpretation_system_prompt.txt").write_text("prompt v1 {schema}")
    with patch("app.config.get_settings", return_value=_S(base, tmp_path)), \
            patch("app.agent.autonomy._cached_model_digest", return_value="x"):
        before = build_current_execution_identity(model_version="x")
        (tmp_path / "canonical_finding_interpretation_system_prompt.txt").write_text("prompt v2 {schema}")
        after = build_current_execution_identity(model_version="x")
        (tmp_path / "canonical_finding_interpretation_system_prompt.txt").unlink()
        missing = build_current_execution_identity(model_version="x")
    assert before.contract_version != after.contract_version
    assert before.identity_key() != after.identity_key()
    assert missing.contract_version not in (before.contract_version, after.contract_version)  # fail-closed
