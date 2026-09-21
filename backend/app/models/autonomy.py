"""Model- and provider-independent AUTONOMOUS CAPABILITY FRAMEWORK.

This module adds a GOVERNANCE layer around the existing architecture -- it
does not reinterpret finding meaning, does not add semantic classifiers, and
does not change the existing human-review posture. `ReviewState.required`
remains hard-pinned to True everywhere in this codebase (see
`app/models/agent.py`); the autonomy decision computed here is an
independently-tracked, additional technical determination ("would this exact
certified execution configuration be capable enough to run this task without
a human"), never a bypass of the auditor's final review.

Core principle (unchanged): the LLM owns semantic interpretation, the
canonical contract owns semantic representation, evidence owns epistemic
support, deterministic code owns structural safety/arithmetic/provenance,
capability certification measures model competence, and this autonomy
controller decides -- from that objective evidence alone -- whether the
CURRENTLY CONFIGURED model may be treated as capable for a given task. It
never inspects raw finding text.
"""
from __future__ import annotations

import hashlib
import time
from enum import Enum

from pydantic import BaseModel, Field


class CapabilityDimension(str, Enum):
    """Independent capability axes (spec §3). A model is never scored on one
    undifferentiated quality number -- each axis is certified separately so a
    model can be capable for some tasks and not others (spec §9)."""

    OBSERVATION_SEMANTICS = "OBSERVATION_SEMANTICS"
    CAUSAL_REASONING = "CAUSAL_REASONING"
    CAUSAL_DEPTH = "CAUSAL_DEPTH"
    EVIDENCE_GROUNDING = "EVIDENCE_GROUNDING"
    EPISTEMIC_PRESERVATION = "EPISTEMIC_PRESERVATION"
    COMPARISON_REASONING = "COMPARISON_REASONING"
    INVESTIGATION_REASONING = "INVESTIGATION_REASONING"
    FIVE_WHY_REASONING = "FIVE_WHY_REASONING"
    REMEDIATION_REASONING = "REMEDIATION_REASONING"
    PRICING_SEMANTICS = "PRICING_SEMANTICS"
    QUANTITY_UNIT_REASONING = "QUANTITY_UNIT_REASONING"
    RECURRENCE_REASONING = "RECURRENCE_REASONING"
    COST_CALCULATION_INTERPRETATION = "COST_CALCULATION_INTERPRETATION"
    CAPA_REASONING = "CAPA_REASONING"
    IMPACT_SEMANTICS = "IMPACT_SEMANTICS"
    SERIALIZATION_CONTRACT = "SERIALIZATION_CONTRACT"
    FAIL_CLOSED_BEHAVIOR = "FAIL_CLOSED_BEHAVIOR"


class CapabilityResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_CERTIFIED = "NOT_CERTIFIED"


class AutonomyStatus(str, Enum):
    NOT_CERTIFIED = "NOT_CERTIFIED"
    CERTIFIED = "CERTIFIED"
    ELIGIBLE = "ELIGIBLE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class AutonomyBlockingReason(str, Enum):
    """Machine-readable diagnostics only (spec §20) -- never rendered into
    auditor-facing natural-language prose."""

    MODEL_NOT_CERTIFIED = "MODEL_NOT_CERTIFIED"
    REQUIRED_CAPABILITY_NOT_CERTIFIED = "REQUIRED_CAPABILITY_NOT_CERTIFIED"
    SEMANTIC_SAFETY_FAILED = "SEMANTIC_SAFETY_FAILED"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    CAUSAL_EVIDENCE_INSUFFICIENT = "CAUSAL_EVIDENCE_INSUFFICIENT"
    EPISTEMIC_CONSISTENCY_FAILED = "EPISTEMIC_CONSISTENCY_FAILED"
    CERTIFICATE_EXPIRED = "CERTIFICATE_EXPIRED"
    CERTIFICATE_INVALIDATED = "CERTIFICATE_INVALIDATED"
    MODEL_EXECUTION_ERROR = "MODEL_EXECUTION_ERROR"
    STRUCTURED_OUTPUT_INVALID = "STRUCTURED_OUTPUT_INVALID"


class ExecutionConfigIdentity(BaseModel):
    """The EXACT execution configuration a certificate is bound to (spec §2,
    §14, §26). Identity is deterministic field equality -- no fuzzy matching,
    no "close enough" model comparison. Any field changing means a DIFFERENT
    identity, which means no certificate is found for it (fail-closed by
    construction -- see `CertificateStore.load`)."""

    model_config = {"frozen": True}

    provider: str
    model: str
    model_version: str = ""
    semantic_schema_version: str
    contract_version: str
    inference_config_hash: str
    benchmark_hash: str

    def identity_key(self) -> str:
        parts = [
            self.provider, self.model, self.model_version, self.semantic_schema_version,
            self.contract_version, self.inference_config_hash, self.benchmark_hash,
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


TASK_CAPABILITY_REQUIREMENTS: dict[str, list[CapabilityDimension]] = {
    # Fixed structural mapping from the existing canonical workflow graph
    # (spec §8) -- NEVER derived by inspecting finding text.
    "RCA": [
        CapabilityDimension.CAUSAL_REASONING,
        CapabilityDimension.EVIDENCE_GROUNDING,
        CapabilityDimension.EPISTEMIC_PRESERVATION,
    ],
    "FIVE_WHY": [
        CapabilityDimension.CAUSAL_REASONING,
        CapabilityDimension.CAUSAL_DEPTH,
        CapabilityDimension.EVIDENCE_GROUNDING,
        CapabilityDimension.EPISTEMIC_PRESERVATION,
    ],
    "INVESTIGATION": [
        CapabilityDimension.INVESTIGATION_REASONING,
        CapabilityDimension.EVIDENCE_GROUNDING,
        CapabilityDimension.EPISTEMIC_PRESERVATION,
    ],
    "REMEDIATION_COSTING": [
        CapabilityDimension.PRICING_SEMANTICS,
        CapabilityDimension.QUANTITY_UNIT_REASONING,
        CapabilityDimension.RECURRENCE_REASONING,
        CapabilityDimension.EVIDENCE_GROUNDING,
    ],
    "CAPA": [
        CapabilityDimension.CAUSAL_REASONING,
        CapabilityDimension.EVIDENCE_GROUNDING,
        CapabilityDimension.EPISTEMIC_PRESERVATION,
    ],
}


class ModelCapabilityCertificate(BaseModel):
    """A certificate is a fact about one EXACT execution configuration,
    generated by running the existing (unmodified) held-out benchmark once.
    It is not permanent trust: a different identity simply has no
    certificate, and a stale one is superseded the next time certification
    runs for that identity (spec §13, §25, §26)."""

    certificate_id: str
    identity: ExecutionConfigIdentity
    generated_at: float = Field(default_factory=time.time)

    correct_count: int = 0
    safe_abstention_count: int = 0
    caught_error_count: int = 0
    silent_material_error_count: int = 0
    run_error_count: int = 0
    total_cases: int = 0

    capability_results: dict[str, str] = Field(default_factory=dict)  # CapabilityDimension.value -> CapabilityResult.value

    semantic_safety_gate: str = "FAIL"            # PASS | FAIL
    autonomous_capability_gate: str = "FAIL"       # PASS | FAIL
    human_reviewed_production_gate: str = "FAIL"   # "PASS (CONDITIONAL)" | FAIL

    @staticmethod
    def make_id(identity: ExecutionConfigIdentity, generated_at: float) -> str:
        return hashlib.sha256(f"{identity.identity_key()}|{generated_at}".encode()).hexdigest()[:24]

    def capability(self, dim: CapabilityDimension) -> CapabilityResult:
        return CapabilityResult(self.capability_results.get(dim.value, CapabilityResult.NOT_CERTIFIED.value))


class AutonomyDecision(BaseModel):
    """Structured output of `app.agent.autonomy.evaluate_autonomy` (spec
    §33). `review_required` is ALWAYS True in this codebase regardless of
    `eligible` -- autonomy is a technical capability determination, never
    auditor approval (spec §27); the existing `ReviewState.required` contract
    is untouched by this framework."""

    eligible: bool = False
    status: str = AutonomyStatus.NOT_CERTIFIED.value
    required_capabilities: list[str] = Field(default_factory=list)
    capability_results: dict[str, str] = Field(default_factory=dict)
    safety_results: dict[str, bool] = Field(default_factory=dict)
    blocking_reasons: list[str] = Field(default_factory=list)
    certificate_identity: ExecutionConfigIdentity | None = None
    certificate_id: str | None = None
    review_required: bool = True
    generated_at: float = Field(default_factory=time.time)
