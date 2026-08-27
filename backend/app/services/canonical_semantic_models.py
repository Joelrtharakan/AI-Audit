"""Canonical semantic finding context: the schema for the ONE LLM
interpretation every downstream module (financial engine, investigation
planner, Five-Why, risk/impact) should reason from, instead of each module
independently re-deriving its own interpretation of the raw finding text.

Extends the financial semantic layer built in the previous pass
(`app.financial.semantic_models.SemanticFindingInterpretation`) rather than
duplicating it -- the financial claims/relationships/calculation proposals
ARE part of this canonical context, not a separate system.

This module defines DATA ONLY. It performs no interpretation, no
validation, and no arithmetic:
  - Interpretation: `app.services.canonical_finding_interpreter`
  - Validation:      `app.services.canonical_context_validator`
  - Arithmetic:       unchanged -- `app.financial.calculator` via
                       `app.financial.relationship_validator`
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.financial.semantic_models import EvidenceStatusStr, SemanticFindingInterpretation

# What kind of thing a piece of extracted meaning IS -- the single most
# important distinction this schema exists to make explicit, so a STATE
# word ("active", "valid", "in force") can never be mistaken for the
# ENTITY it describes, and a CONSEQUENCE/FINANCIAL_METRIC can never be
# mistaken for a CAUSE.
SemanticKind = Literal[
    "ENTITY", "STATE", "EVENT", "CONSEQUENCE", "FINANCIAL_METRIC",
    "HISTORICAL_CONTEXT", "REMEDIATION", "RECOVERY", "CAUSE", "HYPOTHESIS",
]


class CanonicalEntity(BaseModel):
    """One real-world thing the finding concerns (a contract, a line, a
    system, a role) with its STATE tracked as a separate field -- never
    folded into the entity name, and never itself treated as an entity."""

    entity_id: str
    name: str
    kind: SemanticKind = "ENTITY"
    state: str | None = None
    source_evidence_ids: list[str] = Field(default_factory=list)


class CausalClaim(BaseModel):
    """A statement that MAY assert a causal relationship. `is_causal`
    defaults to False -- it may only be True when the evidence text itself
    explicitly asserts causation (e.g. "X caused Y"), never merely because
    two facts co-occur or because one is a financial consequence of the
    other. A financial/historical/recovery/remediation claim is never, by
    itself, evidence of causation."""

    claim_id: str
    statement: str
    is_causal: bool = False
    cause_ref: str | None = None  # entity_id or claim_id
    effect_ref: str | None = None  # entity_id or claim_id
    source_evidence_ids: list[str] = Field(default_factory=list)
    evidence_status: EvidenceStatusStr = "UNVERIFIED"


class EvidenceBoundary(BaseModel):
    """An explicit statement of what the evidence does NOT establish --
    the LLM is expected to name these, not merely omit unsupported
    content silently, so the auditor sees exactly where reasoning must
    stop."""

    description: str
    related_claim_ids: list[str] = Field(default_factory=list)


class CanonicalFindingContext(BaseModel):
    """The canonical semantic interpretation of one finding + its evidence
    ledger. `financial` reuses the existing, unchanged financial semantic
    schema/pipeline from the previous pass -- this context ADDS the
    cross-cutting (non-financial-specific) understanding that financial
    analysis alone never needed: what the actual deviation is, what
    entities/states exist, what is and is not causal, and whether a
    previous CAPA is actually referenced."""

    primary_deviation: str | None = None
    primary_deviation_claim_id: str | None = None
    primary_deviation_confidence: Literal["HIGH", "MEDIUM", "LOW", "NOT_ESTABLISHED"] = "NOT_ESTABLISHED"

    entities: list[CanonicalEntity] = Field(default_factory=list)
    causal_claims: list[CausalClaim] = Field(default_factory=list)

    # Explicit boolean, never inferred from recurrence/historical/repeated
    # wording alone -- see canonical_context_validator.py, which
    # independently cross-checks this against the existing deterministic
    # recurrence_guard.detect_recurrence().has_previous_capa_reference
    # signal and forces it False unless BOTH agree.
    explicit_previous_capa_reference: bool = False
    previous_capa_evidence_ids: list[str] = Field(default_factory=list)

    evidence_boundaries: list[EvidenceBoundary] = Field(default_factory=list)
    unresolved_ambiguities: list[str] = Field(default_factory=list)

    # The unchanged financial semantic layer from the previous pass.
    financial: SemanticFindingInterpretation = Field(default_factory=SemanticFindingInterpretation)


class SemanticDisagreement(BaseModel):
    """One point of disagreement between the existing deterministic
    pipeline's own interpretation and the canonical LLM interpretation --
    recorded for shadow-mode comparison, per the "deterministic result
    stays authoritative until sufficient live validation" requirement.
    Never used to alter any authoritative output by itself."""

    field: str
    deterministic_value: str | None
    canonical_value: str | None
    disagreement_type: Literal[
        "AFFECTED_OBJECT_MISMATCH", "DEVIATION_MISMATCH",
        "PREVIOUS_CAPA_MISMATCH", "POPULATION_MISMATCH", "OTHER",
    ]
    evidence_ids: list[str] = Field(default_factory=list)
    downstream_consequence: str = ""
