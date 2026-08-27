"""LLM semantic-understanding financial engine.

Orchestrates: LLM interpretation -> provider-neutral normalization ->
deterministic structural validation -> materialization into
`FinancialObservation` -> the SAME deterministic calculator used by
`app.financial.engine.analyze_financial_exposure`
(`_build_result_from_observations`), so the semantic path and the
regex-extraction path always share one numeric-authority implementation.

Honest-failure contract (architecture spec sections 18 & 20): this
function does NOT silently substitute a keyword/regex interpretation when
the LLM stage fails. It returns an explicit, number-free
`FinancialAnalysisResult` whose `financial_semantic_status` records what
went wrong. It returns `None` in exactly ONE case -- `NO_EVIDENCE`, where
there is no evidence ledger for the LLM to read and the caller's
deterministic text engine is the only option.
"""

from __future__ import annotations

import logging

from app.financial.engine import _build_result_from_observations
from app.financial.models import (
    ConfirmedFinancialImpact,
    DimensionalConfidence,
    FinancialAnalysisResult,
    FinancialConfidenceLevel,
    FinancialEpistemicStatus,
    FinancialUncertainty,
)
from app.financial.relationship_validator import validate_and_materialize
from app.financial.semantic_models import (
    CalculationTrace,
    SemanticFindingInterpretation,
    SemanticValidationOutcome,
)
from app.models.agent import EvidenceItem
from app.services.semantic_evidence_interpreter import interpret_evidence_semantically

logger = logging.getLogger(__name__)

_RELEVANCE_FROM_STATUS = {
    "LLM_UNAVAILABLE": FinancialEpistemicStatus.FINANCIAL_SEMANTIC_UNAVAILABLE,
    "LLM_INVALID": FinancialEpistemicStatus.FINANCIAL_SEMANTIC_UNAVAILABLE,
    "LLM_INCOMPLETE": FinancialEpistemicStatus.FINANCIAL_SEMANTIC_INCOMPLETE,
}


def _claim_evidence_ids(interpretation: SemanticFindingInterpretation, claim_id: str) -> list[str]:
    for claim in interpretation.claims:
        if claim.claim_id == claim_id:
            return list(claim.source_evidence_ids)
    return []


class SemanticFinancialAnalysisAudit:
    """Non-authoritative audit trail of the semantic path's own reasoning
    -- interpretation, validation outcome -- attached to the result for
    transparency, never consumed by the calculator or the renderer's
    numeric fields."""

    def __init__(
        self,
        interpretation: SemanticFindingInterpretation | None,
        outcome: SemanticValidationOutcome,
    ):
        self.interpretation = interpretation
        self.outcome = outcome


def _honest_failure_result(
    semantic_status: str,
    detail: str,
) -> FinancialAnalysisResult:
    """Build an explicit, number-free result for a non-OK semantic status.
    NO monetary field is populated -- the system fails honestly rather than
    guessing."""
    epistemic = _RELEVANCE_FROM_STATUS.get(semantic_status, FinancialEpistemicStatus.FINANCIAL_SEMANTIC_UNAVAILABLE)
    result = FinancialAnalysisResult(
        status=epistemic,
        confidence=FinancialConfidenceLevel.NOT_ASSESSABLE,
        dimensional_confidence=DimensionalConfidence(rationale=detail),
        assessment_reason=detail,
        reasoning_source="LLM_SEMANTIC",
        financial_semantic_status=semantic_status,
        financial_relevance="NONE",
        uncertainty=FinancialUncertainty(
            # Auditor-facing only -- no engine/provider internals (spec
            # sections 12 & 31). `detail` is retained on the result's
            # internal diagnostic fields, not surfaced here.
            unresolved_factors=[
                "A structurally valid financial quantification could not be produced from the "
                "available evidence."
            ],
            evidence_needed_to_resolve=[
                "Assess any monetary exposure manually from the evidence ledger, recording the "
                "cost factor, amount, currency and evidence status."
            ],
        ),
    )
    return result


def _fill_trace_results(traces: list[CalculationTrace], observations) -> None:
    """Deterministically compute each trace's executor_result from its
    materialized observation and flag any disagreement with the LLM's
    proposed figure. This is arithmetic over the already-validated plan --
    never a re-interpretation."""
    by_id = {o.observation_id: o for o in observations}
    for tr in traces:
        obs = by_id.get(tr.observation_id or "")
        if obs is None:
            continue
        if obs.unit_amount is not None and obs.event_count is not None:
            tr.executor_result = float(obs.unit_amount) * float(obs.event_count)
            tr.formula = f"{obs.event_count:g} x {obs.unit_amount:g}"
        elif obs.amount is not None:
            tr.executor_result = float(obs.amount)
        if (
            tr.executor_result is not None
            and tr.llm_proposed_result is not None
            and abs(tr.executor_result - tr.llm_proposed_result) > max(1.0, abs(tr.executor_result) * 1e-6)
        ):
            tr.disagreement = (
                f"LLM proposed {tr.llm_proposed_result:g}; deterministic executor computed "
                f"{tr.executor_result:g}. The executor value is authoritative."
            )


async def analyze_financial_exposure_semantic(
    finding_text: str,
    evidence_ledger: list[EvidenceItem] | None = None,
    observation_period_months: float | None = None,
    annual_event_frequency: float | None = None,
    frequency_range: tuple[float, float] | None = None,
    client=None,
) -> tuple[FinancialAnalysisResult, SemanticFinancialAnalysisAudit] | None:
    """Run the LLM semantic-understanding path.

    Returns `(result, audit)` in every case EXCEPT `NO_EVIDENCE` (no
    evidence ledger to interpret), where it returns `None` so the caller
    can use its deterministic text engine. A failed LLM stage yields an
    explicit `financial_semantic_status` result with no fabricated numbers
    -- never a silent regex substitution.
    """
    evidence_ledger = evidence_ledger or []

    status, interpretation = await interpret_evidence_semantically(finding_text, evidence_ledger, client=client)

    if status == "NO_EVIDENCE":
        return None
    if status != "OK" or interpretation is None:
        detail = {
            "LLM_UNAVAILABLE": "The LLM financial-interpretation provider was unavailable, so no automated "
            "financial figure was produced. This is reported rather than substituted with a keyword estimate.",
            "LLM_INVALID": "The LLM financial interpretation was not structurally valid, so it was not used. "
            "No automated financial figure was produced.",
        }.get(status, "The LLM financial interpretation could not be completed.")
        return _honest_failure_result(status, detail), SemanticFinancialAnalysisAudit(
            interpretation, SemanticValidationOutcome(semantic_status=status)  # type: ignore[arg-type]
        )

    try:
        observations, outcome = validate_and_materialize(interpretation, evidence_count=len(evidence_ledger))
    except Exception as exc:  # fail-closed: a bug in validation must never fabricate a number
        logger.warning("Semantic financial validation failed unexpectedly (%s).", exc)
        return _honest_failure_result(
            "LLM_INVALID",
            "The financial interpretation could not be validated, so no automated figure was produced.",
        ), SemanticFinancialAnalysisAudit(interpretation, SemanticValidationOutcome(semantic_status="LLM_INVALID"))

    outcome.semantic_status = "OK"

    if not observations:
        # A grounded cost factor can still exist with zero calculable
        # observations (e.g. "24 hours of additional repair labor" with no
        # rate: REWORK_COST is identifiable, nothing to materialize). That
        # positive, LLM-grounded information is preserved -- never
        # discarded, never replaced by a regex pass.
        quant = interpretation.quantification
        if outcome.validated_cost_factor is not None:
            if outcome.rejected:
                blocker = "; ".join(f"{r.calculation_id}: {r.detail}" for r in outcome.rejected)
            else:
                blocker = quant.blocker or (
                    "The evidence establishes the type of financial exposure but does not provide a "
                    "monetary rate, unit price, or total amount to calculate it."
                )
            result = FinancialAnalysisResult(
                status=FinancialEpistemicStatus.COST_FACTOR_IDENTIFIED_NOT_QUANTIFIABLE,
                confidence=FinancialConfidenceLevel.MEDIUM,
                dimensional_confidence=DimensionalConfidence(
                    overall=FinancialConfidenceLevel.MEDIUM,
                    rationale=blocker,
                ),
                confirmed_impact=ConfirmedFinancialImpact(
                    financial_factor=outcome.validated_cost_factor,
                    quantification_status="UNQUANTIFIED",
                    quantification_blocker=blocker,
                    basis=blocker,
                    source_evidence_ids=[
                        eid
                        for cid in interpretation.cost_factor.supporting_claim_ids
                        for eid in _claim_evidence_ids(interpretation, cid)
                    ],
                ),
                assessment_reason=blocker,
                reasoning_source="LLM_SEMANTIC",
                financial_semantic_status="OK",
                financial_relevance=interpretation.financial_relevance or "MATERIAL",
                calculation_traces=list(outcome.traces),
                uncertainty=FinancialUncertainty(
                    unresolved_factors=list(quant.missing_inputs) or [blocker],
                    evidence_needed_to_resolve=list(quant.missing_inputs) or [blocker],
                ),
            )
            return result, SemanticFinancialAnalysisAudit(interpretation, outcome)

        # No grounded factor and nothing materialized. If the LLM itself
        # judged the finding financially relevant, that is an INCOMPLETE
        # interpretation worth surfacing honestly; otherwise there is
        # genuinely no financial content and the caller's deterministic
        # text engine is a reasonable confirmation path.
        if interpretation.financial_relevance in ("MATERIAL", "CONFIRMED"):
            detail = (
                interpretation.quantification.blocker
                or "The evidence was judged financially relevant but the interpretation did not establish "
                "a specific cost factor or a calculable amount."
            )
            result = _honest_failure_result("LLM_INCOMPLETE", detail)
            result.financial_relevance = interpretation.financial_relevance
            return result, SemanticFinancialAnalysisAudit(interpretation, outcome)
        return None

    _fill_trace_results(outcome.traces, observations)

    result = _build_result_from_observations(
        observations,
        observation_period_months=observation_period_months,
        annual_event_frequency=annual_event_frequency,
        frequency_range=frequency_range,
    )
    result.reasoning_source = "LLM_SEMANTIC"
    result.financial_semantic_status = "OK"
    result.financial_relevance = interpretation.financial_relevance or (
        "CONFIRMED" if any(o.verification_status == "VERIFIED" for o in observations) else "MATERIAL"
    )
    result.calculation_traces = list(outcome.traces)

    # The cost factor is the LLM's semantic judgment (structurally
    # validated), NOT a property of evidence verification. The shared
    # observation-based builder only names a factor from VERIFIED
    # exposure; when the grounded semantic factor is more specific than
    # what the builder concluded, surface it -- WITHOUT touching any
    # monetary field or evidence status (spec sections 8, 9, 19: a
    # REPORTED duplicate-payment finding still reads "Cost Factor:
    # DUPLICATE PAYMENT / Evidence: REPORTED").
    if (
        outcome.validated_cost_factor is not None
        and result.confirmed_impact is not None
        and result.confirmed_impact.financial_factor in (None, "NOT_ESTABLISHED")
    ):
        result.confirmed_impact.financial_factor = outcome.validated_cost_factor
    return result, SemanticFinancialAnalysisAudit(interpretation, outcome)
