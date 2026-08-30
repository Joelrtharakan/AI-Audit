"""Consolidated Node: core_synthesis

Replaces serial rca -> impact -> capa -> ca_draft LLM calls with a single
high-reasoning, structured JSON call. Executes RCA, 5-Why, Impact, CAPA, and
CA Draft simultaneously in ONE LLM round-trip.

Failure path (never skips straight to deterministic synthesis on a mere
output-budget ceiling -- see `_classify_failure` and `_parse_causal_fields`):

    Ollama primary call
      -> parse JSON / schema validate
      -> ACCEPT if valid (regardless of how close to the token ceiling it got)
      -> else: compact JSON-first recovery call (causal fields only)
      -> ACCEPT recovery if valid (analysis_mode stays "LLM")
      -> else: deterministic evidence-grounded synthesis (analysis_mode = "DETERMINISTIC")
"""

from __future__ import annotations

import json
import re
import logging
from typing import Any

from app.agent.analytical_validator import (
    hypothesis_confidence,
    leading_hypothesis_confidence,
    leading_hypothesis_display,
    leading_hypothesis_status,
)
from app.agent.causal_guard import (
    MechanismInfo,
    answer_asserts_verified_but_is_reported,
    classify_mixed_evidence_answer,
    hypothesis_attacks_statement_credibility,
    hypothesis_contradicts_mechanism,
    hypothesis_contradicts_verified_completion,
    hypothesis_discrimination_cites_wrong_id,
    hypothesis_overclaims_human_error,
    is_circular_why_answer,
    is_evidence_gap_not_hypothesis,
    is_generic_non_analysis_filler,
    is_reporting_why_question,
    mechanism_already_names_generic_hypothesis,
    question_reopens_mechanism,
    repeats_previous_why_answer,
    restates_observation,
)
from app.agent.grounding_guard import (
    build_source_text,
    claims_unsupported_effectiveness,
    clean_structured_leak,
    is_placeholder_leak,
    mentions_unsupported_domain,
    ungrounded_entities,
)
from app.agent.permissions import build_ca_draft
from app.agent.state import AgentState
from app.config import get_settings
from app.models.agent import (
    AgentTraceStep,
    CADraft,
    CandidateHypothesis,
    CanonicalFindingState,
    CapaAnalysis,
    CapaStatus,
    ContributingFactor,
    CoreSynthesisOutput,
    EvidenceStatus,
    FiveWhyAnalysis,
    FiveWhyStep,
    ImpactAssessment,
    ImpactStatus,
    InvestigationPlan,
    InvestigationQuestion,
    RootCauseAnalysis,
    RootCauseStatus,
    SupportLevel,
)
from app.services.llm_client import LLMError, LLMNetworkError, LLMTimeoutError, get_llm_client
from app.services.llm_json import parse_llm_json
from app.services.taxonomy import coerce_category

logger = logging.getLogger(__name__)


def _assign_claim_ids(evidence_ledger: list) -> list[tuple[str, Any]]:
    """ONE canonical claim-ID assignment (C1, C2, ... in evidence-ledger
    order), reused for both the primary and recovery prompts and for
    validating the LLM's supporting_claim_ids/contradicting_claim_ids
    against. IDs are positional in the ORIGINAL (untrimmed) ledger, so "C3"
    means the same claim whether it appears in the primary call's full
    ledger or the recovery call's trimmed subset -- never re-indexed per
    call, which would let the same ID silently mean two different claims
    across the two prompts."""
    return [(f"C{i + 1}", e) for i, e in enumerate(evidence_ledger)]


def parse_core_synthesis_output(raw: str) -> tuple[dict, CoreSynthesisOutput]:
    """Single authoritative parser and validator for core synthesis output.
    Used identically for primary and recovery responses.
    """
    parsed_dict = parse_llm_json(raw)
    try:
        validated = CoreSynthesisOutput.model_validate(parsed_dict)
    except Exception as exc:
        logger.debug("Core synthesis schema validation error: %s (payload keys: %s)", exc, list(parsed_dict.keys()) if isinstance(parsed_dict, dict) else type(parsed_dict))
        raise exc
    return parsed_dict, validated


def _classify_failure(exc: Exception | None, ollama_meta: dict) -> str:
    """Map a core_synthesis call failure to one of the specific categories
    from the performance-logging contract, instead of collapsing everything
    into two coarse buckets.

    Crucially: OUTPUT_TRUNCATED is only ever returned when Ollama itself
    reports `done_reason == "length"` (surfaced here as
    `ollama_meta["hit_output_limit"]`) AND parsing/validation actually
    failed. generated_tokens == max_output_tokens alone is never sufficient
    proof of truncation -- a model can legitimately fill the entire output
    budget and still return complete, valid JSON, in which case no exception
    reaches this function at all.
    """
    from pydantic import ValidationError as _PydanticValidationError
    import json

    if isinstance(exc, LLMTimeoutError):
        return "TIMEOUT"
    if isinstance(exc, LLMNetworkError):
        return "PROVIDER_FAILURE"
    if isinstance(exc, _PydanticValidationError):
        return "SCHEMA_VALIDATION_FAILURE"
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return "JSON_PARSE_ERROR"
    if isinstance(exc, LLMError):
        return "PROVIDER_FAILURE"
    if ollama_meta.get("hit_output_limit"):
        return "OUTPUT_TRUNCATED"
    return "INVALID_JSON"


def _failure_metric_suffix(failure_type: str) -> str:
    """Maps the fine-grained _classify_failure() categories onto the three
    llm_metrics failure buckets (Phase 6: count each event exactly once,
    in the right bucket) -- TIMEOUT is its own bucket; JSON/schema/output
    problems are "invalid_json" (something WAS returned but couldn't be
    used); PROVIDER_FAILURE (network/HTTP/empty-completion) is
    "other_failure" (the call itself didn't complete normally, distinct
    from a parsing problem)."""
    if failure_type == "TIMEOUT":
        return "timeout"
    if failure_type == "PROVIDER_FAILURE":
        return "other_failure"
    return "invalid_json"


_MAX_LLM_HYPOTHESES = 3


def _parse_causal_fields(
    parsed: dict,
    mechanism: MechanismInfo,
    evidence_ledger: list,
    source_text: str,
    observed_deviation: str,
    finding_text: str,
    trace: list,
    has_unresolved_conflict: bool = False,
    claim_ids: list[tuple[str, Any]] | None = None,
    canonical_subject: str | None = None,
    canonical: CanonicalFindingState | None = None,
    semantic_context: Any = None,
) -> tuple[RootCauseAnalysis, FiveWhyAnalysis, list[ContributingFactor]]:
    """Parse+guard root_cause / five_why / contributing_factors from a
    core_synthesis-shaped JSON object.

    This is the SINGLE canonical parser for these three fields -- used both
    for the primary full-schema response and for the compact recovery
    response (Section 4/19: one canonical representation, not two competing
    implementations of the same guards). All the causal-grounding guards
    (contradiction detection, domain-leakage, evidence-gap-not-hypothesis,
    etc.) apply identically regardless of which call produced the JSON.
    """
    # -----------------------------------------------------------------
    # 1. Root cause + candidate hypotheses
    # -----------------------------------------------------------------
    raw_rc = parsed.get("root_cause", {})
    from app.services.status_normalizer import normalize_root_cause_status
    rc_status = normalize_root_cause_status(raw_rc.get("status"))

    rc_category = coerce_category(raw_rc.get("category"))
    if rc_status == RootCauseStatus.NOT_ESTABLISHED:
        rc_category = "TO_BE_CONFIRMED"

    from app.services.status_normalizer import normalize_hypothesis_status
    verified_facts = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.VERIFIED]
    valid_claim_ids = {cid for cid, _ in (claim_ids or [])}
    claim_text_by_id = {cid: e.claim for cid, e in (claim_ids or [])}
    _raw_hyp_count = len(raw_rc.get("candidate_hypotheses", []))
    # Phase 7 Section 5: the set of hypothesis ids the LLM actually proposed
    # in THIS response — a `deepens_hypothesis_id` referencing anything
    # outside this set is a dangling/hallucinated reference and must be
    # dropped, never trusted (Section 26: never fake the feature).
    _raw_hyp_ids = {
        str(ch.get("id")) for ch in raw_rc.get("candidate_hypotheses", [])
        if isinstance(ch, dict) and ch.get("id")
    }
    cand_hypotheses = []
    for ch in raw_rc.get("candidate_hypotheses", []):
        if isinstance(ch, dict):
            rank_raw = str(ch.get("relevance_rank", "HIGH")).upper()
            rank = rank_raw if rank_raw in ("HIGH", "MEDIUM", "LOW") else "HIGH"
            # Single authoritative status-normalization boundary (shared with
            # root_cause.status above) -- maps LLM synonyms like "CONFIRMED"/
            # "REJECTED"/"TIED" onto their correct canonical status instead of
            # collapsing every unrecognized value to a blunt "POSSIBLE"
            # default, while still never raising on an invalid enum.
            hyp_status = normalize_hypothesis_status(ch.get("status"))
            statement = clean_structured_leak(ch.get("statement", ""))
            # HUMAN-ERROR OVERCLAIMING (structural, not finding-specific):
            # a hypothesis that stops at "human oversight/error" without
            # also asking what process/control allowed the error is
            # premature -- never delete it (human execution error can be
            # a genuine hypothesis), just don't let it outrank a
            # systemic explanation that hasn't been given the same
            # benefit of the doubt.
            if hypothesis_overclaims_human_error(statement) and rank == "HIGH":
                rank = "MEDIUM"
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: hypothesis {ch.get('id', 'H')} attributes the deviation to human "
                    "error/oversight without a process/control framing — relevance downgraded to avoid "
                    "prematurely leading with an individual-blame explanation"
                ))
            if hyp_status == "REFUTED":
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — directly contradicted by evidence"
                ))
                from app.services import llm_metrics as _llm_metrics
                _llm_metrics.record_validation_rejection(reason="unsupported_causation", node="core_synthesis")
                continue
            # CONTRADICTION-AWARE FILTER (structural, not finding-specific):
            # a hypothesis claiming "performed but not recorded" cannot
            # coexist with an established mechanism confirming the
            # activity was not performed at all — regardless of what
            # status the LLM assigned it.
            if hypothesis_contradicts_mechanism(statement, mechanism):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — contradicts the "
                    f"established mechanism ({mechanism.statement!r})"
                ))
                from app.services import llm_metrics as _llm_metrics
                _llm_metrics.record_validation_rejection(reason="unsupported_causation", node="core_synthesis")
                continue
            # REDUNDANCY FILTER: a hedged restatement of the already-established
            # mechanism is not a new hypothesis about its CAUSE.
            if mechanism_already_names_generic_hypothesis(statement, mechanism):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — restates the already-"
                    "established mechanism as an open possibility instead of reasoning about its cause"
                ))
                from app.services import llm_metrics as _llm_metrics
                _llm_metrics.record_validation_rejection(reason="invalid_hypothesis", node="core_synthesis")
                continue
            # VERIFIED-COMPLETION CONTRADICTION (structural, not tied to any
            # topic word): a hypothesis proposing a deficiency/absence of
            # something a VERIFIED fact already confirms WAS done (e.g.
            # "training deficiency" when the ledger confirms training was
            # completed) is contradicted, not merely unlikely.
            if hypothesis_contradicts_verified_completion(statement, verified_facts):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — contradicts a VERIFIED "
                    "fact that the thing it claims is deficient was actually completed"
                ))
                from app.services import llm_metrics as _llm_metrics
                _llm_metrics.record_validation_rejection(reason="unsupported_causation", node="core_synthesis")
                continue
            # STATEMENT-CREDIBILITY GUARD (structural, not finding-specific):
            # a hypothesis whose mechanism is "the supervisor's/operator's
            # claim was inaccurate/dishonest" attacks the credibility of a
            # REPORTED statement instead of reasoning about the underlying
            # proposition -- a person's statement is evidence, never itself
            # a root-cause mechanism.
            if hypothesis_attacks_statement_credibility(statement):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — attacks the credibility of "
                    "a reported statement instead of reasoning about the underlying proposition"
                ))
                from app.services import llm_metrics as _llm_metrics
                _llm_metrics.record_validation_rejection(reason="invalid_hypothesis", node="core_synthesis")
                continue
            # HYPOTHESIS CAUSALITY FILTER (structural, not finding-specific):
            # a "hypothesis" that just restates an evidence gap already
            # stated in the finding (e.g. "the certificate was not
            # available") is not a causal explanation for WHY the
            # deviation occurred -- it's a fact the finding already
            # gives, dressed up as a hypothesis. Reject it rather than
            # let it crowd out an actual candidate cause.
            from app.agent.causal_guard import is_evidence_state_not_hypothesis
            if is_evidence_state_not_hypothesis(statement, ch.get("name")):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — describes an evidence state "
                    "or investigation uncertainty rather than proposing a concrete causal mechanism"
                ))
                from app.services import llm_metrics as _llm_metrics
                _llm_metrics.record_validation_rejection(reason="invalid_hypothesis", node="core_synthesis")
                continue
            if is_evidence_gap_not_hypothesis(statement, source_text):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — restates an evidence "
                    "gap/fact already stated in the finding rather than proposing a causal explanation"
                ))
                from app.services import llm_metrics as _llm_metrics
                _llm_metrics.record_validation_rejection(reason="invalid_hypothesis", node="core_synthesis")
                continue
            # PROVENANCE ENFORCEMENT (unconditional -- Phase 1 of the final
            # hardening pass: exactly ONE provenance policy, no legacy-shape
            # bypass). Every hypothesis MUST cite at least one claim id via
            # supporting_claim_ids or contradicting_claim_ids, and every
            # cited id MUST exist in this finding's canonical evidence
            # ledger. A hypothesis with no citable claim, or one citing an
            # id the ledger never issued, is rejected outright -- the LLM
            # cannot invent provenance, and a plausible-reading statement
            # with zero evidence linkage is not a hypothesis, it's prose.
            raw_supporting_ids = ch.get("supporting_claim_ids")
            raw_contradicting_ids = ch.get("contradicting_claim_ids")
            supporting_ids = [c for c in (raw_supporting_ids or []) if isinstance(c, str)]
            contradicting_ids = [c for c in (raw_contradicting_ids or []) if isinstance(c, str)]
            invalid_ids = [c for c in (*supporting_ids, *contradicting_ids) if c not in valid_claim_ids]
            if invalid_ids:
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — cited claim id(s) "
                    f"{invalid_ids} that do not exist in the evidence ledger"
                ))
                from app.services import llm_metrics as _llm_metrics
                _llm_metrics.record_validation_rejection(reason="invalid_provenance", node="core_synthesis")
                continue
            if not supporting_ids and not contradicting_ids:
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — cited zero supporting "
                    "or contradicting claim provenance"
                ))
                from app.services import llm_metrics as _llm_metrics
                _llm_metrics.record_validation_rejection(reason="missing_provenance", node="core_synthesis")
                continue
            new_hyp = CandidateHypothesis(
                id=str(ch.get("id", "H")),
                name=str(ch.get("name", "HYPOTHESIS")),
                statement=statement,
                status=hyp_status,
                evidence_needed=clean_structured_leak(ch.get("evidence_needed", "")) or "Investigation required",
                discrimination_evidence=clean_structured_leak(ch.get("discrimination_evidence")) or None,
                relevance_rank=rank,
                rationale=clean_structured_leak(ch.get("rationale")) or None,
                evidence_against=clean_structured_leak(ch.get("evidence_against")) or None,
                confirms_if=clean_structured_leak(ch.get("confirms_if")) or None,
                refutes_if=clean_structured_leak(ch.get("refutes_if")) or None,
                supporting_evidence=[claim_text_by_id[c] for c in supporting_ids if c in claim_text_by_id],
                contradicting_evidence=[claim_text_by_id[c] for c in contradicting_ids if c in claim_text_by_id],
            )
            # Phase 7 Section 5: validate deepens_hypothesis_id against the
            # actual set of ids proposed in this response before trusting
            # it — never against the id of the hypothesis itself either.
            _raw_deepens = ch.get("deepens_hypothesis_id")
            if isinstance(_raw_deepens, str) and _raw_deepens in _raw_hyp_ids and _raw_deepens != new_hyp.id:
                new_hyp.deepens_hypothesis_id = _raw_deepens
            # CROSS-HYPOTHESIS SEMANTIC CONSISTENCY: reject a hypothesis
            # whose own discrimination/confirms/refutes text asserts that
            # its evidence "supports"/"confirms" a DIFFERENT hypothesis id
            # -- evidence and discrimination criteria must describe the SAME
            # causal proposition as the hypothesis they're attached to.
            if hypothesis_discrimination_cites_wrong_id(new_hyp):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {new_hyp.id} — its discrimination criterion "
                    "describes evidence that supports a DIFFERENT hypothesis id, not its own"
                ))
                from app.services import llm_metrics as _llm_metrics
                _llm_metrics.record_validation_rejection(reason="invalid_hypothesis", node="core_synthesis")
                continue

            from app.agent.causal_guard import evaluate_causal_eligibility
            is_eligible, rejection_reason = evaluate_causal_eligibility(
                new_hyp,
                evidence_ledger=evidence_ledger,
                mechanism=mechanism,
                source_text=source_text,
            )
            if not is_eligible:
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {new_hyp.id} — failed causal eligibility ({rejection_reason})"
                ))
                from app.services import llm_metrics as _llm_metrics
                _llm_metrics.record_validation_rejection(reason="causal_eligibility", node="core_synthesis")
                continue

            cand_hypotheses.append(new_hyp)

    # HYPOTHESIS COUNT CAP (Phase 4/9 of the LLM-boundary rebuild): the
    # compact prompt now asks for at most _MAX_LLM_HYPOTHESES, but this is
    # the deterministic enforcement of that limit -- never trust prompt
    # wording alone to bound how many a model actually returns. Highest-
    # ranked (then first-seen) survive; this is independent of, and
    # stricter than, the separate 4-hypothesis cap final_evidence_
    # verification applies later as defense-in-depth.
    from app.services import llm_metrics as _llm_metrics
    if len(cand_hypotheses) > _MAX_LLM_HYPOTHESES:
        _rank_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        dropped = cand_hypotheses[_MAX_LLM_HYPOTHESES:]
        cand_hypotheses = sorted(cand_hypotheses, key=lambda h: _rank_order.get(h.relevance_rank, 1))[:_MAX_LLM_HYPOTHESES]
        trace.append(AgentTraceStep.warn(
            f"Core Synthesis: LLM returned {len(dropped) + len(cand_hypotheses)} hypotheses — capped to "
            f"{_MAX_LLM_HYPOTHESES} highest-ranked"
        ))
        for _ in dropped:
            _llm_metrics.record_validation_rejection(reason="other", node="core_synthesis")

    # Semantically separated per Phase 5: "generated" is what the LLM
    # actually proposed in this response (_raw_hyp_count), "accepted" is
    # what survived every guard above including the count cap -- never
    # conflated with deterministic-fallback-generated hypotheses, which
    # are counted separately at their own call sites.
    from app.agent.causal_graph import evaluate_root_cause_eligibility, select_authoritative_leading_hypothesis
    for h in cand_hypotheses:
        el, supp, _, _, c_lvl, promo = evaluate_root_cause_eligibility(
            h,
            evidence_items=evidence_ledger,
            conflicts=canonical.evidence_conflicts if canonical else None,
            referenced_docs=canonical.referenced_documents if canonical else None,
            canonical_state=canonical,
        )
        if promo and supp == SupportLevel.SUPPORTED:
            h.status = "SUPPORTED"
            h.causal_level = c_lvl

    lead_id, lead_mode, authoritative_rc_status, lead_rationale = select_authoritative_leading_hypothesis(
        cand_hypotheses,
        conflicts=canonical.evidence_conflicts if canonical else None,
        evidence_ledger=evidence_ledger,
    )

    root_cause = RootCauseAnalysis(
        status=authoritative_rc_status,
        category=rc_category,
        statement=clean_structured_leak(raw_rc.get("statement")) or None,
        leading_hypothesis=lead_id,
        candidate_hypotheses=cand_hypotheses,
        risk_of_recurrence=raw_rc.get("risk_of_recurrence", "NOT_ASSESSABLE"),
        narrative=clean_structured_leak(raw_rc.get("narrative")) or "The available evidence establishes the observed condition but does not establish why it occurred.",
        root_cause_basis=clean_structured_leak(raw_rc.get("root_cause_basis")) or None,
        evidence_required=[
            clean_structured_leak(x) for x in raw_rc.get("evidence_required", []) if clean_structured_leak(x)
        ],
        leading_hypothesis_rationale=lead_rationale or clean_structured_leak(raw_rc.get("leading_hypothesis_rationale")) or None,
    )

    # -----------------------------------------------------------------
    # 2. 5-Why
    # -----------------------------------------------------------------
    raw_fw = parsed.get("five_why", {})
    fw_steps = []
    reported_facts = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.REPORTED]
    from app.services.status_normalizer import normalize_five_why_status

    # F4b — The deferral gate fires when requirement status is truly uncertain
    # (primary uncertainty is REQUIREMENT_UNCERTAIN, e.g. finding states the requirement
    # itself could not be determined or located).
    if canonical and (
        getattr(canonical, "primary_uncertainty", None) == "REQUIREMENT_UNCERTAIN"
        or getattr(canonical, "semantic_type", None) == "REQUIREMENT_UNCERTAIN"
    ):
        trace.append(AgentTraceStep.warn(
            "Core Synthesis: 5-Why deferred pending requirement resolution (INV-5WHY-UNCERTAINTY-001)"
        ))
        fw_steps.append(FiveWhyStep(
            level=1,
            question="Why-chain deferred pending requirement resolution",
            answer="Why-chain deferred because the applicable requirement/deviation status has not yet been established.",
            status="UNKNOWN",
        ))
        five_why = FiveWhyAnalysis(
            steps=fw_steps,
            is_complete=False,
            status_note="EVIDENCE BOUNDARY — Applicable requirement is unresolved; causal analysis deferred until compliance status is established.",
        )
    else:
        for s in raw_fw.get("steps", []):
            if isinstance(s, dict):
                # Single authoritative status-normalization boundary (same
                # module used for root_cause/hypothesis statuses above) --
                # maps LLM synonyms ("CONFIRMED", "CONFLICTING", "STATED", etc.)
                # onto their correct canonical status instead of a blunt
                # substring-only fallback, while never raising on an invalid enum.
                st = normalize_five_why_status(s.get("status"))
            question = clean_structured_leak(s.get("question", ""))
            answer = clean_structured_leak(s.get("answer", ""))

            # MIXED-EVIDENCE GUARD (structural): a compound answer whose
            # clauses carry DIFFERENT evidence provenance (e.g. "X
            # stated Y occurred, but Z was independently observed") must
            # never collapse to one status word -- checked BEFORE the
            # pure-REPORTED check below, since a mixed sentence would
            # otherwise get blanket-downgraded to REPORTED_UNVERIFIED,
            # hiding that part of it (e.g. "Z was not available") is
            # independently a VERIFIED observation, not a reported one.
            mixed = classify_mixed_evidence_answer(answer)
            if mixed:
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: 5-Why step {len(fw_steps) + 1} answer combines clauses with "
                    "different evidence provenance (reported + independently observed) — labeled MIXED"
                ))
                st = mixed
            # REPORTED != VERIFIED GUARD (structural): the model
            # labeled this step VERIFIED, but its content traces back to
            # a REPORTED evidence-ledger claim (someone's account), not
            # a VERIFIED one -- never let a status word alone promote a
            # reported statement to established fact.
            elif answer_asserts_verified_but_is_reported(answer, st, reported_facts, verified_facts):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: 5-Why step {len(fw_steps) + 1} labeled VERIFIED but its content "
                    "traces to a REPORTED statement, not a VERIFIED fact — downgraded to REPORTED_UNVERIFIED"
                ))
                st = "REPORTED_UNVERIFIED"

            # CONSOLIDATED WHY-QUESTION QUALITY GATE (Phase 5)
            from app.agent.causal_guard import validate_why_question
            prev_ans = fw_steps[-1].answer if fw_steps else None
            is_valid_q, invalid_reason = validate_why_question(
                question=question,
                previous_answer=prev_ans,
                observation=observed_deviation,
                mechanism=mechanism,
                finding_text=finding_text,
            )
            if not is_valid_q:
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: 5-Why step {len(fw_steps) + 1} question rejected ({invalid_reason}) — truncating chain here"
                ))
                break

            if is_circular_why_answer(question, answer):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: 5-Why step {len(fw_steps) + 1} answer restated its own "
                    "question instead of explaining it — truncating chain here"
                ))
                answer = "The available evidence does not establish this — the prior answer could not be traced to a distinct causal explanation."
                st = "UNKNOWN"
            elif fw_steps and repeats_previous_why_answer(fw_steps[-1].answer, answer):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: 5-Why step {len(fw_steps) + 1} repeated the previous "
                    "step's answer instead of advancing — truncating chain here"
                ))
                answer = "The available evidence does not establish a further cause beyond the preceding step."
                st = "UNKNOWN"
            elif restates_observation(answer, observed_deviation, question):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: 5-Why step {len(fw_steps) + 1} answer merely restated the "
                    "original observation instead of explaining it — truncating chain here"
                ))
                answer = "The available evidence establishes that the deviation occurred, but does not establish why."
                st = "UNKNOWN"
            elif ungrounded_entities(answer, source_text):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: 5-Why step {len(fw_steps) + 1} answer referenced an "
                    "ungrounded entity/number — truncating chain here"
                ))
                answer = "The available evidence does not establish this — the answer could not be traced to this finding's evidence."
                st = "UNKNOWN"
            else:
                # CAUSAL BOUNDARY GUARD (INV-5WHY-CAUSAL-001): a 5-Why answer
                # must never functionally select/assert an UNVERIFIED
                # candidate hypothesis as though it were the established
                # explanation, no matter what status label the LLM gave the
                # step or how the claim is hedged ("may have ..."). The
                # 5-Why engine must consume the canonical causal-graph
                # hypothesis state, not independently invent an explanation.
                from app.agent.causal_guard import (
                    answer_selects_unverified_hypothesis,
                    build_causal_boundary_answer,
                    five_why_answer_contains_unverified_modal_causation,
                )
                _has_verified_mechanism = getattr(mechanism, "status", None) == "VERIFIED"
                _selected_hyp = answer_selects_unverified_hypothesis(answer, cand_hypotheses, status=st)
                _modal_causation = five_why_answer_contains_unverified_modal_causation(
                    answer, st, has_verified_mechanism=_has_verified_mechanism
                )
                if _selected_hyp is not None or _modal_causation:
                    if _selected_hyp is not None:
                        trace.append(AgentTraceStep.warn(
                            f"Core Synthesis: 5-Why step {len(fw_steps) + 1} answer selected unverified "
                            f"hypothesis {getattr(_selected_hyp, 'id', '?')} ({getattr(_selected_hyp, 'name', '?')}) "
                            "as the explanation — replaced with the canonical evidence boundary"
                        ))
                    else:
                        trace.append(AgentTraceStep.warn(
                            f"Core Synthesis: 5-Why step {len(fw_steps) + 1} answer contained an unverified "
                            "modal causal claim (INV-5WHY-CAUSAL-002) with no verified mechanism — replaced "
                            "with the canonical evidence boundary"
                        ))
                    answer = build_causal_boundary_answer(
                        candidate_hypotheses=cand_hypotheses,
                        comparison_type=getattr(canonical, "comparison_type", None),
                        comparison_left=getattr(canonical, "comparison_left", None),
                        comparison_right=getattr(canonical, "comparison_right", None),
                        comparison_left_qualifier=getattr(canonical, "comparison_left_qualifier", None),
                        comparison_subtype=getattr(canonical, "comparison_subtype", None),
                        measurement_value=getattr(getattr(canonical, "measurement", None), "value", None),
                        measurement_unit=getattr(getattr(canonical, "measurement", None), "unit", None),
                        measurement_qualifier=getattr(getattr(canonical, "measurement", None), "qualifier", None),
                        observed_deviation=observed_deviation,
                    )
                    st = "UNKNOWN"

            fw_steps.append(FiveWhyStep(
                level=int(s.get("level", len(fw_steps) + 1)),
                question=question,
                answer=answer,
                status=st,
                evidence_reference=clean_structured_leak(s.get("evidence_reference")),
            ))
            # Evidence-bound stop: truncate right after the first step
            # whose status is UNKNOWN/NOT_ESTABLISHED (including one this
            # guard just forced) — never fabricate steps past that point.
            if st in ("UNKNOWN", "NOT_ESTABLISHED"):
                break

    five_why = FiveWhyAnalysis(
        steps=fw_steps,
        is_complete=bool(raw_fw.get("is_complete", False)) and not any(s.status in ("UNKNOWN", "NOT_ESTABLISHED") for s in fw_steps),
        status_note=clean_structured_leak(raw_fw.get("status_note")) or "Stopped at evidence boundary",
    )

    # -----------------------------------------------------------------
    # 3. Contributing factors (dynamically derived, not a fixed
    # "not established" filler — an empty list is a correct outcome when
    # the finding supports none).
    # -----------------------------------------------------------------
    contributing_factors: list[ContributingFactor] = []
    valid_cf_status = {"ESTABLISHED", "POTENTIAL", "POSSIBLE_UNCONFIRMED", "VERIFIED"}
    for cf in parsed.get("contributing_factors", []):
        if not isinstance(cf, dict):
            continue
        description = clean_structured_leak(cf.get("description", ""))
        if not description:
            continue
        # A boilerplate "not established" non-answer carries zero
        # analytical content -- an empty contributing_factors list is
        # the correct representation of "nothing supports a factor
        # here", not an entry that just says so in words.
        if is_generic_non_analysis_filler(description):
            trace.append(AgentTraceStep.warn(
                "Core Synthesis: dropped contributing factor — generic non-analysis filler text"
            ))
            continue
        # Same causality discipline applied to hypotheses: a
        # "contributing factor" that's really just a restated evidence
        # gap or the reported mechanism itself (not a CONDITION that
        # made the mechanism more likely) carries no analytical content
        # beyond what's already stated elsewhere in the report.
        if is_evidence_gap_not_hypothesis(description, source_text):
            trace.append(AgentTraceStep.warn(
                "Core Synthesis: dropped contributing factor — restates an evidence gap/fact "
                f"already stated in the finding rather than a condition contributing to it: {description!r}"
            ))
            continue
        cf_evidence_status = str(cf.get("evidence_status", "INFERRED")).upper()
        if cf_evidence_status not in {s.value for s in EvidenceStatus}:
            cf_evidence_status = "INFERRED"
        cf_status = str(cf.get("status", "POTENTIAL")).upper()
        if cf_status not in valid_cf_status:
            cf_status = "POTENTIAL" if cf_evidence_status != "VERIFIED" else "ESTABLISHED"
        contributing_factors.append(ContributingFactor(
            description=description,
            evidence_status=EvidenceStatus(cf_evidence_status),
            status=cf_status,
            rationale=clean_structured_leak(cf.get("rationale")) or None,
            evidence_required=clean_structured_leak(cf.get("evidence_required")) or None,
        ))

    # -----------------------------------------------------------------
    # 4. Production-Grade Post-Processing Guards
    # -----------------------------------------------------------------

    # CERTAINTY MONOTONICITY & CATEGORY LOCK: if root cause is
    # NOT_ESTABLISHED, category MUST be TO_BE_CONFIRMED.
    if root_cause.status == RootCauseStatus.NOT_ESTABLISHED:
        root_cause.category = "TO_BE_CONFIRMED"

    # DOMAIN LEAKAGE & CAUSAL ANCHOR GUARD
    # Drop candidate hypotheses that invoke unsupported domains or lack causal anchors
    kept_hypotheses = []
    for h in root_cause.candidate_hypotheses:
        if mentions_unsupported_domain(h.statement, source_text) or mentions_unsupported_domain(h.name, source_text):
            trace.append(AgentTraceStep.warn(
                f"Core Synthesis: dropped hypothesis mentioning unsupported domain: {h.statement!r}"
            ))
        elif ungrounded_entities(h.statement, source_text):
            trace.append(AgentTraceStep.warn(
                f"Core Synthesis: dropped hypothesis with ungrounded entity: {h.statement!r}"
            ))
        else:
            kept_hypotheses.append(h)

    root_cause.candidate_hypotheses = kept_hypotheses

    # Same grounding/domain discipline applied to contributing factors —
    # they carry the same fabrication risk as hypotheses.
    kept_factors = []
    for cf in contributing_factors:
        if mentions_unsupported_domain(cf.description, source_text):
            trace.append(AgentTraceStep.warn(
                f"Core Synthesis: dropped contributing factor — unsupported domain: {cf.description!r}"
            ))
        elif ungrounded_entities(cf.description, source_text):
            trace.append(AgentTraceStep.warn(
                f"Core Synthesis: dropped contributing factor — ungrounded entity: {cf.description!r}"
            ))
        else:
            kept_factors.append(cf)
    contributing_factors = kept_factors

    # REQUIREMENT_UNCERTAIN CAUSAL GATE (INV-UNCERTAINTY-002):
    # When the governing requirement/standard is unresolved, causal hypothesis generation
    # is blocked. Plausible mechanisms (e.g. lack of awareness/training) cannot be asserted
    # as candidate causes until the requirement and deviation status are established.
    if canonical and (getattr(canonical, "primary_uncertainty", None) == "REQUIREMENT_UNCERTAIN" or getattr(canonical, "semantic_type", None) == "REQUIREMENT_UNCERTAIN"):
        root_cause.candidate_hypotheses = []
        root_cause.status = RootCauseStatus.NOT_ESTABLISHED
        root_cause.category = "TO_BE_CONFIRMED"
        trace.append(AgentTraceStep.ok(
            "Core Synthesis: causal hypothesis generation blocked because governing requirement is unresolved"
        ))
    else:
        # Ensure candidate_hypotheses is NEVER empty for other non-blocked finding types
        if not root_cause.candidate_hypotheses:
            from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
            fallback_hyps, _ = build_deterministic_investigation_plan(
                finding_text, evidence_ledger, canonical_subject=canonical_subject, canonical_state=canonical,
                semantic_context=semantic_context,
            )
            root_cause.candidate_hypotheses = fallback_hyps
            trace.append(AgentTraceStep.warn("Core Synthesis: generated fallback candidate hypotheses from finding text"))

    # Every surviving hypothesis carries its own deterministic confidence
    # grade (never asserted by the LLM) so a report never shows a bare
    # hypothesis with no sense of how strong it is.
    for h in root_cause.candidate_hypotheses:
        h.confidence = hypothesis_confidence(h)

    # LEADING HYPOTHESIS ENFORCEMENT: root_cause.status staying
    # NOT_ESTABLISHED must not mean the report goes silent about which
    # candidate is best-evidenced -- deterministic, not dependent on the
    # LLM remembering to fill leading_hypothesis. select_leading_hypothesis
    # is the single source of truth for this rule (also used by the
    # analytical validator) -- it deliberately returns None when
    # hypotheses are equally plausible rather than forcing a pick. That
    # TIED case must still be visible, not indistinguishable from "no
    # hypotheses at all" -- leading_hypothesis_status/_display make that
    # explicit instead of leaving the field silently blank.
    root_cause.leading_hypothesis_status = leading_hypothesis_status(root_cause.candidate_hypotheses)
    if not root_cause.leading_hypothesis:
        root_cause.leading_hypothesis = leading_hypothesis_display(root_cause.candidate_hypotheses)
        if root_cause.leading_hypothesis_status == "SELECTED":
            root_cause.confidence = leading_hypothesis_confidence(
                root_cause.candidate_hypotheses, root_cause.leading_hypothesis
            )

    # CONFLICT-AWARE TIE OVERRIDE: candidate hypotheses that are competing
    # explanations for the SAME unresolved evidence conflict must never have
    # a single leader promoted merely because of an incidental scoring
    # difference between equally-plausible statements -- see
    # apply_conflict_tie_override's docstring. A no-op unless the scorer
    # actually picked a winner above.
    from app.agent.analytical_validator import apply_conflict_tie_override
    apply_conflict_tie_override(root_cause, has_unresolved_conflict)

    # REPORTED STATEMENT ATTRIBUTION GUARD
    # If narrative claims carelessness/negligence from reported statements, flag & sanitize
    if "careless" in root_cause.narrative.lower() or "negligen" in root_cause.narrative.lower():
        root_cause.narrative = (
            "Personnel reported attribution to human factors/carelessness. "
            "This statement is unverified and does not establish objective root cause."
        )

    return root_cause, five_why, contributing_factors


def _derive_ca_draft_fields(root_cause, impact, canonical=None) -> dict:
    """Derives the 5 CA-draft fields from the already-synthesized
    root_cause/impact objects instead of asking the LLM for a second,
    separate restatement of the same analysis."""
    affected = (impact.affected_object if impact and impact.affected_object else None) or "the affected process/item"
    from app.services.semantic_subject import validate_semantic_subject
    if not validate_semantic_subject(affected):
        affected = "the affected process/record"

    leading_stmt = None
    if root_cause and getattr(root_cause, "leading_hypothesis_status", None) == "SELECTED" and getattr(root_cause, "candidate_hypotheses", None):
        leading_id = (getattr(root_cause, "leading_hypothesis", None) or "").split(" — ", 1)[0].strip()
        match = next((h for h in root_cause.candidate_hypotheses if h.id == leading_id), None)
        if match:
            leading_stmt = match.statement

    is_established = bool(root_cause) and root_cause.status not in (
        RootCauseStatus.NOT_ESTABLISHED, "NOT_ESTABLISHED",
    )
    if is_established:
        root_cause_text = root_cause.narrative
    elif leading_stmt:
        root_cause_text = f"NOT_ESTABLISHED — leading hypothesis: {leading_stmt}"
    elif root_cause:
        root_cause_text = f"NOT_ESTABLISHED — {root_cause.narrative}"
    else:
        root_cause_text = "NOT_ESTABLISHED"

    preventive_action = (
        f"Address the confirmed cause via {leading_stmt[0].lower()}{leading_stmt[1:]}" if leading_stmt
        else "Strengthen the relevant control once root cause is confirmed."
    )

    import re as _re
    # DELIVERY_VS_RECEIPT conflict shape ("X receipt", built by the
    # delivery/receipt branch of _derive_deterministic_impact) -- "permitting
    # independent execution or release" presupposes a release/operation
    # dependency this finding never establishes; the appropriate action is
    # verifying notification/receipt/acknowledgement status, conditional on
    # reliance (Section 16).
    # For duplicate payment / transaction findings: containment and recovery review
    search_text = f"{affected} {getattr(impact, 'narrative', '') or ''}"
    if _re.search(r"\b(?:duplicate\s+payment|overpayment|paid\s+twice|double\s+payment)\b", search_text, _re.IGNORECASE):
        from app.services.cost_analysis import extract_explicit_amounts, format_currency_amount
        explicit_amts = extract_explicit_amounts(search_text)
        amt_str = f" {format_currency_amount(explicit_amts[0][0], explicit_amts[0][1])}" if explicit_amts else ""
        immediate_action = (
            f"Verify whether the duplicate{amt_str} payment has been reversed, recovered, or credited by the supplier "
            "and place the transaction under appropriate financial reconciliation review."
        )
    elif canonical is not None and getattr(canonical, "semantic_type", None) == "EVENT_SEQUENCE_CONTROL" and getattr(canonical, "transition_type", None):
        # EVENT_SEQUENCE_CONTROL finding (Section 18): the immediate action
        # follows the control risk -- verify the authorization chain for
        # the transition, and separately protect any downstream decision
        # that may depend on it, without assuming either was improper.
        _transition_label = canonical.transition_type.replace("_", " ").lower()
        if getattr(canonical, "downstream_action_present", False):
            immediate_action = (
                f"Verify the authorization chain for the {_transition_label} before relying on it, and place "
                "any downstream decision that may depend on this transition under appropriate review pending "
                "that verification."
            )
        else:
            immediate_action = (
                f"Verify the authorization chain for the {_transition_label} and retrieve contemporaneous "
                "records to determine whether the applicable control was executed before relying on the "
                "affected record for further decisions."
            )
    elif canonical is not None and getattr(canonical, "semantic_type", None) == "MISSING_RECORD" and getattr(canonical, "missing_record_activity", None):
        # MISSING_RECORD finding (Section 12): generated from the current
        # evidence state -- activity status is UNKNOWN until evidence
        # establishes it, so the action is to verify execution and
        # reconstruct the record trail, never to assume non-performance.
        # When a downstream action was also detected, additionally flag
        # that it should be assessed for continued support -- never
        # asserted as invalid.
        _activity = canonical.missing_record_activity
        if getattr(canonical, "downstream_action_present", False):
            immediate_action = (
                f"Verify execution of the required {_activity} and reconstruct the applicable record trail "
                "before relying on the affected control evidence, and separately assess whether the "
                "downstream action reported in the finding remains appropriately supported."
            )
        else:
            immediate_action = (
                f"Verify execution of the required {_activity} and reconstruct the applicable record trail "
                "before relying on the affected control evidence."
            )
    elif canonical is not None and getattr(canonical, "comparison_type", None) and getattr(canonical, "comparison_left", None) and getattr(canonical, "comparison_right", None):
        # COMPARISON_MISMATCH finding (Section 10): generated from the
        # finding TYPE, not hardcoded to any one domain -- the same
        # sentence applies whether the compared values are a yield, a
        # temperature, an invoice amount, or a quantity.
        _qualified_left = (
            f"{canonical.comparison_left_qualifier} {canonical.comparison_left}"
            if getattr(canonical, "comparison_left_qualifier", None) else canonical.comparison_left
        )
        if getattr(canonical, "comparison_subtype", None) == "PARAMETER_MISMATCH":
            # Subtype-specific wording (Section 13): no "calculation" basis
            # to reconcile against for a parameter mismatch -- the relevant
            # authority is the approved parameter itself plus batch/process
            # records.
            immediate_action = (
                f"Verify the applicable approved parameter and independently reconcile the {_qualified_left} "
                "against authoritative batch and process records before relying on the affected batch record "
                "for further decisions."
            )
        else:
            immediate_action = (
                f"Independently reconcile the {_qualified_left} against the underlying "
                f"{canonical.comparison_basis or 'source records'} and applicable calculation/reference basis, "
                "and assess the discrepancy before relying on the affected record for further decisions."
            )
    elif affected.strip().lower().endswith(" receipt"):
        _base_subject = affected[: -len(" receipt")]
        _base_subject = _base_subject[0].lower() + _base_subject[1:] if _base_subject else _base_subject
        immediate_action = (
            "Verify the current notification, receipt, and acknowledgement status for affected personnel "
            f"against authorized communication records before relying on {_base_subject} for "
            "further action, where applicable."
        )
    # Avoid "Verify the current status of X status" when the affected-object
    # phrase already names a status/qualification (e.g. "Operator training
    # status for the revised procedure") -- say it once, not twice.
    elif affected.strip().lower().endswith(("status", "qualification")) or " status for " in affected.lower() or " qualification for " in affected.lower():
        lead_word = affected.split(" ", 1)[0].lower()
        article = "" if lead_word in ("the", "a", "an") else "the "
        immediate_action = f"Verify {article}{affected[0].lower()}{affected[1:]} against authorized records before permitting independent execution or release, where applicable."
    elif _re.search(r"\b(records?|logs?|documentation)\b", affected, _re.IGNORECASE):
        # The affected object IS itself a record/log/documentation artifact
        # (e.g. "Temperature monitoring records for refrigerator
        # QC-REF-02") -- "permitting independent execution" doesn't fit a
        # records-completeness finding; the appropriate immediate action is
        # reviewing the affected records and assessing current status
        # against the applicable procedure, not a personnel authorization
        # decision.
        lead_word = affected.split(" ", 1)[0].lower()
        article = "" if lead_word in ("the", "a", "an") else "the "
        immediate_action = (
            f"Review {article}{affected[0].lower()}{affected[1:]} and assess the current status of the "
            "affected item(s) in accordance with the applicable procedure."
        )
    else:
        immediate_action = (
            f"Verify the current status of {affected} against authorized records before permitting "
            "independent execution or release, where applicable."
        )

    return {
        "immediate_action": immediate_action,
        "root_cause": root_cause_text,
        "root_cause_category": (
            "TO_BE_CONFIRMED" if (not root_cause or not is_established) else root_cause.category
        ),
        "preventive_action": preventive_action,
        "impact_analysis": (impact.narrative if (impact and impact.narrative) else "Impact scope pending auditor verification."),
    }


def _trim_evidence_for_recovery(claim_ids: list[tuple[str, Any]]) -> list[dict]:
    """Build a materially smaller evidence context for the recovery retry.

    Keeps only VERIFIED/REPORTED items (the tiers the causal reasoning
    actually leans on) and caps the count, ranked by relevance -- this is
    what makes the recovery call "compact", not just a shorter output
    budget on the exact same input. Takes the SAME (id, EvidenceItem) pairs
    `_assign_claim_ids` produced for the primary call, so a claim's ID is
    identical across both prompts, and emits the same compact
    {id, claim, status} shape the primary prompt uses -- not a full
    `.model_dump()` (source/source_reference/relevance/notes), which is
    both unnecessary token weight and inconsistent with the primary
    prompt's evidence shape.
    """
    relevance_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    candidates = [
        (cid, e) for cid, e in claim_ids
        if e.status in (EvidenceStatus.VERIFIED, EvidenceStatus.REPORTED)
    ]
    candidates.sort(key=lambda pair: relevance_rank.get(getattr(pair[1], "relevance", "MEDIUM"), 1))
    trimmed = candidates[:6] or claim_ids[:6]
    return [
        {"id": cid, "claim": e.claim, "status": getattr(e.status, "value", str(e.status))}
        for cid, e in trimmed
    ]


_EXPIRY_CONTEXT_RE = re.compile(r"\bexpir\w*\b[^.]{0,40}?(\d{1,2}\s+\w+\s+\d{4})", re.IGNORECASE)
_USE_CONTEXT_RE = re.compile(
    r"\b(?:used|performed|conducted|carried\s+out|executed)\b[^.]{0,60}?(\d{1,2}\s+\w+\s+\d{4})",
    re.IGNORECASE,
)


def _detect_expiry_then_use(finding_text: str) -> bool:
    """Structural (date-comparison, not domain-vocabulary) detection of the
    "X expired on date A; X was used on date B where B is on/after A"
    pattern -- generalizes across calibration certificates, reagent expiry,
    qualification expiry, etc. without a per-domain keyword list (Section
    32: structural guards over keyword guards). Used only to select a more
    specific, still evidence-bounded potential_effect narrative below --
    never to assert a mechanism or root cause."""
    expiry_match = _EXPIRY_CONTEXT_RE.search(finding_text or "")
    use_match = _USE_CONTEXT_RE.search(finding_text or "")
    if not expiry_match or not use_match:
        return False
    try:
        from dateutil import parser as _date_parser
        expiry_date = _date_parser.parse(expiry_match.group(1))
        use_date = _date_parser.parse(use_match.group(1))
    except (ValueError, OverflowError):
        return False
    return use_date >= expiry_date


_CREDENTIAL_DOMAIN_RE = re.compile(
    r"\b([a-z]+)\s+(?:certificate|certification|license|licence|qualification|accreditation|permit)\b",
    re.IGNORECASE,
)


def _extract_expiry_domain_word(finding_text: str) -> str | None:
    """The word modifying the credential/document that expired ("calibration
    certificate" -> "calibration", "training certification" -> "training")
    -- structural (pattern: word immediately before a credential-type noun),
    not a hardcoded domain list. Used for process_at_risk so the control
    domain reflects what actually expired, not the entity resolver's own
    object noun (Phase 6: "do not let the entity resolver determine the
    process name" -- "balance BAL-014" the entity is not "calibration" the
    control domain, even though both describe the same finding)."""
    match = _CREDENTIAL_DOMAIN_RE.search(finding_text or "")
    return match.group(1).lower() if match else None


def _extract_use_date_text(finding_text: str) -> str | None:
    """The raw date TEXT (not a parsed object) from the "used/performed on
    <date>" clause -- the affected/exposure period for an expiry-then-use
    finding is the date the out-of-status resource was actually used, not
    the audit/discovery date (Known Failure 8). Only called after
    _detect_expiry_then_use has already confirmed this pattern applies."""
    use_match = _USE_CONTEXT_RE.search(finding_text or "")
    return use_match.group(1) if use_match else None


def _reportedly_clause(obj: str, condition: str | None) -> str:
    """Builds a grammatically safe "<obj> was reportedly <condition>" (or
    the active-voice equivalent) clause for a deviation_condition value that
    could be EITHER an adjective/participle predicate ("incomplete", "not
    completed", "operated outside its validated range" -- correctly follows
    "was reportedly") OR a bare verb-phrase naming an OMITTED action
    ("distribute the revised SOP", from a "failed to <verb> <object>"
    source pattern -- "X was reportedly distribute the revised SOP" is
    ungrammatical; the deviation here IS the failure to do that action, so
    "X reportedly failed to <verb-phrase>" is the correct reconstruction).
    Reuses the same closed adjective vocabulary and "not "-prefix check
    format_deviation_why_question uses for the identical classification
    problem, so both call sites agree on which conditions are which shape."""
    cond = (condition or "").strip().rstrip(".")
    if not cond or cond.upper() == "UNKNOWN":
        return f"{obj} was reportedly in a condition that has not been verified against applicable requirements."
    from app.services.semantic_subject import _CONDITION_ADJECTIVES
    first_word = cond.split()[0].lower()
    if cond.lower().startswith(("was ", "were ", "not ")) or first_word in _CONDITION_ADJECTIVES:
        stripped = re.sub(r"^(?:was|were)\s+", "", cond, flags=re.IGNORECASE)
        return f"{obj} was reportedly {stripped}."
    # A condition that IS (or starts with) a quantity/amount descriptor
    # ("approximately ₹4 lakh of rework costs", "about 12 units") is neither
    # an adjective predicate nor an omitted-action verb phrase -- it names a
    # magnitude the finding associates with the subject, not something the
    # subject "failed to do". "X reportedly failed to approximately ₹4
    # lakh..." (the bare-leading-word verb heuristic below misreading
    # "approximately" as a verb) is exactly the defect this guards against.
    if re.match(r"^(?:approximately|about|roughly|nearly|₹|\$|€|£|\d)", cond, re.IGNORECASE):
        return f"{obj} was reportedly associated with {cond}."
    # A bare leading-word match alone is NOT sufficient to treat `cond` as an
    # omitted-action verb phrase -- it also matches an already-inflected
    # past-tense predicate describing something that DID happen (e.g.
    # "processed the transaction twice due to a retry-queue duplication
    # bug"), where "X reportedly failed to processed..." is both
    # ungrammatical (tense mismatch) and semantically inverted (nothing was
    # omitted). Only a RECOGNIZED infinitive-form omitted-action verb (the
    # same closed whitelist format_deviation_why_question uses for the
    # identical problem) gets the "failed to" reconstruction; anything else
    # falls through to the neutral dash rendering below, matching
    # semantic_subject.py's own established safe fallback for an
    # unclassified condition.
    from app.services.semantic_subject import _TRANSITIVE_FAILED_TO_VERBS
    verb_match = re.match(r"^([a-z]+)\s+\S", cond, re.IGNORECASE)
    if verb_match and verb_match.group(1).lower() in _TRANSITIVE_FAILED_TO_VERBS:
        return f"{obj} reportedly failed to {cond}."
    if verb_match:
        return f"{obj} — {cond}."
    return f"{obj} was reportedly {cond}."


def _derive_deterministic_impact(request_finding_text: str, canonical, observed_deviation: str, semantic_context: Any = None) -> tuple[ImpactAssessment, str, str, str | None]:
    """Shared deterministic impact derivation used both when a compact
    recovery call succeeds (recovery's schema deliberately excludes impact)
    and when full deterministic synthesis runs. Returns
    (impact, clean_noun, topic, actor) so callers can reuse clean_noun/topic
    for CAPA derivation without recomputing them."""
    from app.services.semantic_subject import (
        build_affected_object_phrase,
        extract_temporal_clause,
        resolve_deviation,
        split_topic_and_tail,
        strip_leading_article,
        is_established_subject,
        strip_quantity_prefix,
        topic_word,
        validate_semantic_subject,
    )

    # SINGLE AUTHORITATIVE SUBJECT SOURCE: canonical.finding_subject is
    # produced exactly once, by understand_finding_node's deterministic
    # resolver (resolve_deviation).
    if semantic_context is not None:
        # States A/B (promotion pass): the validated LLM canonical context
        # is authoritative over the legacy CanonicalFindingState/raw-text
        # resolver when present -- an explicit "no entity resolved" (state
        # B) must never be replaced by resolve_deviation()'s own guess.
        from app.services.canonical_context_validator import get_affected_object_candidate
        _canonical_affected = get_affected_object_candidate(semantic_context)
        clean_noun = _canonical_affected or "UNKNOWN — no affected object could be isolated from the finding text"
    else:
        canon_subject = (
            getattr(canonical, "finding_subject", None)
            or getattr(canonical, "affected_object", None)
        ) if canonical else None
        _UNRESOLVED = "UNKNOWN — no affected object could be isolated from the finding text"
        if canon_subject and canon_subject != "UNKNOWN" and validate_semantic_subject(canon_subject):
            clean_noun = canon_subject
        elif canon_subject and str(canon_subject).strip().upper().startswith(
            ("UNKNOWN", "UNRESOLVED", "NOT ESTABLISHED")
        ):
            # The upstream resolver already established there is no isolable
            # subject -- honour that, never independently re-derive one here.
            clean_noun = _UNRESOLVED
        elif canon_subject:
            # PART 1 (architectural convergence): canonical_finding_state is
            # the ONE authoritative subject source -- understand_finding_node
            # already vetted it. Use it as-is even if this stricter local
            # gate would reject it. NEVER re-parse the raw finding when a
            # canonical subject exists.
            clean_noun = canon_subject
        elif canonical is not None:
            # A canonical state is present but carries no subject -> honour
            # that fail-closed result rather than re-deriving one.
            clean_noun = _UNRESOLVED
        else:
            # No canonical state at all (isolated/legacy call path): fall back
            # to the deterministic floor resolver -- this is the floor itself,
            # not a downstream re-derivation past a canonical subject.
            resolved = resolve_deviation(request_finding_text, [])
            clean_noun = resolved.subject or _UNRESOLVED
    clean_noun = strip_quantity_prefix(clean_noun) or clean_noun
    # Normalise every "no usable subject" marker the pipeline can store
    # ("UNRESOLVED — ...", "Finding subject not specifically identified", a
    # generic placeholder) to the ONE marker the branches below gate on with
    # `.startswith("UNKNOWN")`. Without this, a subject that failed resolution
    # upstream would still be fed into `topic_word(...)` and concatenated into
    # a fabricated "<word> operational process" / "<word> and compliance
    # control" -- inventing an affected process the evidence never established
    # (spec §13/§34: prefer an explicit unresolved value over a plausible
    # invention).
    from app.services.semantic_subject import is_established_subject as _is_est_subj
    if not _is_est_subj(clean_noun):
        clean_noun = "UNKNOWN — no affected object could be isolated from the finding text"
    topic = topic_word(clean_noun)
    topic_cap = topic[0].upper() + topic[1:]
    temporal_clause = extract_temporal_clause(request_finding_text)
    expiry_then_use = _detect_expiry_then_use(request_finding_text)
    _use_date_text = _extract_use_date_text(request_finding_text) if expiry_then_use else None
    if _use_date_text:
        # Checked BEFORE the generic temporal_clause extractor (Known
        # Failure 8): a specific date the finding states the resource was
        # actually USED on is always more precise than a vague "during the
        # audit" clause that extract_temporal_clause may also match --
        # audit/discovery date must never win over an actual exposure date
        # the finding provides.
        degraded_period = _use_date_text
    elif temporal_clause:
        degraded_period = temporal_clause[0].upper() + temporal_clause[1:]
    else:
        degraded_period = (
            canonical.affected_period if (canonical and canonical.affected_period != "UNKNOWN") else None
        ) or "UNKNOWN"
    # Strip a stray leading article (e.g. an actor captured mid-sentence as
    # "The operator") before re-embedding it into new sentences, so it never
    # produces a doubled/misplaced article like "The Operator training
    # status" or "the the operator".
    actor = strip_leading_article((canonical.actor if canonical else None) or None)

    # DELIVERY_VS_RECEIPT conflict (Conflict-Center hardening, Section 7-9):
    # checked FIRST, before expiry/tail/record-shaped branches below, since
    # a delivery-vs-receipt conflict is not about equipment use, validated
    # ranges, or record completeness -- those templates would contaminate a
    # notification/communication finding with an unrelated domain (Section
    # 19: "a finding about notification must never inherit the impact
    # template from equipment/calibration findings"). affected_object
    # represents the CONFLICTED EVENT (receipt), not just the document/
    # message; process_at_risk is the communication/acknowledgement
    # control, never a validated-use/equipment template; potential_effect
    # stays conditional on whether personnel acted before the conflict
    # resolved -- never asserts delivery, receipt, or acknowledgement
    # actually failed.
    _delivery_receipt_conflict = None
    if canonical and getattr(canonical, "evidence_conflicts", None):
        _delivery_receipt_conflict = next(
            (c for c in canonical.evidence_conflicts
             if getattr(c, "proposition_type", None) == "DELIVERY_VS_RECEIPT"
             and getattr(c, "status", "UNRESOLVED") == "UNRESOLVED"),
            None,
        )
    # EVENT_SEQUENCE_CONTROL finding (Section 17): the transition and its
    # missing justification are VERIFIED (the finding directly states
    # them) -- state that directly. The downstream action (if any) is a
    # SEPARATE event whose dependency on the transition requires
    # assessment; never assert the transition or downstream action was
    # improper. Generalizes across any transition type (invalidation,
    # override, exception, waiver, ...), not specific to any one domain.
    if canonical and getattr(canonical, "semantic_type", None) == "EVENT_SEQUENCE_CONTROL" and getattr(canonical, "transition_type", None):
        transition_label = canonical.transition_type.replace("_", " ").lower()
        _es_subject = canonical.finding_subject if is_established_subject(getattr(canonical, "finding_subject", None)) else None
        derived_obj = (_es_subject or transition_label)
        derived_obj = derived_obj[0].upper() + derived_obj[1:] if derived_obj else transition_label.capitalize()
        derived_process = canonical.affected_process if (canonical.affected_process not in ("UNKNOWN", "NOT ESTABLISHED", "")) else f"{transition_label.capitalize()} control and authorization"
        downstream_clause = (
            " A downstream action is also reported in the finding; whether it depended on this transition "
            "and remains appropriately supported requires assessment. This does not establish that the "
            "downstream action was improper."
            if getattr(canonical, "downstream_action_present", False) else ""
        )
        if _es_subject and _es_subject.lower() != transition_label:
            derived_effect = (
                f"The {_es_subject} {transition_label} occurred and the required justification is not documented, based on "
                f"the available evidence.{downstream_clause} The applicable control, change justification, and downstream consequences require assessment."
            )
        else:
            derived_effect = (
                f"The {transition_label} occurred and the required justification is not documented, based on "
                f"the available evidence.{downstream_clause} The applicable control and its execution require assessment."
            )
        derived_evidence_needed = (
            f"Records documenting the {transition_label} event, the applicable procedure governing it, and "
            "any authorization/review record for this specific transition"
        )
    # RECURRENCE finding (Section 11/16): the recurrence observation itself
    # is VERIFIED (the finding directly states the deviation was identified
    # across a population of occurrences) -- state it directly. Separate
    # the recurrence OBSERVATION from any prior-action relationship (never
    # implied "ineffective") and from future recurrence RISK (a distinct
    # dimension, never inferred from the fact recurrence already occurred).
    elif canonical and getattr(canonical, "semantic_type", None) == "RECURRENCE" and is_established_subject(getattr(canonical, "finding_subject", None)):
        deviation_subject = canonical.finding_subject
        derived_obj = deviation_subject[0].upper() + deviation_subject[1:]
        derived_process = canonical.affected_process if (canonical.affected_process not in ("UNKNOWN", "NOT ESTABLISHED", "")) else f"{topic_word(deviation_subject).capitalize()} operational process"
        population_clause = f" across {canonical.occurrence_population}" if getattr(canonical, "occurrence_population", None) else ""
        prior_action_clause = (
            " A prior corrective action is also referenced; this does not by itself establish whether "
            "that action was effective, fully implemented, or related to the current mechanism."
            if getattr(canonical, "previous_capa_referenced", False) else ""
        )
        derived_effect = (
            f"The {deviation_subject}{population_clause} is confirmed by the available evidence.{prior_action_clause} "
            "The scope of the affected population and the relationship to any prior corrective action require assessment."
        )
        derived_evidence_needed = (
            f"Records for each occurrence of the {deviation_subject}, the prior corrective action record "
            "(if any), and its implementation/effectiveness evidence"
        )
    # MISSING_RECORD finding (Section 11): the missing evidence itself is
    # VERIFIED (the finding directly states it) -- state that directly,
    # never "reportedly". Separate the OBSERVED condition (record missing)
    # from POTENTIAL impact (compliance-demonstration ability) from any
    # DOWNSTREAM action, and never assert the downstream action was
    # improper absent objective evidence -- only that it requires
    # assessment. Generalizes across any domain (inspection, review,
    # approval, verification, ...), not specific to any one activity type.
    elif canonical and getattr(canonical, "semantic_type", None) == "MISSING_RECORD" and getattr(canonical, "missing_record_activity", None):
        activity = canonical.missing_record_activity
        derived_obj = activity[0].upper() + activity[1:]
        derived_process = canonical.affected_process if (canonical.affected_process not in ("UNKNOWN", "NOT ESTABLISHED", "")) else f"{topic_word(activity).capitalize()} documentation and record control"
        observed_clause = f"The required record for the {activity} is missing."
        potential_clause = (
            "This may affect the ability to demonstrate that the required activity was performed in "
            "compliance with the applicable requirement."
        )
        if getattr(canonical, "downstream_action_present", False):
            downstream_clause = (
                " A subsequent action is separately reported in the finding; this does not establish that "
                "the missing record affected that action, and the action is not automatically considered "
                "improper. Whether it remains appropriately supported requires assessment."
            )
        else:
            downstream_clause = ""
        derived_effect = f"{observed_clause} {potential_clause}{downstream_clause}"
        derived_evidence_needed = (
            f"{derived_obj} execution/completion records, secondary or independent verification records, "
            "and the applicable requirement defining the activity"
        )
    # COMPARISON/MISMATCH finding (Section 6/7): the observation itself is
    # VERIFIED (both compared values and their discrepancy are stated facts,
    # not a report of someone's account) -- render it directly from the
    # canonical comparison event instead of the generic "<obj> was
    # reportedly <condition>" template, which incorrectly implies the
    # comparison itself is unconfirmed rather than merely its CAUSE.
    elif canonical and getattr(canonical, "comparison_type", None) and getattr(canonical, "comparison_left", None) and getattr(canonical, "comparison_right", None):
        from app.services.semantic_subject import render_comparison_sentence
        _measurement = getattr(canonical, "measurement", None)
        derived_obj = clean_noun[0].upper() + clean_noun[1:] if clean_noun and not clean_noun.startswith("UNKNOWN") else canonical.comparison_left.capitalize()
        derived_process = canonical.affected_process if (canonical.affected_process not in ("UNKNOWN", "NOT ESTABLISHED", "")) else f"{topic_cap} reconciliation and verification"
        _sentence = render_comparison_sentence(
            canonical.comparison_type, canonical.comparison_left, canonical.comparison_right,
            measurement_value=getattr(_measurement, "value", None),
            measurement_unit=getattr(_measurement, "unit", None),
            measurement_qualifier=getattr(_measurement, "qualifier", None),
            left_qualifier=getattr(canonical, "comparison_left_qualifier", None),
        ) or f"{derived_obj} differed from the compared value."
        _batch_suffix = f" for {canonical.comparison_batch_id}" if getattr(canonical, "comparison_batch_id", None) else ""
        if _sentence.endswith("."):
            _sentence = f"{_sentence[:-1]}{_batch_suffix}."
        # Downstream-impact phrasing is keyed by comparison SUBTYPE (Section
        # 11) -- a parameter mismatch's open question is whether the actual
        # PROCESS operated outside the approved parameter, distinct from a
        # calculation mismatch's generic "disposition/quality/reporting"
        # framing.
        if getattr(canonical, "comparison_subtype", None) == "PARAMETER_MISMATCH":
            derived_effect = (
                f"{_sentence} The impact assessment should determine whether the actual process operated "
                "outside the approved parameter and whether this affected process performance, product "
                "quality, batch disposition, or release."
            )
        else:
            derived_effect = (
                f"{_sentence} The scope and downstream consequences of this discrepancy require assessment, "
                "including whether it affected batch disposition, quality evaluation, or production reporting."
            )
        derived_evidence_needed = (
            f"Original records for the {canonical.comparison_left}, the {canonical.comparison_right}, and the "
            "applicable calculation/reference basis"
        )
    elif clean_noun and not clean_noun.startswith("UNKNOWN") and _delivery_receipt_conflict is not None:
        _clean_noun_cap = clean_noun[0].upper() + clean_noun[1:]
        derived_obj = _clean_noun_cap
        derived_process = f"{_clean_noun_cap} distribution and delivery control"
        derived_effect = (
            f"If affected personnel were required to act on {clean_noun} before receipt or acknowledgement "
            "was confirmed, the scope and compliance status of those actions may require assessment. This "
            "does not establish that delivery, receipt, or any related control failed."
        )
        derived_evidence_needed = (
            f"Independent receipt/access confirmation and acknowledgement records for {clean_noun}, and "
            "records of any activity performed by affected personnel relative to it"
        )
    elif clean_noun and not clean_noun.startswith("UNKNOWN") and expiry_then_use:
        # Structural expiry/use-date relationship (Section 18/19): distinct
        # from both the qualification-status branch and the generic
        # record-absence branch below -- the issue here isn't "we can't
        # confirm an activity happened" (the activity IS verified to have
        # happened), it's "the activity happened after a stated expiry",
        # which requires assessing VALIDITY, not existence. Never asserts
        # the measurement/use was invalid, that a product was affected, or
        # what caused the expiry to go unaddressed -- those are separate,
        # unestablished propositions (P2/P3/P4 in the spec this implements).
        derived_obj = clean_noun[0].upper() + clean_noun[1:]
        _expiry_domain = _extract_expiry_domain_word(request_finding_text)
        _process_domain_word = _expiry_domain.capitalize() if _expiry_domain else topic_cap
        derived_process = f"{_process_domain_word} and equipment-use control"
        derived_effect = (
            f"If {clean_noun} was used after the stated expiry as this finding describes, the activity "
            "performed may require assessment to determine whether it remained valid under the applicable "
            "requirements. This does not establish that the activity was invalid or that any downstream "
            "output was affected."
        )
        derived_evidence_needed = (
            f"Current {topic} status/history, any approved extension or waiver, and records of what "
            "relied on the activity performed after the stated expiry"
        )
    elif clean_noun and not clean_noun.startswith("UNKNOWN"):
        # tail = the subject phrase with its own leading "<topic> for "
        # prefix stripped (e.g. "training for the revised procedure" ->
        # "the revised procedure"). Only set when the subject actually HAS
        # that "<topic> for <tail>" activity/qualification shape.
        tail = split_topic_and_tail(clean_noun, topic)
        if tail:
            # Activity/qualification-shaped subject (e.g. "training for the
            # revised procedure") -- affected object is a role/qualification
            # status, built from role + topic + tail exactly ONCE each.
            derived_process = f"{topic_cap} and compliance control"
            derived_obj = build_affected_object_phrase(clean_noun, actor)
            actor_word = actor.capitalize() if actor else "Personnel"
            effect_subject = actor_word.lower() if actor_word == "Personnel" else f"the {actor_word.lower()}"
            derived_effect = (
                f"If required {topic} was not completed before the applicable effective date, {effect_subject} "
                f"may have proceeded without confirmed qualification for {tail}."
            )
            derived_evidence_needed = f"Approved {topic} completion and authorization record"
        elif re.search(r"\b(records?|logs?|documentation|report|checklist|form|sheet|completion)\b", clean_noun, re.IGNORECASE):
            derived_obj = clean_noun[0].upper() + clean_noun[1:]
            if "checklist" in clean_noun.lower():
                derived_process = "Inspection checklist completion and record control"
            elif "completion" in clean_noun.lower():
                derived_process = f"{topic_cap} monitoring and record control"
            else:
                derived_process = f"{topic_cap} monitoring and record control"
            condition = (
                canonical.deviation_condition if (canonical and canonical.deviation_condition not in (None, "UNKNOWN"))
                else "incomplete"
            )
            if "checklist" in clean_noun.lower() or "completion" in clean_noun.lower():
                derived_effect = (
                    f"Failure to complete {clean_noun.lower()} may prevent confirmation that required inspection "
                    "activities were performed during the affected period. Any product, process, or compliance "
                    "impact requires assessment against the applicable requirement and objective records."
                )
            else:
                derived_effect = (
                    f"Potential inability to confirm the required activity was performed: {condition} {topic} "
                    "records prevent confirmation that the required activity was performed during the "
                    "affected period. Any downstream impact requires assessment of actual conditions and "
                    "applicable requirements."
                )
            derived_evidence_needed = f"Independent {topic} verification record (e.g. instrument audit trail, electronic log, or supervisory review)"
        else:
            # Plain entity/physical-object subject (e.g. "equipment",
            # "balance BAL-014") with a structurally captured deviation
            # condition (e.g. "operated outside its validated range") that
            # is NOT itself a record/document -- the record-shaped
            # template's "records prevent confirmation" framing doesn't fit
            # a physical object, so this branch states the condition
            # directly instead of forcing it through record vocabulary.
            derived_obj = clean_noun[0].upper() + clean_noun[1:]
            condition = (
                canonical.deviation_condition if (canonical and canonical.deviation_condition not in (None, "UNKNOWN"))
                else "in a condition that has not been verified against applicable requirements"
            )
            # "Operation and validated-use control" / "validity of any
            # activity or output associated with that use" ONLY fits a
            # condition that actually describes something being
            # operated/used outside a range/limit/parameter (e.g. equipment
            # used after expiry, operated outside a validated range)
            _use_related_condition = bool(re.search(
                r"\b(?:operat\w*|us(?:e|ed|ing)|perform\w*)\s+outside\b|"
                r"\boutside\s+(?:its|the|their)\s+[\w\s]{0,20}?(?:range|limits?|parameters?|specification|tolerance|threshold)\b|"
                r"\b(?:validated|operating)\s+(?:range|limits?)\b",
                f"{request_finding_text} {condition} {observed_deviation}", re.IGNORECASE,
            ))
            if _use_related_condition:
                derived_process = f"{topic_cap} operation and validated-use control"
                derived_effect = (
                    f"{_reportedly_clause(derived_obj, condition)} This may require assessment of the validity or "
                    "acceptability of any activity or output associated with that use against the applicable "
                    "requirements. This does not establish that any specific output was invalid or that a "
                    "particular cause was responsible."
                )
                derived_evidence_needed = (
                    f"Approved validation/qualification records, operating logs, exception/deviation records, "
                    "and control system/interlock logs for {clean_noun}"
                )
            else:
                if canonical and getattr(canonical, "affected_process", None) and canonical.affected_process not in ("UNKNOWN", "NOT ESTABLISHED", ""):
                    derived_process = canonical.affected_process
                else:
                    derived_process = f"{derived_obj} control"
                derived_effect = (
                    f"{_reportedly_clause(derived_obj, condition)} Scope and downstream consequence require "
                    "assessment against the applicable requirement and objective records. This does not "
                    "establish a cause."
                )
                derived_evidence_needed = f"Applicable requirement/specification for {clean_noun}, and objective records relevant to the affected period"
    else:
        derived_process = canonical.affected_process if (canonical and getattr(canonical, "affected_process", None) and canonical.affected_process not in ("UNKNOWN", "NOT ESTABLISHED", "")) else "NOT ESTABLISHED"
        derived_obj = "NOT ESTABLISHED"
        derived_effect = "Scope and downstream consequence require auditor assessment."
        derived_evidence_needed = "Auditor assessment of records relevant to this finding"

    rel_change = (
        canonical.relevant_change if (canonical and canonical.relevant_change)
        else (
            "Revision of the inspection checklist" if ("revised" in request_finding_text.lower() and "checklist" in request_finding_text.lower())
            else ("Revision of the procedure" if ("revised" in request_finding_text.lower() or "revision" in request_finding_text.lower())
            else "NOT ESTABLISHED")
        )
    )

    derived_control = canonical.control_at_risk if (canonical and getattr(canonical, "control_at_risk", None)) else None

    impact = ImpactAssessment(
        status=ImpactStatus.IMPACT_REQUIRES_ASSESSMENT,
        affected_object=derived_obj,
        affected_period=degraded_period,
        finding_detected_period=getattr(canonical, "finding_detected_period", None),
        transaction_period=getattr(canonical, "transaction_period", None),
        process_at_risk=derived_process,
        control_at_risk=derived_control,
        relevant_change=rel_change,
        potential_effect=derived_effect,
        evidence_needed=derived_evidence_needed,
        impact_observed=observed_deviation,
        impact_inferred=None,
        impact_unknown="Full scope and downstream consequence require auditor assessment.",
        narrative="Assessment of scope and downstream consequences is required based on objective records.",
    )
    return impact, clean_noun, topic, actor


async def core_synthesis_node(state: AgentState) -> AgentState:
    """Run consolidated RCA, 5-Why, Impact, CAPA, and CA Draft synthesis."""
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    # Internal execution-state record (Phase 3 of the final hardening pass):
    # distinguishes WHICH path actually produced the result without
    # changing the public analysis_mode contract (still just "LLM" /
    # "DETERMINISTIC" / "DEGRADED" -- unaudited frontend consumers only
    # ever see that field). `source` narrows to PRIMARY_LLM/RECOVERY_LLM/
    # DETERMINISTIC; `validation_repairs` counts hypotheses/steps this
    # node itself corrected (contradiction/domain/provenance drops) so
    # "the LLM succeeded but needed repair" is distinguishable from "the
    # LLM's raw output survived untouched" without inventing a new public
    # enum value.
    synthesis_execution: dict = {
        "source": "PRIMARY_LLM",
        "recovery_used": False,
        "deterministic_fallback_used": False,
        "validation_repairs": 0,
        "validation_rejections": 0,
    }
    request = state["request"]
    evidence_ledger = state.get("evidence_ledger", [])
    quality = state.get("observation_quality")
    canonical = state.get("canonical_finding_state")
    # Captured before this pass overwrites root_cause -- non-None only on the
    # critic-send-back re-investigation loop's second (or later) pass, where
    # it drives the monotonic merge guard below (see
    # app.agent.causal_graph.merge_candidate_hypotheses).
    previous_root_cause = state.get("root_cause")
    # Computed once, used everywhere a leading hypothesis is derived below
    # (primary path, recovery path, and full deterministic fallback) so the
    # conflict-tie override is applied consistently regardless of which path
    # actually produced the hypotheses.
    has_unresolved_conflict = bool(
        canonical and any(getattr(c, "status", "UNRESOLVED") == "UNRESOLVED" for c in canonical.evidence_conflicts)
    )
    # NON-ACTIONABLE FAST PATH: if input is not actionable, return clean NOT_APPLICABLE/empty structures
    if canonical and not getattr(canonical, "is_actionable", True):
        trace.append(AgentTraceStep.ok("Core synthesis: non-actionable input — synthesis not applicable"))
        root_cause = RootCauseAnalysis(
            status=RootCauseStatus.NOT_APPLICABLE,
            category=None,
            statement=None,
            leading_hypothesis=None,
            candidate_hypotheses=[],
            narrative="No actionable audit observation provided for investigation.",
            evidence_status=EvidenceStatus.UNKNOWN,
            verification_needed="Not applicable — input is non-actionable.",
        )
        five_why = FiveWhyAnalysis(
            steps=[],
            is_complete=False,
            status_note="NOT APPLICABLE — NON-ACTIONABLE INPUT",
        )
        impact = ImpactAssessment(
            status=ImpactStatus.IMPACT_NOT_IDENTIFIED,
            areas=[],
            narrative=None,
            affected_object=None,
            affected_people=None,
            affected_period=None,
            process_at_risk=None,
            control_at_risk=None,
            relevant_change=None,
            potential_effect=None,
            evidence_needed=None,
        )
        capa = CapaAnalysis(
            status=CapaStatus.NO_CAPA_RECOMMENDATION_YET,
            potential_areas=[],
            recommended_investigation=[],
            conditional_actions=[],
        )
        ca_draft = CADraft(
            immediate_action="Not applicable — input is non-actionable.",
            root_cause="Not applicable — input is non-actionable.",
            root_cause_category="NOT_APPLICABLE",
            preventive_action="Not applicable — input is non-actionable.",
            impact_analysis="Not applicable — input is non-actionable.",
        )
        return {
            **state,
            "root_cause": root_cause,
            "five_why": five_why,
            "impact_assessment": impact,
            "capa_analysis": capa,
            "contributing_factors": [],
            "ca_draft": ca_draft,
            "analysis_mode": "DETERMINISTIC",
            "analysis_engine": "DETERMINISTIC",
            "synthesis_execution": {"source": "DETERMINISTIC", "non_actionable": True},
            "provider_used": None,
            "fallback_used": False,
            "provider_attempts": [],
            "investigation_plan": InvestigationPlan(areas=[], questions=[], evidence_to_collect=[]),
            "trace": trace,
            "errors": errors,
        }

    settings = get_settings()

    system_prompt = (settings.agent_prompts_dir / "system_prompt.txt").read_text(encoding="utf-8")
    template = (settings.agent_prompts_dir / "core_synthesis.txt").read_text(encoding="utf-8")
    recovery_template = (settings.agent_prompts_dir / "core_synthesis_recovery.txt").read_text(encoding="utf-8")

    observed_deviation = canonical.observed_deviation if canonical else "not extracted"
    mechanism_statement = canonical.immediate_mechanism if canonical else None
    mechanism_status = canonical.immediate_mechanism_status if canonical else "UNKNOWN"
    mechanism = MechanismInfo(
        statement=mechanism_statement,
        status=mechanism_status,
        polarity=None,
    )
    # Recompute polarity locally (cheap, deterministic) so the contradiction
    # guard below works even though CanonicalFindingState only persists the
    # statement/status, not the polarity classification.
    if mechanism_statement:
        from app.agent.causal_guard import classify_mechanism_polarity
        mechanism.polarity = classify_mechanism_polarity(mechanism_statement)

    # ONE canonical representation of the case passed to the LLM: the
    # structured evidence ledger + evidence conflicts + mechanism/deviation
    # already resolved by understand_finding_node, never the raw finding
    # text repeated a second/third time alongside it.
    #
    # Every claim carries its canonical id (C1, C2, ...) so the LLM's
    # hypotheses can cite supporting_claim_ids/contradicting_claim_ids by
    # reference instead of restating claim text -- both more compact and
    # the prerequisite for rejecting a hypothesis with invented provenance
    # (a claim ID the ledger never issued) or zero provenance at all.
    claim_ids = _assign_claim_ids(evidence_ledger)
    valid_claim_ids = {cid for cid, _ in claim_ids}
    compact_ledger = [
        {"id": cid, "claim": e.claim, "status": getattr(e.status, "value", str(e.status))}
        for cid, e in claim_ids[:6]
    ]
    compact_conflicts = [
        {"proposition": getattr(c, "proposition", str(c)), "status": getattr(c, "status", "UNRESOLVED")}
        for c in (canonical.evidence_conflicts if canonical else [])
    ]

    prompt = template.format(
        finding_text=request.finding_text,
        evidence_ledger_json=json.dumps(compact_ledger, default=str),
        observed_deviation=observed_deviation,
        immediate_mechanism=mechanism_statement or "none established",
        immediate_mechanism_status=mechanism_status,
        evidence_conflicts_json=json.dumps(compact_conflicts, default=str) if compact_conflicts else "[]",
    )

    provider_used: str | None = None
    fallback_used = False
    provider_attempts: list[str] = []

    async def _call(prompt_text: str, max_tokens: int, timeout_seconds: float, node: str, ctx_tokens: int | None = None) -> str:
        call_client = get_llm_client(timeout_seconds=timeout_seconds)
        return await call_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.1,
            response_format_json=True,
            max_tokens=max_tokens,
            num_ctx=ctx_tokens,
            node=node,
        )

    from app.services import llm_metrics
    import uuid
    _request_id = uuid.uuid4().hex[:8]
    synthesis_execution["request_id"] = _request_id
    llm_metrics.increment("llm_primary_attempted")

    try:
        # Section 1/2: compact schema/prompt, sufficient-not-excessive token
        # ceiling. A response that fills the whole budget and still parses
        # as valid, schema-conformant JSON is ACCEPTED below -- reaching the
        # ceiling is not itself treated as failure.
        raw = await _call(
            prompt, settings.ollama_core_synthesis_max_tokens,
            settings.ollama_primary_synthesis_timeout_seconds, "core_synthesis",
            settings.ollama_core_synthesis_num_ctx,
        )
        from app.services.llm_router import get_last_call_metadata
        _router_meta = get_last_call_metadata()
        provider_used = _router_meta.get("provider_used")
        fallback_used = bool(_router_meta.get("fallback_used", False))
        provider_attempts = list(_router_meta.get("provider_attempts", []))
        parsed, _ = parse_core_synthesis_output(raw)
        llm_metrics.increment("llm_primary_success")
        from app.services.ollama_client import get_last_call_metadata as _get_ollama_meta
        _ollama_meta = _get_ollama_meta()
        llm_metrics.record_execution(
            request_id=_request_id, node="core_synthesis", model=settings.ollama_model, phase="primary",
            elapsed_ms=_ollama_meta.get("elapsed_ms"), prompt_tokens=_ollama_meta.get("prompt_eval_count"),
            output_tokens=_ollama_meta.get("eval_count"),
        )
        source_text = build_source_text(request.finding_text, evidence_ledger)

        # Structured counter delta (Phase 4: no trace-message substring
        # inference) -- _parse_causal_fields's guards call
        # llm_metrics.record_validation_rejection/_repair directly at each
        # decision site; the before/after delta on those cumulative
        # counters is what synthesis_execution reports for this call.
        _rejections_before = llm_metrics.snapshot().get("validation_rejections_total", 0)
        _repairs_before = llm_metrics.snapshot().get("validation_repairs_total", 0)
        root_cause, five_why, contributing_factors = _parse_causal_fields(
            parsed, mechanism, evidence_ledger, source_text, observed_deviation, request.finding_text, trace,
            has_unresolved_conflict, claim_ids,
            canonical_subject=getattr(canonical, "finding_subject", None),
            canonical=canonical,
            semantic_context=state.get("canonical_semantic_context"),
        )
        synthesis_execution["validation_rejections"] = (
            llm_metrics.snapshot().get("validation_rejections_total", 0) - _rejections_before
        )
        synthesis_execution["validation_repairs"] = (
            llm_metrics.snapshot().get("validation_repairs_total", 0) - _repairs_before
        )

        # ---------------------------------------------------------------------
        # Impact & CAPA: constructed deterministically from the already-
        # synthesized root_cause/hypotheses, not requested from the LLM.
        # These are the same derivation functions the recovery/fallback paths
        # below already use (one canonical implementation, not a competing
        # LLM-authored one) -- the LLM's job is compact causal reasoning
        # (hypotheses, 5-Why), never CAPA/impact prose, which reduces both
        # the schema the model must fill and the hallucination surface area.
        # ---------------------------------------------------------------------
        impact, clean_noun, topic, actor = _derive_deterministic_impact(
            request.finding_text, canonical, observed_deviation,
            semantic_context=state.get("canonical_semantic_context"),
        )
        # NOTE: investigation_plan_override is deliberately left unset here
        # (stays None). build_deterministic_investigation_plan() generates
        # its OWN independent hypothesis set (its own H1..H4 with its own
        # statements) purely to derive CAPA potential_areas below -- those
        # IDs are NOT root_cause.candidate_hypotheses (the LLM's actual,
        # validated hypotheses) and must never be surfaced as investigation
        # questions, or the report would reference hypothesis IDs that don't
        # exist in the displayed hypothesis list (hypothesis-ID drift).
        # Investigation questions are derived downstream (final_evidence_
        # verification_node) directly from root_cause.candidate_hypotheses
        # whenever state's own investigation_plan.questions is empty --
        # the single consistent source of hypothesis-bound questions.
        from app.agent.nodes.plan_investigation_fallback import build_conditional_capa_actions
        from app.agent.nodes.plan_investigation_fallback import (
            build_deterministic_investigation_plan as _build_area_plan,
        )
        _, _area_plan = _build_area_plan(
            request.finding_text, evidence_ledger, canonical_subject=getattr(canonical, "finding_subject", None),
            canonical_state=canonical,
            semantic_context=state.get("canonical_semantic_context"),
        )
        # Phase 17: an explicit, structurally-grounded NO_ACTIONABLE_
        # UNCERTAINTY judgment already made upstream (Stage A --
        # app.agent.nodes.graph_investigation_planner) must not be clobbered
        # by this CAPA-potential-areas-only plan -- CAPA still gets its
        # areas from _area_plan.areas below regardless; only the
        # investigation_plan override is skipped.
        _existing_inv = state.get("investigation_plan")
        if getattr(_existing_inv, "status", None) == "NO_ACTIONABLE_UNCERTAINTY":
            investigation_plan_override = None
        else:
            investigation_plan_override = _area_plan
        capa = CapaAnalysis(
            status=CapaStatus.INVESTIGATION_REQUIRED,
            potential_areas=_area_plan.areas,
            recommended_investigation=[
                "Verify the candidate causal hypotheses above through objective record investigation."
            ],
            conditional_actions=build_conditional_capa_actions(root_cause.candidate_hypotheses, clean_noun, topic),
        )

        # CA DRAFT: built deterministically from the already-synthesized
        # root_cause/impact instead of asking the LLM to restate them a
        # second time. One canonical implementation.
        ca_draft = build_ca_draft(_derive_ca_draft_fields(root_cause, impact, canonical))

        analysis_mode = "LLM"
        analysis_engine = "LLM"
        synthesis_execution["source"] = "PRIMARY_LLM"
        synthesis_execution["recovery_used"] = False
        trace.append(AgentTraceStep.ok(
            "Core synthesis: primary LLM call produced verified causal analysis (RCA, 5-Why, Impact, CAPA)."
        ))

    except Exception as primary_exc:
        from app.services.ollama_client import get_last_call_metadata as get_last_ollama_metadata
        primary_ollama_meta = get_last_ollama_metadata()
        primary_failure_type = _classify_failure(primary_exc, primary_ollama_meta)
        llm_metrics.increment(f"llm_primary_{_failure_metric_suffix(primary_failure_type)}")
        llm_metrics.record_execution(
            request_id=_request_id, node="core_synthesis", model=settings.ollama_model, phase="primary",
            elapsed_ms=primary_ollama_meta.get("elapsed_ms"), failure_type=primary_failure_type,
        )
        logger.info(
            "node=core_synthesis failure_type=%s hit_output_limit=%s eval_count=%s max_output_tokens=%s exc=%s",
            primary_failure_type,
            primary_ollama_meta.get("hit_output_limit"),
            primary_ollama_meta.get("eval_count"),
            primary_ollama_meta.get("max_output_tokens"),
            primary_exc,
        )
        trace.append(AgentTraceStep.warn(
            f"Core synthesis primary call did not produce a usable result ({primary_failure_type}) — "
            "attempting compact JSON-first recovery before deterministic synthesis."
        ))

        from app.services.llm_router import get_last_call_metadata
        _router_meta = get_last_call_metadata()
        provider_used = None
        fallback_used = True
        provider_attempts = list(_router_meta.get("provider_attempts", []))

        source_text = build_source_text(request.finding_text, evidence_ledger)
        recovery_used = False
        root_cause: RootCauseAnalysis | None = None
        five_why: FiveWhyAnalysis | None = None
        contributing_factors: list[ContributingFactor] = []

        # -------------------------------------------------------------
        # Section 4: JSON-first recovery. Do NOT immediately abandon the
        # LLM path and fall to deterministic synthesis -- one compact,
        # causal-fields-only retry against a materially smaller prompt and
        # a smaller (but sufficient) output budget first.
        # -------------------------------------------------------------
        llm_metrics.increment("llm_recovery_attempted")
        try:
            recovery_prompt = recovery_template.format(
                finding_text=request.finding_text,
                evidence_ledger_json=json.dumps(_trim_evidence_for_recovery(claim_ids), default=str),
                observed_deviation=observed_deviation,
                immediate_mechanism=mechanism_statement or "none established",
                immediate_mechanism_status=mechanism_status,
            )
            recovery_raw = await _call(
                recovery_prompt, settings.ollama_recovery_max_tokens,
                settings.ollama_recovery_synthesis_timeout_seconds, "core_synthesis_recovery",
                settings.ollama_recovery_num_ctx,
            )
            recovery_parsed, _ = parse_core_synthesis_output(recovery_raw)
            llm_metrics.increment("llm_recovery_success")
            _recovery_ollama_meta = get_last_ollama_metadata()
            llm_metrics.record_execution(
                request_id=_request_id, node="core_synthesis_recovery", model=settings.ollama_model, phase="recovery",
                elapsed_ms=_recovery_ollama_meta.get("elapsed_ms"), prompt_tokens=_recovery_ollama_meta.get("prompt_eval_count"),
                output_tokens=_recovery_ollama_meta.get("eval_count"),
            )
            _rejections_before_recovery = llm_metrics.snapshot().get("validation_rejections_total", 0)
            _repairs_before_recovery = llm_metrics.snapshot().get("validation_repairs_total", 0)
            root_cause, five_why, contributing_factors = _parse_causal_fields(
                recovery_parsed, mechanism, evidence_ledger, source_text, observed_deviation, request.finding_text, trace,
                has_unresolved_conflict, claim_ids,
                canonical_subject=getattr(canonical, "finding_subject", None),
                canonical=canonical,
                semantic_context=state.get("canonical_semantic_context"),
            )
            synthesis_execution["source"] = "RECOVERY_LLM"
            synthesis_execution["recovery_used"] = True
            synthesis_execution["validation_rejections"] = (
                llm_metrics.snapshot().get("validation_rejections_total", 0) - _rejections_before_recovery
            )
            synthesis_execution["validation_repairs"] = (
                llm_metrics.snapshot().get("validation_repairs_total", 0) - _repairs_before_recovery
            )
            recovery_used = True
            trace.append(AgentTraceStep.ok(
                "Core Synthesis: compact JSON recovery call produced valid causal-reasoning JSON — "
                "analysis_mode remains LLM (a reduced-scope but genuine LLM analysis, not a provider "
                "failure). Impact/CAPA below use deterministic derivation since the recovery schema is "
                "causal-reasoning only."
            ))
        except Exception as recovery_exc:
            recovery_meta = get_last_ollama_metadata()
            recovery_failure_type = _classify_failure(recovery_exc, recovery_meta)
            llm_metrics.increment(f"llm_recovery_{_failure_metric_suffix(recovery_failure_type)}")
            llm_metrics.record_execution(
                request_id=_request_id, node="core_synthesis_recovery", model=settings.ollama_model, phase="recovery",
                elapsed_ms=recovery_meta.get("elapsed_ms"), failure_type=recovery_failure_type,
            )
            logger.info(
                "node=core_synthesis_recovery failure_type=%s recovery=DETERMINISTIC_SYNTHESIS exc=%s",
                recovery_failure_type,
                recovery_exc,
            )
            trace.append(AgentTraceStep.ok(
                f"Core synthesis recovery call also failed ({recovery_failure_type}) — transitioning to "
                "deterministic evidence-grounded synthesis."
            ))

        # -------------------------------------------------------------
        # Deterministic derivations shared by both outcomes: impact/CAPA
        # are always derived this way after a primary-call failure (the
        # recovery schema deliberately excludes them), and the
        # topic/clean_noun/actor values feed CAPA regardless of whether
        # root_cause/five_why came from recovery or full deterministic
        # synthesis below.
        # -------------------------------------------------------------
        impact, clean_noun, topic, actor = _derive_deterministic_impact(
            request.finding_text, canonical, observed_deviation,
            semantic_context=state.get("canonical_semantic_context"),
        )

        from app.agent.nodes.plan_investigation_fallback import (
            build_conditional_capa_actions,
            build_deterministic_investigation_plan,
        )
        _, fallback_plan = build_deterministic_investigation_plan(
            request.finding_text, evidence_ledger, canonical_subject=getattr(canonical, "finding_subject", None), canonical_state=canonical,
            semantic_context=state.get("canonical_semantic_context"),
        )
        # Phase 17: do not clobber an upstream NO_ACTIONABLE_UNCERTAINTY
        # judgment -- see the matching guard in the primary-success branch
        # above for the full rationale.
        investigation_plan_override = None if getattr(state.get("investigation_plan"), "status", None) == "NO_ACTIONABLE_UNCERTAINTY" else fallback_plan

        if recovery_used and root_cause is not None and five_why is not None:
            analysis_mode = "LLM"
            analysis_engine = "LLM"
            capa = CapaAnalysis(
                status=CapaStatus.INVESTIGATION_REQUIRED,
                potential_areas=fallback_plan.areas,
                recommended_investigation=[
                    "Verify the candidate causal hypotheses above through objective record investigation."
                ],
                conditional_actions=build_conditional_capa_actions(root_cause.candidate_hypotheses, clean_noun, topic),
            )
        else:
            # -----------------------------------------------------------
            # Full deterministic evidence-grounded synthesis (Section 16):
            # reached only when BOTH the primary call and the compact
            # recovery retry failed to produce usable, valid JSON -- never
            # merely because a response consumed its configured token
            # budget while still being complete and valid.
            # -----------------------------------------------------------
            analysis_mode = "DETERMINISTIC"
            analysis_engine = "DETERMINISTIC"
            llm_metrics.increment("deterministic_fallback")
            synthesis_execution["source"] = "DETERMINISTIC"
            synthesis_execution["deterministic_fallback_used"] = True
            # Same _request_id as the primary/recovery attempts above
            # (Phase 7 correlation) -- a caller correlating recent_executions()
            # by request_id sees the full primary -> recovery -> fallback
            # chain for one synthesis call, not three unrelated ids.
            llm_metrics.record_execution(
                request_id=_request_id, node="core_synthesis", model=settings.ollama_model, phase="fallback",
            )

            from app.agent.nodes.five_why_fallback import build_deterministic_five_why
            five_why = build_deterministic_five_why(
                request.finding_text, evidence_ledger, canonical_subject=getattr(canonical, "finding_subject", None),
                semantic_context=state.get("canonical_semantic_context"),
            )
            fallback_hyps, fallback_plan = build_deterministic_investigation_plan(
                request.finding_text, evidence_ledger, canonical_subject=getattr(canonical, "finding_subject", None), canonical_state=canonical,
                semantic_context=state.get("canonical_semantic_context"),
            )
            # Phase 17: same NO_ACTIONABLE_UNCERTAINTY guard as above.
            investigation_plan_override = None if getattr(state.get("investigation_plan"), "status", None) == "NO_ACTIONABLE_UNCERTAINTY" else fallback_plan
            contributing_factors = []
            llm_metrics.increment("deterministic_hypotheses_generated", len(fallback_hyps))

            from app.agent.causal_graph import evaluate_root_cause_eligibility, select_authoritative_leading_hypothesis
            for h in fallback_hyps:
                el, supp, _, _, c_lvl, promo = evaluate_root_cause_eligibility(
                    h,
                    evidence_items=evidence_ledger,
                    conflicts=canonical.evidence_conflicts if canonical else None,
                    referenced_docs=canonical.referenced_documents if canonical else None,
                    canonical_state=canonical,
                )
                if promo and supp == SupportLevel.SUPPORTED:
                    h.status = "SUPPORTED"
                    h.causal_level = c_lvl

            lead_id, lead_mode, authoritative_rc_status, lead_rationale = select_authoritative_leading_hypothesis(
                fallback_hyps,
                conflicts=canonical.evidence_conflicts if canonical else None,
                evidence_ledger=evidence_ledger,
            )

            rc_basis = (
                "Available evidence contains conflicting reported statements regarding completion. Objective records are required to determine the actual state."
                if has_unresolved_conflict
                else ("Objective records establish the verified causal factor for this finding."
                      if authoritative_rc_status in (RootCauseStatus.ESTABLISHED, RootCauseStatus.SUPPORTED)
                      else "Root cause is not established from the available evidence alone; objective records are required to confirm the causal mechanism.")
            )

            lead_status_literal = "SELECTED" if lead_mode == "SELECTED" else ("TIED" if lead_mode == "TIED" else "NONE")
            root_cause = RootCauseAnalysis(
                status=authoritative_rc_status,
                category="TO_BE_CONFIRMED",
                candidate_hypotheses=fallback_hyps,
                leading_hypothesis=lead_id if authoritative_rc_status != RootCauseStatus.NOT_ESTABLISHED else None,
                leading_hypothesis_status=lead_status_literal,
                leading_hypothesis_rationale=lead_rationale,
                root_cause_basis=rc_basis,
                evidence_required=["Auditor investigation and objective records required to confirm root cause."],
                narrative=(
                    "The available evidence establishes the observed condition and verified records confirm the underlying causal mechanism."
                    if authoritative_rc_status in (RootCauseStatus.ESTABLISHED, RootCauseStatus.SUPPORTED)
                    else "The available evidence establishes the observed condition but does not establish why it occurred. Auditor investigation is required to verify the candidate causal hypotheses."
                ),
            )
            from app.agent.analytical_validator import apply_conflict_tie_override
            apply_conflict_tie_override(root_cause, has_unresolved_conflict)

            from app.agent.recurrence_guard import build_recurrence_rationale, detect_recurrence
            recurrence_info = detect_recurrence(request.finding_text)
            if recurrence_info.is_recurring:
                root_cause.risk_of_recurrence = "HIGH"
                root_cause.risk_of_recurrence_rationale = build_recurrence_rationale(recurrence_info)

            # CAPA areas reuse the same dynamically-derived, finding-grounded
            # investigation plan used for hypotheses above rather than a fixed
            # universal category list. conditional_actions map each surviving
            # hypothesis to an organizational corrective action (never an
            # evidence source dressed up as an action) — CAPA stays pending on
            # root cause even in deterministic mode, never a final action.
            capa = CapaAnalysis(
                status=CapaStatus.INVESTIGATION_REQUIRED,
                potential_areas=fallback_plan.areas,
                recommended_investigation=[
                    "Verify the candidate causal hypotheses above through objective record investigation."
                ],
                conditional_actions=build_conditional_capa_actions(fallback_hyps, clean_noun, topic),
            )

        ca_draft = build_ca_draft(_derive_ca_draft_fields(root_cause, impact, canonical))

    from app.agent.causal_graph import capture_epistemic_snapshot, merge_candidate_hypotheses

    if previous_root_cause is not None and root_cause is not None:
        root_cause.candidate_hypotheses = merge_candidate_hypotheses(
            previous_root_cause.candidate_hypotheses, root_cause.candidate_hypotheses
        )

    snapshot_history = list(state.get("epistemic_snapshot_history", []))
    snapshot_history.append(capture_epistemic_snapshot(root_cause, canonical))

    return {
        **state,
        "root_cause": root_cause,
        "five_why": five_why,
        "impact_assessment": impact,
        "capa_analysis": capa,
        "contributing_factors": contributing_factors,
        "ca_draft": ca_draft,
        "analysis_mode": analysis_mode,
        "analysis_engine": analysis_engine,
        "synthesis_execution": synthesis_execution,
        "provider_used": provider_used,
        "fallback_used": fallback_used,
        "provider_attempts": provider_attempts,
        "investigation_plan": investigation_plan_override if investigation_plan_override is not None else state.get("investigation_plan"),
        "epistemic_snapshot_history": snapshot_history,
        "trace": trace,
        "errors": errors,
    }
