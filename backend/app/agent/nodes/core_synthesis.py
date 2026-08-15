"""Consolidated Node: core_synthesis

Replaces serial rca -> impact -> capa -> ca_draft LLM calls with a single
high-reasoning, structured JSON call. Executes RCA, 5-Why, Impact, CAPA, and
CA Draft simultaneously in ONE LLM round-trip (~8-12s vs 120s).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.agent.analytical_validator import leading_hypothesis_confidence, select_leading_hypothesis
from app.agent.causal_guard import (
    MechanismInfo,
    hypothesis_contradicts_mechanism,
    hypothesis_contradicts_verified_completion,
    is_circular_why_answer,
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
    CandidateHypothesis,
    CapaAnalysis,
    CapaStatus,
    ConditionalCapaAction,
    ContributingFactor,
    EvidenceStatus,
    FiveWhyAnalysis,
    FiveWhyStep,
    ImpactAssessment,
    ImpactStatus,
    InvestigationQuestion,
    RootCauseAnalysis,
    RootCauseStatus,
)
from app.services.llm_client import LLMError, get_llm_client
from app.services.llm_json import parse_llm_json
from app.services.taxonomy import coerce_category

logger = logging.getLogger(__name__)


async def core_synthesis_node(state: AgentState) -> AgentState:
    """Run consolidated RCA, 5-Why, Impact, CAPA, and CA Draft synthesis."""
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    request = state["request"]
    evidence_ledger = state.get("evidence_ledger", [])
    quality = state.get("observation_quality")
    canonical = state.get("canonical_finding_state")

    settings = get_settings()
    client = get_llm_client()

    system_prompt = (settings.agent_prompts_dir / "system_prompt.txt").read_text(encoding="utf-8")
    template = (settings.agent_prompts_dir / "core_synthesis.txt").read_text(encoding="utf-8")

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

    prompt = template.format(
        finding_text=request.finding_text,
        evidence_ledger_json=json.dumps([e.model_dump() for e in evidence_ledger], default=str),
        observation_quality=quality.status.value if quality else "SUFFICIENT",
        missing_info=", ".join(quality.missing_information) if (quality and quality.missing_information) else "None",
        observed_deviation=observed_deviation,
        immediate_mechanism=mechanism_statement or "none established — the finding does not directly state HOW the deviation happened",
        immediate_mechanism_status=mechanism_status,
    )

    provider_used: str | None = None
    fallback_used = False
    provider_attempts: list[str] = []

    try:
        raw = await client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format_json=True,
            max_tokens=1536,
        )
        # Provider metadata (infrastructure only — no analytical meaning):
        # populated by the LLM router when get_llm_client() returned it
        # (the default; empty when llm_provider="ollama" bypasses the
        # router). Read immediately after our own await so this reflects
        # THIS call, not a concurrent one (see llm_router's ContextVar note).
        from app.services.llm_router import get_last_call_metadata
        _router_meta = get_last_call_metadata()
        provider_used = _router_meta.get("provider_used")
        fallback_used = bool(_router_meta.get("fallback_used", False))
        provider_attempts = list(_router_meta.get("provider_attempts", []))
        parsed = parse_llm_json(raw)
        source_text = build_source_text(request.finding_text, evidence_ledger)

        # ---------------------------------------------------------------------
        # 1. Parse Root Cause & 5-Why
        # ---------------------------------------------------------------------
        raw_rc = parsed.get("root_cause", {})
        rc_status_str = str(raw_rc.get("status", "NOT_ESTABLISHED")).upper()
        try:
            rc_status = RootCauseStatus(rc_status_str)
        except ValueError:
            rc_status = RootCauseStatus.NOT_ESTABLISHED

        rc_category = coerce_category(raw_rc.get("category"))
        if rc_status == RootCauseStatus.NOT_ESTABLISHED:
            rc_category = "TO_BE_CONFIRMED"

        valid_hyp_statuses = {"POSSIBLE", "SUPPORTED", "REFUTED", "UNVERIFIED"}
        verified_facts = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.VERIFIED]
        cand_hypotheses = []
        for ch in raw_rc.get("candidate_hypotheses", []):
            if isinstance(ch, dict):
                rank_raw = str(ch.get("relevance_rank", "HIGH")).upper()
                rank = rank_raw if rank_raw in ("HIGH", "MEDIUM", "LOW") else "HIGH"
                status_raw = str(ch.get("status", "POSSIBLE")).upper()
                hyp_status = status_raw if status_raw in valid_hyp_statuses else "POSSIBLE"
                statement = clean_structured_leak(ch.get("statement", ""))
                if hyp_status == "REFUTED":
                    trace.append(AgentTraceStep.warn(
                        f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — directly contradicted by evidence"
                    ))
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
                    continue
                # REDUNDANCY FILTER: a hedged restatement of the already-established
                # mechanism is not a new hypothesis about its CAUSE.
                if mechanism_already_names_generic_hypothesis(statement, mechanism):
                    trace.append(AgentTraceStep.warn(
                        f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — restates the already-"
                        "established mechanism as an open possibility instead of reasoning about its cause"
                    ))
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
                    continue
                cand_hypotheses.append(CandidateHypothesis(
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
                ))

        root_cause = RootCauseAnalysis(
            status=rc_status,
            category=rc_category,
            statement=clean_structured_leak(raw_rc.get("statement")) or None,
            leading_hypothesis=clean_structured_leak(raw_rc.get("leading_hypothesis")) or None,
            candidate_hypotheses=cand_hypotheses,
            risk_of_recurrence=raw_rc.get("risk_of_recurrence", "NOT_ASSESSABLE"),
            narrative=clean_structured_leak(raw_rc.get("narrative")) or "The available evidence establishes the observed condition but does not establish why it occurred.",
        )


        raw_fw = parsed.get("five_why", {})
        fw_steps = []
        valid_fw_statuses = {"VERIFIED", "SUPPORTED", "REPORTED", "REPORTED_STATEMENT", "REPORTED_UNVERIFIED", "INFERRED", "UNKNOWN", "REQUIRES_EVIDENCE", "NOT_ESTABLISHED"}
        for s in raw_fw.get("steps", []):
            if isinstance(s, dict):
                st_raw = str(s.get("status", "UNKNOWN")).upper()
                st = st_raw if st_raw in valid_fw_statuses else ("REPORTED_UNVERIFIED" if "REPORT" in st_raw else "UNKNOWN")
                question = clean_structured_leak(s.get("question", ""))
                answer = clean_structured_leak(s.get("answer", ""))

                # CAUSAL COHERENCE GUARD (structural, not finding-specific):
                # reject a step whose answer just restates its own question,
                # or restates the immediately preceding step's answer — the
                # chain must advance causally, never circle in place.
                if is_reporting_why_question(question):
                    trace.append(AgentTraceStep.warn(
                        f"Core Synthesis: 5-Why step {len(fw_steps) + 1} asked about reporting behavior "
                        "instead of causal mechanism — truncating chain here"
                    ))
                    break

                # MECHANISM-REOPENING GUARD: a question that re-litigates
                # whether the already-established mechanism occurred (asking
                # about the OPPOSITE polarity, e.g. "was it performed but
                # undocumented?" once non-performance is confirmed) reopens
                # a resolved distinction instead of asking why it occurred.
                if question_reopens_mechanism(question, mechanism):
                    trace.append(AgentTraceStep.warn(
                        f"Core Synthesis: 5-Why step {len(fw_steps) + 1} question reopened the "
                        f"already-established mechanism ({mechanism.statement!r}) — truncating chain here"
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
                elif len(fw_steps) >= 1 and restates_observation(answer, observed_deviation):
                    trace.append(AgentTraceStep.warn(
                        f"Core Synthesis: 5-Why step {len(fw_steps) + 1} answer merely restated the "
                        "original observation instead of explaining it — truncating chain here"
                    ))
                    answer = "The available evidence does not establish a cause beyond the observation itself."
                    st = "UNKNOWN"
                elif ungrounded_entities(answer, source_text):
                    trace.append(AgentTraceStep.warn(
                        f"Core Synthesis: 5-Why step {len(fw_steps) + 1} answer referenced an "
                        "ungrounded entity/number — truncating chain here"
                    ))
                    answer = "The available evidence does not establish this — the answer could not be traced to this finding's evidence."
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

        # ---------------------------------------------------------------------
        # 2. Parse Impact Assessment
        # ---------------------------------------------------------------------
        raw_imp = parsed.get("impact_assessment", {})
        impact = ImpactAssessment(
            status=ImpactStatus.IMPACT_REQUIRES_ASSESSMENT,
            areas=[clean_structured_leak(x) for x in raw_imp.get("areas", []) if clean_structured_leak(x)],
            narrative=clean_structured_leak(raw_imp.get("narrative")) or "Impact requires assessment — scope unconfirmed.",
            affected_object=clean_structured_leak(raw_imp.get("affected_object")),
            affected_people=clean_structured_leak(raw_imp.get("affected_people")),
            affected_period=clean_structured_leak(raw_imp.get("affected_period")),
            process_at_risk=clean_structured_leak(raw_imp.get("process_at_risk")),
            relevant_change=clean_structured_leak(raw_imp.get("relevant_change")),
            potential_effect=clean_structured_leak(raw_imp.get("potential_effect")),
            evidence_needed=clean_structured_leak(raw_imp.get("evidence_needed")),
        )

        # ---------------------------------------------------------------------
        # 3. Parse CAPA Analysis
        # ---------------------------------------------------------------------
        raw_capa = parsed.get("capa", {})
        cond_actions = []
        for ca in raw_capa.get("conditional_actions", []):
            if isinstance(ca, dict):
                # A conditional action is by construction the "systemic
                # action pending confirmation" representation (Phase 4): the
                # root cause isn't established, so this can never be a
                # completed/definitive corrective action -- deterministic,
                # not dependent on the LLM remembering to classify it.
                cond_actions.append(ConditionalCapaAction(
                    if_cause_confirmed=clean_structured_leak(ca.get("if_cause_confirmed", "")),
                    recommended_action=clean_structured_leak(ca.get("recommended_action", "")),
                    action_type="SYSTEMIC_ACTION",
                    verification_method=clean_structured_leak(ca.get("verification_method")) or None,
                ))

        capa = CapaAnalysis(
            status=CapaStatus.INVESTIGATION_REQUIRED,
            potential_areas=[clean_structured_leak(x) for x in raw_capa.get("potential_areas", []) if clean_structured_leak(x)],
            recommended_investigation=[clean_structured_leak(x) for x in raw_capa.get("recommended_investigation", []) if clean_structured_leak(x)],
            conditional_actions=cond_actions,
        )

        # ---------------------------------------------------------------------
        # 3b. Parse Contributing Factors (dynamically derived, not a fixed
        # "not established" filler — an empty list is a correct outcome when
        # the finding supports none).
        # ---------------------------------------------------------------------
        contributing_factors: list[ContributingFactor] = []
        valid_cf_status = {"POSSIBLE_UNCONFIRMED", "VERIFIED"}
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
            cf_evidence_status = str(cf.get("evidence_status", "INFERRED")).upper()
            if cf_evidence_status not in {s.value for s in EvidenceStatus}:
                cf_evidence_status = "INFERRED"
            cf_status = str(cf.get("status", "POSSIBLE_UNCONFIRMED")).upper()
            if cf_status not in valid_cf_status or cf_evidence_status != "VERIFIED":
                cf_status = "POSSIBLE_UNCONFIRMED"
            contributing_factors.append(ContributingFactor(
                description=description,
                evidence_status=EvidenceStatus(cf_evidence_status),
                status=cf_status,
                rationale=clean_structured_leak(cf.get("rationale")) or None,
                evidence_required=clean_structured_leak(cf.get("evidence_required")) or None,
            ))

        # ---------------------------------------------------------------------
        # 4. Parse CA Draft
        # ---------------------------------------------------------------------
        raw_draft = parsed.get("ca_draft", {})
        ca_draft = build_ca_draft({
            "immediate_action": clean_structured_leak(raw_draft.get("immediate_action")) or "Review affected activity and contain items.",
            "root_cause": clean_structured_leak(raw_draft.get("root_cause")) or f"NOT_ESTABLISHED — {root_cause.narrative}",
            "root_cause_category": "TO_BE_CONFIRMED" if root_cause.status == RootCauseStatus.NOT_ESTABLISHED else root_cause.category,
            "preventive_action": clean_structured_leak(raw_draft.get("preventive_action")) or "Strengthen control process once root cause is confirmed.",
            "impact_analysis": clean_structured_leak(raw_draft.get("impact_analysis")) or "Impact scope pending auditor verification.",
        })

        # ---------------------------------------------------------------------
        # 5. Production-Grade Post-Processing Guards (Rules 4, 18, 19, 28)
        # ---------------------------------------------------------------------

        # CERTAINTY MONOTONICITY & CATEGORY LOCK (Rule 19 & Rule 28)
        # If root cause is NOT_ESTABLISHED, category MUST be TO_BE_CONFIRMED.
        if root_cause.status == RootCauseStatus.NOT_ESTABLISHED:
            root_cause.category = "TO_BE_CONFIRMED"
            ca_draft.root_cause_category = "TO_BE_CONFIRMED"

        # DOMAIN LEAKAGE & CAUSAL ANCHOR GUARD (Rules 1, 4 & 5)
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

        # Section 4 & 5: Ensure candidate_hypotheses is NEVER empty
        if not root_cause.candidate_hypotheses:
            from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
            fallback_hyps, _ = build_deterministic_investigation_plan(request.finding_text, evidence_ledger)
            root_cause.candidate_hypotheses = fallback_hyps
            trace.append(AgentTraceStep.warn("Core Synthesis: generated fallback candidate hypotheses from finding text"))

        # LEADING HYPOTHESIS ENFORCEMENT: root_cause.status staying
        # NOT_ESTABLISHED must not mean the report goes silent about which
        # candidate is best-evidenced -- deterministic, not dependent on the
        # LLM remembering to fill leading_hypothesis. select_leading_hypothesis
        # is the single source of truth for this rule (also used by the
        # analytical validator) -- it deliberately returns None when
        # hypotheses are equally plausible rather than forcing a pick.
        if not root_cause.leading_hypothesis:
            root_cause.leading_hypothesis = select_leading_hypothesis(root_cause.candidate_hypotheses)
            if root_cause.leading_hypothesis:
                root_cause.confidence = leading_hypothesis_confidence(
                    root_cause.candidate_hypotheses, root_cause.leading_hypothesis
                )

        # REPORTED STATEMENT ATTRIBUTION GUARD (Rule 3)
        # If narrative claims carelessness/negligence from reported statements, flag & sanitize
        if "careless" in root_cause.narrative.lower() or "negligen" in root_cause.narrative.lower():
            root_cause.narrative = (
                "Personnel reported attribution to human factors/carelessness. "
                "This statement is unverified and does not establish objective root cause."
            )

        analysis_mode = "LLM"
        trace.append(AgentTraceStep.ok("Consolidated core synthesis completed and validated against production rules."))


    except Exception as exc:
        logger.warning("Core synthesis LLM call failed or produced invalid JSON: %s", exc)
        trace.append(AgentTraceStep.warn(f"Core synthesis error: {exc} — using safe deterministic fallback"))
        errors.append(f"Synthesis error: {exc}")
        analysis_mode = "DEGRADED"
        # The router raises only after exhausting every configured provider,
        # recording each attempt as it goes -- surface that even though the
        # overall call failed, so degraded-mode reporting can still show
        # which providers were actually tried (never fabricated: empty when
        # llm_provider="ollama" bypassed the router, or when this exception
        # came from JSON parsing rather than the provider call itself).
        from app.services.llm_router import get_last_call_metadata
        _router_meta = get_last_call_metadata()
        provider_used = None
        fallback_used = True
        provider_attempts = list(_router_meta.get("provider_attempts", []))

        # Section 10: Deterministic fallback 5-Why — never outputs "Analysis unavailable"
        from app.agent.nodes.five_why_fallback import build_deterministic_five_why
        from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan

        five_why = build_deterministic_five_why(request.finding_text, evidence_ledger)
        fallback_hyps, fallback_plan = build_deterministic_investigation_plan(request.finding_text, evidence_ledger)
        # Degraded mode is marked explicitly (not silently blended with a
        # normal analytical result) and never fabricates contributing
        # factors — an LLM failure preserves already-extracted facts (via
        # the deterministic fallbacks above) but must not pretend to have
        # done causal reasoning it didn't do.
        contributing_factors = []

        # Safe non-hallucinating defaults for other sections
        fact_claims = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.VERIFIED]
        from app.services.semantic_subject import resolve_deviation
        resolved = resolve_deviation(request.finding_text, fact_claims)
        clean_noun = resolved.subject or "UNKNOWN — no affected object could be isolated from the finding text"

        leading_hypothesis = select_leading_hypothesis(fallback_hyps)

        root_cause = RootCauseAnalysis(
            status=RootCauseStatus.NOT_ESTABLISHED,
            category="TO_BE_CONFIRMED",
            candidate_hypotheses=fallback_hyps,
            leading_hypothesis=leading_hypothesis,
            narrative=(
                "DEGRADED MODE — LLM-based causal analysis was unavailable for this finding. "
                "The available evidence establishes the observed condition but full causal "
                "reasoning was not performed; the hypotheses below were generated "
                "deterministically from the finding text and require auditor review."
            ),
        )
        # Degraded-mode impact: only fields directly derivable without LLM
        # reasoning are populated; everything else is honestly UNKNOWN rather
        # than a fabricated generic claim about "execution logs".
        impact = ImpactAssessment(
            status=ImpactStatus.IMPACT_REQUIRES_ASSESSMENT,
            affected_object=clean_noun,
            affected_period="Timeframe stated in finding or requires confirmation",
            process_at_risk="NOT ESTABLISHED — degraded mode, requires auditor assessment",
            relevant_change="NOT ESTABLISHED",
            potential_effect="NOT ESTABLISHED — degraded mode, LLM-based impact analysis was unavailable",
            evidence_needed="Auditor assessment of records/logs relevant to the affected object above",
            narrative=(
                "DEGRADED MODE — LLM-based impact analysis was unavailable for this finding. "
                "Auditor assessment is required to determine actual impact."
            ),
        )

        # CAPA areas reuse the same dynamically-derived, finding-grounded
        # investigation plan used for hypotheses above rather than a fixed
        # universal category list. conditional_actions map each surviving
        # hypothesis to what verification it needs — CAPA stays pending on
        # root cause even in degraded mode, never a final action.
        capa = CapaAnalysis(
            status=CapaStatus.INVESTIGATION_REQUIRED,
            potential_areas=fallback_plan.areas,
            recommended_investigation=[
                "DEGRADED MODE — LLM-based CAPA analysis was unavailable; "
                "verify the causal hypotheses above through manual investigation."
            ],
            conditional_actions=[
                ConditionalCapaAction(
                    if_cause_confirmed=f"If {h.id} ({h.name.replace('_', ' ').lower()}) is confirmed",
                    recommended_action=f"Address the confirmed cause via {h.evidence_needed}; define a targeted systemic action once confirmed.",
                    action_type="SYSTEMIC_ACTION",
                    verification_method=f"Re-audit {h.evidence_needed} after the action is implemented to confirm recurrence does not continue.",
                )
                for h in fallback_hyps
            ],
        )
        ca_draft = build_ca_draft({
            "immediate_action": f"Review and, where procedurally permitted, correct the affected {clean_noun.lower()}.",
            "root_cause": "NOT_ESTABLISHED — DEGRADED MODE, LLM-based causal analysis was unavailable.",
            "root_cause_category": "TO_BE_CONFIRMED",
            "preventive_action": "Pending confirmation of root cause — auditor investigation required.",
            "impact_analysis": "DEGRADED MODE — impact scope requires auditor assessment.",
        })



    return {
        **state,
        "root_cause": root_cause,
        "five_why": five_why,
        "impact_assessment": impact,
        "capa_analysis": capa,
        "contributing_factors": contributing_factors,
        "ca_draft": ca_draft,
        "analysis_mode": analysis_mode,
        "provider_used": provider_used,
        "fallback_used": fallback_used,
        "provider_attempts": provider_attempts,
        "trace": trace,
        "errors": errors,
    }
