"""Node 8: critic

Self-review of the entire analysis. Checks evidence grounding, root cause
validity, CAPA appropriateness, impact claims, and 5-Why integrity.

This is the pipeline's SEMANTIC grounding layer, not just an advisory
review: the deterministic regex/keyword guards in grounding_guard.py catch
cheap, obvious cases (invented entities, a fixed list of generic QMS
domains) for free, but they cannot generalize to every possible fabricated
mechanism a model might invent -- a hallucinated hypothesis can use only
ordinary, already-grounded vocabulary (e.g. "the operator miscommunicated
the status to the auditor" for a calibration-label finding that never
mentions any auditor interaction) and still slip past keyword matching. The
critic uses the model's own semantic judgment to catch that class of error,
and — unlike the old version of this node — actually ACTS on what it finds:
flagged hypotheses are removed and a flagged root-cause narrative is
replaced, not just logged as feedback for a human to notice later.

Can:
  - Approve the analysis
  - Reject with specific corrections needed (but continue to report generation)
  - Request another investigation round (capped by agent_max_critic_iterations)
  - Strip specific hypotheses/narrative it identifies as ungrounded
"""

from __future__ import annotations

import json
import logging

from app.agent.grounding_guard import SAFE_ROOT_CAUSE_FALLBACK
from app.agent.state import AgentState
from app.config import get_settings
from app.models.agent import AgentTraceStep
from app.services.llm_client import LLMError, get_llm_client
from app.services.llm_json import parse_llm_json

logger = logging.getLogger(__name__)


async def critic_node(state: AgentState) -> AgentState:
    """Self-review the analysis for evidence grounding and correctness."""
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    critic_iteration = state.get("critic_iteration", 0)
    evidence_ledger = state.get("evidence_ledger", [])
    root_cause = state.get("root_cause")
    five_why = state.get("five_why")
    settings = get_settings()

    # DETERMINISTIC PRE-CRITIC QUALITY GATE: the critic is an exception-path
    # quality-control mechanism, not a mandatory step on every successful
    # synthesis (it costs a full extra LLM round-trip -- ~15-20s on this
    # hardware -- for no benefit on an already-clean result). It only runs
    # when a concrete, structural concern survives everything core_synthesis
    # already checked deterministically:
    #   - a fabricated mechanism/entity core_synthesis's own guards missed
    #     (ungrounded_entities / mentions_unsupported_domain -- the same
    #     checks core_synthesis applies to its own output, re-run here
    #     because THIS is the last point before the report is finalized)
    #   - a required analytical field came back empty when it shouldn't
    #     (no candidate_hypotheses, no five_why steps) -- core_synthesis's
    #     own fallbacks should already prevent this, so seeing it here means
    #     something upstream genuinely misbehaved and deserves a second look
    # A clean result skips straight to report generation.
    from app.agent.grounding_guard import build_source_text, mentions_unsupported_domain, ungrounded_entities

    source_text = build_source_text(state["request"].finding_text, evidence_ledger)

    def _flagged(text: str | None) -> bool:
        if not text:
            return False
        return bool(ungrounded_entities(text, source_text)) or mentions_unsupported_domain(text, source_text)

    has_ungrounded = False
    if root_cause:
        if _flagged(root_cause.narrative) or _flagged(root_cause.statement):
            has_ungrounded = True
        if not root_cause.candidate_hypotheses:
            has_ungrounded = True
        for hyp in root_cause.candidate_hypotheses:
            if _flagged(hyp.statement) or _flagged(hyp.name):
                has_ungrounded = True
                break
    if five_why is not None and not five_why.steps:
        has_ungrounded = True

    if state.get("analysis_mode") == "DEGRADED" or not has_ungrounded:
        trace.append(AgentTraceStep.ok("Deterministic critic firewall: Analysis grounded & structurally valid (0ms fast path)"))
        state["critic_approved"] = True
        state["critic_feedback"] = "Deterministic verification approved."
        state["critic_send_back"] = False
        state["critic_status"] = "SKIPPED"
        state["trace"] = trace
        state["errors"] = errors
        return state

    client = get_llm_client(timeout_seconds=settings.ollama_critic_timeout_seconds)

    system_prompt = (settings.agent_prompts_dir / "system_prompt.txt").read_text(encoding="utf-8")
    template = (settings.agent_prompts_dir / "critic.txt").read_text(encoding="utf-8")

    # PHASE 6: the critic only ever ACTS on domain-relevance/grounding
    # findings (drops flagged hypotheses, replaces a flagged narrative --
    # see the SEMANTIC ENFORCEMENT block below); contributing_factors/capa/
    # impact were never read from its response, so resending them was pure
    # wasted input tokens. Evidence is trimmed the same way the compact
    # synthesis-recovery path trims it (top VERIFIED/REPORTED items only) --
    # the critic needs enough to judge grounding, not the full ledger.
    from app.agent.nodes.core_synthesis import _assign_claim_ids, _trim_evidence_for_recovery
    reason_parts = []
    if root_cause and not root_cause.candidate_hypotheses:
        reason_parts.append("no candidate hypotheses were generated")
    if five_why is not None and not five_why.steps:
        reason_parts.append("5-Why chain is empty")
    if root_cause and (_flagged(root_cause.narrative) or _flagged(root_cause.statement)):
        reason_parts.append("root cause narrative/statement may reference an unsupported entity or domain")
    if root_cause and any(_flagged(h.statement) or _flagged(h.name) for h in root_cause.candidate_hypotheses):
        reason_parts.append("one or more hypotheses may reference an unsupported entity or domain")
    flag_reason = "; ".join(reason_parts) or "deterministic grounding check flagged a possible issue"

    prompt = template.format(
        finding_text=state["request"].finding_text,
        evidence_ledger_json=json.dumps(
            _trim_evidence_for_recovery(_assign_claim_ids(evidence_ledger)), default=str
        ),
        root_cause_narrative=root_cause.narrative if root_cause else "Not established.",
        candidate_hypotheses_json=json.dumps(
            [h.model_dump() for h in (root_cause.candidate_hypotheses if root_cause else [])], default=str
        ),
        flag_reason=flag_reason,
    )

    # Default: approve with warning if LLM fails
    approved = True
    send_back = False
    critic_feedback = "Critic review unavailable — LLM error. Proceeding to report generation."
    critic_status = "OK"

    try:
        from app.services.ollama_client import set_current_node
        set_current_node("critic")
        raw = await client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format_json=True,
            max_tokens=settings.ollama_critic_max_tokens,
            num_ctx=settings.ollama_critic_num_ctx,
        )
        parsed = parse_llm_json(raw)

        approved = bool(parsed.get("approved", True))
        send_back = bool(parsed.get("send_back_for_investigation", False))
        critic_feedback = str(parsed.get("feedback", ""))
        corrections = [str(c) for c in parsed.get("corrections_required", [])]
        issues = parsed.get("issues", [])

        # SEMANTIC ENFORCEMENT: act on the critic's own domain-relevance
        # judgment, not just log it. This is what makes the critic a real
        # firewall instead of advisory-only feedback nobody reads.
        unsupported_ids = {str(x) for x in parsed.get("unsupported_hypothesis_ids", [])}
        narrative_unsupported = bool(parsed.get("root_cause_narrative_unsupported", False))

        if root_cause and unsupported_ids:
            kept = [h for h in root_cause.candidate_hypotheses if h.id not in unsupported_ids]
            dropped_ids = {h.id for h in root_cause.candidate_hypotheses} & unsupported_ids
            if dropped_ids:
                root_cause.candidate_hypotheses = kept
                trace.append(AgentTraceStep.warn(
                    f"Critic: dropped hypothesis(es) {sorted(dropped_ids)} — proposed mechanism not "
                    "evidenced by this finding"
                ))

        if root_cause and narrative_unsupported:
            trace.append(AgentTraceStep.warn(
                "Critic: root cause narrative proposed a mechanism not evidenced by this finding — replaced"
            ))
            root_cause.narrative = SAFE_ROOT_CAUSE_FALLBACK
            root_cause.statement = None
            root_cause.leading_hypothesis = None
            if root_cause.status != "NOT_ESTABLISHED":
                root_cause.status = "NOT_ESTABLISHED"  # type: ignore[assignment]

        # Cap send_back at max_critic_iterations
        if send_back and critic_iteration >= settings.agent_max_critic_iterations:
            logger.warning("Critic wants to send back but max iterations reached — proceeding to report")
            send_back = False
            trace.append(AgentTraceStep.warn(
                "Critic review: max iterations reached — proceeding despite outstanding issues"
            ))

        if approved and not corrections:
            trace.append(AgentTraceStep.ok("Critic review: analysis approved"))
        else:
            blocking = [i for i in issues if i.get("severity") == "BLOCKING"]
            trace.append(AgentTraceStep.warn(
                f"Critic review: {len(blocking)} blocking issues, {len(corrections)} corrections needed"
            ))
            for correction in corrections[:5]:  # log first 5
                trace.append(AgentTraceStep.warn(f"  Critic: {correction}"))

    except Exception as exc:
        # Broad catch is deliberate here (mirrors core_synthesis_node): any
        # LLM-path failure -- provider error, timeout, malformed JSON, or
        # anything else -- must degrade gracefully to the pre-set safe
        # defaults above (approve with a warning) rather than crash the
        # whole graph and lose the analysis already produced upstream.
        # Critical: this node NEVER sets/touches state["analysis_mode"]. The
        # critic is a secondary/optional review of an already-successful
        # core_synthesis result -- its own failure is not a primary-synthesis
        # failure and must never demote analysis_mode="LLM" to "DEGRADED".
        logger.warning("Critic node failed: %s", exc)
        trace.append(AgentTraceStep.warn(
            f"Critic review unavailable ({type(exc).__name__}) — core synthesis result preserved as-is, "
            f"analysis_mode unchanged: {exc}"
        ))
        errors.append(f"Critic error: {exc}")
        critic_status = "UNAVAILABLE"

    return {
        **state,
        "critic_approved": approved,
        "critic_send_back": send_back,
        "critic_feedback": critic_feedback,
        "critic_status": critic_status,
        "critic_iteration": critic_iteration + 1,
        "trace": trace,
        "errors": errors,
    }
