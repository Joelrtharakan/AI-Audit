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
    capa = state.get("capa_analysis")
    impact = state.get("impact_assessment")
    settings = get_settings()
    client = get_llm_client()

    system_prompt = (settings.agent_prompts_dir / "system_prompt.txt").read_text(encoding="utf-8")
    template = (settings.agent_prompts_dir / "critic.txt").read_text(encoding="utf-8")

    prompt = template.format(
        finding_text=state["request"].finding_text,
        evidence_ledger_json=json.dumps(
            [e.model_dump() for e in evidence_ledger], default=str
        ),
        root_cause_status=root_cause.status if root_cause else "NOT_ESTABLISHED",
        root_cause_narrative=root_cause.narrative if root_cause else "Not established.",
        candidate_hypotheses_json=json.dumps(
            [h.model_dump() for h in (root_cause.candidate_hypotheses if root_cause else [])], default=str
        ),
        contributing_factors_json=json.dumps(
            [cf.model_dump() for cf in state.get("contributing_factors", [])], default=str
        ),
        five_why_json=five_why.model_dump_json(indent=2) if five_why else "{}",
        capa_status=capa.status.value if capa else "INVESTIGATION_REQUIRED",
        impact_status=impact.status if impact else "REQUIRES_ASSESSMENT",
    )

    # Default: approve with warning if LLM fails
    approved = True
    send_back = False
    critic_feedback = "Critic review unavailable — LLM error. Proceeding to report generation."

    try:
        raw = await client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format_json=True,
            max_tokens=1024,
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

    except (LLMError, ValueError, KeyError) as exc:
        logger.warning("Critic node failed: %s", exc)
        trace.append(AgentTraceStep.warn(f"Critic review failed — proceeding to report: {exc}"))
        errors.append(f"Critic error: {exc}")

    return {
        **state,
        "critic_approved": approved,
        "critic_send_back": send_back,
        "critic_feedback": critic_feedback,
        "critic_iteration": critic_iteration + 1,
        "trace": trace,
        "errors": errors,
    }
