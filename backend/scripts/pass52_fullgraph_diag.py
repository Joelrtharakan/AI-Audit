"""PASS 52 -- full compiled-graph diagnostic for the electrical-panel finding.

Captures the evidence ledger, canonical context, the remediation prompt+response,
and the final report.remediation_cost -- to locate where pricing is lost on the
PRODUCTION graph path (vs the stage path, which Pass 51 already fixed).
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(
    "/private/tmp/claude-501/-Users-joeltharakan-Documents-Audit-Management-System/"
    "0f92e64a-30fe-4f34-bdab-c0cb8ea9bd77/scratchpad"
)

FINDING = (
    "Eight electrical panels require corrective labeling and inspection. New labels cost "
    "Rs 350 per panel. An electrician requires 1.5 hours per panel at Rs 900 per hour, "
    "followed by a safety inspection costing Rs 6,000 for the complete area."
)

_IO: list[dict] = []


def _install() -> None:
    from app.services.llm.providers import litellm_provider as lp

    orig = lp.LiteLLMProvider.chat_completion

    async def wrapped(self, messages, *a, **kw):
        try:
            raw = await orig(self, messages, *a, **kw)
        except Exception as e:  # noqa: BLE001
            _IO.append({"node": kw.get("node", "?"), "messages": messages, "error": repr(e)})
            raise
        _IO.append({"node": kw.get("node", "?"), "messages": messages, "response": raw})
        return raw

    lp.LiteLLMProvider.chat_completion = wrapped


def _j(o):
    try:
        return o.model_dump()
    except Exception:
        return o


async def main() -> None:
    _install()
    from app.agent.graph import build_agent_graph
    from app.models.agent import InvestigateRequest

    g = build_agent_graph()
    state = {
        "request": InvestigateRequest(finding_text=FINDING),
        "evidence_ledger": [], "iteration_count": 0, "tool_call_count": 0,
        "critic_iteration": 0, "trace": [], "errors": [],
    }
    final = await g.ainvoke(state, {"recursion_limit": 80})

    rep = final.get("report")
    rc = getattr(rep, "remediation_cost", None) if rep else None
    led = final.get("evidence_ledger") or []
    ctx = final.get("canonical_semantic_context")

    trace = {
        "finding": FINDING,
        "needs_investigation": final.get("needs_investigation"),
        "semantic_mode": str(final.get("semantic_mode")),
        "evidence_ledger": [
            {"evidence_id": getattr(e, "evidence_id", None), "status": str(getattr(e, "status", None)),
             "claim": (getattr(e, "claim", None) or getattr(e, "text", ""))[:300],
             "source": getattr(e, "source", None)}
            for e in led
        ],
        "canonical_comparison": _j(getattr(ctx, "comparison", None)) if ctx else None,
        "canonical_recurrence": _j(getattr(ctx, "recurrence", None)) if ctx else None,
        "canonical_root_cause_status": str(getattr(ctx, "root_cause_status", None)) if ctx else None,
        "canonical_remediation_obligation": str(getattr(ctx, "remediation_obligation", None)) if ctx else None,
        "final_remediation_cost": _j(rc),
        "llm_nodes": [io["node"] for io in _IO],
    }
    for io in _IO:
        if io["node"] == "remediation_cost_interpretation":
            trace["remediation_prompt"] = io["messages"][-1]["content"]
            trace["remediation_raw_response"] = io["response"]
            break

    (OUT / "pass52_fullgraph_panels.json").write_text(json.dumps(trace, indent=2, default=str))
    print("=== evidence ledger ===")
    for e in trace["evidence_ledger"]:
        print(" ", e["evidence_id"], e["status"], "::", e["claim"][:160])
    print("\nneeds_investigation:", trace["needs_investigation"])
    print("llm nodes:", trace["llm_nodes"])
    print("canonical comparison:", json.dumps(trace["canonical_comparison"], default=str))
    print("canonical recurrence:", json.dumps(trace["canonical_recurrence"], default=str))
    print("canonical rca:", trace["canonical_root_cause_status"], "obligation:", trace["canonical_remediation_obligation"])
    print("\n=== FINAL REMEDIATION COST ===")
    fr = trace["final_remediation_cost"] or {}
    for k in ("status", "pricing_status", "one_time_cost", "recurring_cost", "recurring_horizon_total",
              "most_likely_estimate", "currency", "is_partial_estimate", "review_required",
              "unpriced_activities", "auditor_inputs_required", "not_assessable_reason",
              "cost_components", "rejected_items"):
        if isinstance(fr, dict):
            print(f"{k}: {json.dumps(fr.get(k), default=str)[:900]}")
    print("\n=== REMEDIATION PROMPT (EVIDENCE section) ===")
    p = trace.get("remediation_prompt", "")
    i = p.find("EVIDENCE")
    print(p[max(0, i - 200): i + 900] if i >= 0 else p[-1200:])
    print("\n=== REMEDIATION RAW RESPONSE (first 2500) ===")
    print(str(trace.get("remediation_raw_response", ""))[:2500])


if __name__ == "__main__":
    asyncio.run(main())
