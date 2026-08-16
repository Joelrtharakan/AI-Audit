import asyncio
from app.agent.graph import build_agent_graph
from app.models.agent import InvestigateRequest

FINDING = (
    "The daily equipment inspection checklist was not completed for three consecutive days. "
    "The operator stated that they were unaware that the checklist procedure had been revised."
)


async def run_once(i):
    graph = build_agent_graph()
    state = {
        "request": InvestigateRequest(finding_text=FINDING),
        "iteration_count": 0,
        "tool_call_count": 0,
        "critic_iteration": 0,
        "observation_quality": None,
        "extraction": None,
        "investigation_plan": None,
        "needs_investigation": False,
        "planned_tools": [],
        "completed_tools": [],
        "current_tool": None,
        "tool_results": {},
        "evidence_ledger": [],
        "evidence_gaps": [],
        "root_cause": None,
        "contributing_factors": [],
        "five_why": None,
        "impact_assessment": None,
        "capa_analysis": None,
        "critic_approved": False,
        "critic_feedback": None,
        "critic_send_back": False,
        "report": None,
        "ca_draft": None,
        "final_state": None,
        "trace": [],
        "errors": [],
    }
    res = await graph.ainvoke(state)
    report = res.get("report")
    print(f"\n=== RUN {i} ===")
    print(f"Root Cause Status: {report.root_cause.status}")
    print(f"Leading Hypothesis: {report.root_cause.leading_hypothesis}")
    print(f"Why: {report.root_cause.root_cause_basis}")
    print("Candidate Hypotheses:")
    for h in report.root_cause.candidate_hypotheses:
        print(f"  [{h.id}] ({h.status}): {h.name} | {h.statement}")
        print(f"      Evidence needed: {h.evidence_needed}")
    inv = res.get("investigation_plan")
    if inv is not None:
        print("Investigation Areas:", getattr(inv, "areas", None))
        for q in (getattr(inv, "questions", None) or [])[:6]:
            print("  Q:", getattr(q, "text", q))
    print("5-Why:")
    for j, s in enumerate(report.five_why.steps, start=1):
        print(f"  Why {j}: [{s.status}] {s.answer}")


async def main():
    for i in range(3):
        await run_once(i)


asyncio.run(main())
