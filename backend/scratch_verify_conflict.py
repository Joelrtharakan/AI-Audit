import asyncio
import json
from app.agent.graph import build_agent_graph
from app.models.agent import InvestigateRequest

async def main():
    graph = build_agent_graph()
    finding = "The operator stated they had not received training on the revised procedure, but the supervisor claimed the training was completed."
    state = {
        "request": InvestigateRequest(finding_text=finding),
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
    print("\n--- REPORT OUTPUT ---")
    print(f"Observation Quality: {report.observation_quality} (Confidence: {report.observation_confidence})")
    print(f"Root Cause Status: {report.root_cause.status} (Confidence: {report.root_cause_confidence})")
    print(f"Overall Confidence: {report.overall_confidence}")
    print(f"Leading Hypothesis: {report.root_cause.leading_hypothesis}")
    print(f"Leading Hypothesis Status: {report.root_cause.leading_hypothesis_status}")
    print("\nCandidate Hypotheses:")
    for h in report.root_cause.candidate_hypotheses:
        print(f"  [{h.id}] ({h.status}, strength={h.evidence_strength}): {h.statement}")
        print(f"      Confirms if: {h.confirms_if}")
        print(f"      Refutes if: {h.refutes_if}")
    print("\n5-Why Steps:")
    for i, s in enumerate(report.five_why.steps, start=1):
        print(f"  Why {i}: {s.question}")
        print(f"     -> [{s.status}] {s.answer}")
    print("\nImmediate Action:")
    print(f"  {res.get('ca_draft').immediate_action}")
    print("\nImpact Potential Effect:")
    print(f"  {report.impact_assessment.potential_effect}")

asyncio.run(main())
