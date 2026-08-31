"""PASS 51 -- deep diagnostic trace for one finding.

Captures every transition: raw canonical prompt+response, validated canonical
context, remediation prompt+response, parsed RemediationInterpretation,
validation outcome, calculation proposals, rejected items, final result.

    python scripts/pass51_diagnose.py "P51_panels_evidence_backed"
    python scripts/pass51_diagnose.py --text "free-text finding ..."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(
    "/private/tmp/claude-501/-Users-joeltharakan-Documents-Audit-Management-System/"
    "0f92e64a-30fe-4f34-bdab-c0cb8ea9bd77/scratchpad"
)

_IO: list[dict] = []


def _install_io_capture() -> None:
    from app.services.llm.providers import litellm_provider as lp

    orig = lp.LiteLLMProvider.chat_completion

    async def wrapped(self, messages, *a, **kw):
        node = kw.get("node") or (a and a[-1]) or "?"
        raw = await orig(self, messages, *a, **kw)
        _IO.append(
            {
                "node": kw.get("node", "?"),
                "model": getattr(self, "model", getattr(self, "_model", "?")),
                "messages": messages,
                "response": raw,
            }
        )
        return raw

    lp.LiteLLMProvider.chat_completion = wrapped


def _j(obj):
    try:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
    except Exception:
        pass
    return obj


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="")
    ap.add_argument("--text", default="")
    args = ap.parse_args()

    if args.text:
        ft = args.text
        sid = "adhoc"
    else:
        from tests.certification.pass51_dataset import by_id

        scn = by_id(args.scenario)
        ft = scn["finding"]
        sid = scn["id"]

    _install_io_capture()

    from app.services.canonical_finding_interpreter import interpret_finding_canonically_with_status
    from app.services.canonical_context_validator import validate_canonical_context
    from app.remediation.interpreter import interpret_remediation
    from app.remediation.validator import validate_and_plan
    from app.remediation.engine import estimate_remediation_cost

    trace: dict = {"scenario": sid, "finding": ft}

    status, raw_ctx = await interpret_finding_canonically_with_status(
        finding_text=ft, evidence_ledger=[], timeout_seconds=None
    )
    trace["canonical_status"] = str(status)
    trace["canonical_raw_ctx"] = _j(raw_ctx)
    ctx = validate_canonical_context(raw_ctx, [], ft) if raw_ctx is not None else None
    trace["canonical_validated_ctx"] = _j(ctx)
    if ctx is not None:
        trace["canonical_comparison"] = _j(getattr(ctx, "comparison", None))
        trace["canonical_recurrence"] = _j(getattr(ctx, "recurrence", None))
        trace["canonical_root_cause_status"] = str(getattr(ctx, "root_cause_status", None))
        trace["canonical_remediation_obligation"] = str(getattr(ctx, "remediation_obligation", None))
        trace["canonical_remediation_activities"] = _j(getattr(ctx, "remediation_activities", None))
        trace["canonical_pricing_information"] = _j(getattr(ctx, "pricing_information", None))
        trace["canonical_entities"] = _j(getattr(ctx, "entities", None))

    rstatus, interp = await interpret_remediation(
        finding_text=ft, evidence_ledger=[], semantic_context=ctx
    )
    trace["remediation_status"] = str(rstatus)
    trace["remediation_interpretation"] = _j(interp)

    if interp is not None:
        try:
            outcome = validate_and_plan(interp)
            trace["validation_outcome"] = _j(outcome)
        except Exception as e:
            trace["validation_outcome_error"] = repr(e)

    rc = await estimate_remediation_cost(finding_text=ft, semantic_context=ctx)
    trace["final_result"] = _j(rc)

    trace["llm_io"] = _IO

    out = OUT / f"pass51_diag_{sid}.json"
    out.write_text(json.dumps(trace, indent=2, default=str))
    print(f"wrote {out}")

    # console summary
    print("\n=== CANONICAL ===", trace["canonical_status"])
    print("comparison:", json.dumps(trace.get("canonical_comparison"), default=str))
    print("recurrence:", json.dumps(trace.get("canonical_recurrence"), default=str))
    print("root_cause_status:", trace.get("canonical_root_cause_status"))
    print("remediation_obligation:", trace.get("canonical_remediation_obligation"))
    print("remediation_activities:", json.dumps(trace.get("canonical_remediation_activities"), default=str)[:600])
    print("pricing_information:", json.dumps(trace.get("canonical_pricing_information"), default=str)[:800])
    print("\n=== REMEDIATION INTERP ===", trace["remediation_status"])
    ri = trace.get("remediation_interpretation") or {}
    if isinstance(ri, dict):
        print("overall_status:", ri.get("overall_status"), "estimability:", ri.get("estimability"),
              "not_assessable_reason:", ri.get("not_assessable_reason"))
        print("activities:", json.dumps(ri.get("activities"), default=str)[:900])
        print("cost_components:", json.dumps(ri.get("cost_components"), default=str)[:1500])
        print("calculation_proposals:", json.dumps(ri.get("calculation_proposals"), default=str)[:1500])
        print("auditor_inputs:", json.dumps(ri.get("auditor_inputs_required") or ri.get("auditor_inputs"), default=str)[:900])
    print("\n=== VALIDATION OUTCOME ===")
    vo = trace.get("validation_outcome") or {}
    if isinstance(vo, dict):
        for k in ("planned_components", "calculation_proposals", "rejected_items", "components"):
            if k in vo:
                print(f"{k}:", json.dumps(vo[k], default=str)[:1200])
    print("\n=== FINAL RESULT ===")
    fr = trace.get("final_result") or {}
    if isinstance(fr, dict):
        for k in ("status", "pricing_status", "one_time_cost", "recurring_cost", "recurring_period",
                  "recurring_horizon_total", "most_likely_estimate", "currency", "is_partial_estimate",
                  "review_required", "unpriced_activities", "auditor_inputs_required",
                  "not_assessable_reason", "cost_components", "rejected_items", "unresolved_pricing_drivers"):
            print(f"{k}:", json.dumps(fr.get(k), default=str)[:1400])


if __name__ == "__main__":
    asyncio.run(main())
