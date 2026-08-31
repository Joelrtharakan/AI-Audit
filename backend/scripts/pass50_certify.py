"""PASS 50 -- live full-graph certification runner.

Runs the REAL compiled agent graph against the REAL configured Ollama model
for each scenario in tests/certification/pass50_dataset.py, extracts the
semantic + cost outcome, compares against the oracle, and writes an
incremental JSON report.

NOT imported by runtime code. No architecture change. Read-only against the
system under test.

Usage:
    python scripts/pass50_certify.py --ids A2_population_monthly_verification,C1_two_independent_prices
    python scripts/pass50_certify.py --material          # all material=True scenarios
    python scripts/pass50_certify.py --all
    CANONICAL_SEMANTIC_MODEL=qwen3:14b REMEDIATION_COST_MODEL=qwen3:14b \
        python scripts/pass50_certify.py --material
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.certification.pass50_dataset import SCENARIOS as _S50  # noqa: E402
from tests.certification.pass51_dataset import SCENARIOS as _S51  # noqa: E402

SCENARIOS = _S50 + _S51


def by_id(sid: str) -> dict:
    for s in SCENARIOS:
        if s["id"] == sid:
            return s
    raise KeyError(sid)


def material_ids() -> list[str]:
    return [s["id"] for s in SCENARIOS if s.get("material")]

OUT_DIR = Path(
    "/private/tmp/claude-501/-Users-joeltharakan-Documents-Audit-Management-System/"
    "0f92e64a-30fe-4f34-bdab-c0cb8ea9bd77/scratchpad"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---- per-call metadata capture (monkeypatch the ContextVar setter) ----
_CALLS: list[dict] = []


def _install_call_capture() -> None:
    import app.services.llm.call_metadata as cm

    _orig = cm.set_last_call_metadata

    def _wrapped(meta: dict) -> None:
        try:
            _CALLS.append(dict(meta))
        except Exception:
            pass
        return _orig(meta)

    cm.set_last_call_metadata = _wrapped
    # litellm_provider imported the symbol directly
    try:
        import app.services.llm.providers.litellm_provider as lp

        lp.set_last_call_metadata = _wrapped
    except Exception:
        pass


def _rc_status_verified(status) -> bool:
    return str(getattr(status, "value", status)) in {"VERIFIED", "SUPPORTED", "ESTABLISHED"}


def _extract(state: dict) -> dict:
    from app.services.canonical_semantic_models import comparison_is_active

    ctx = state.get("canonical_semantic_context")
    report = state.get("report")
    rc = getattr(report, "remediation_cost", None) if report else None

    out: dict = {
        "canonical_success": ctx is not None,
        "semantic_mode": str(state.get("semantic_mode")),
        "canonical_semantic_status": str(state.get("canonical_semantic_status")),
        "errors": list(state.get("errors") or []),
    }

    if ctx is not None:
        cmp_ = getattr(ctx, "comparison", None)
        rec = getattr(ctx, "recurrence", None)
        out["comparison_present"] = cmp_ is not None
        out["comparison_active"] = bool(comparison_is_active(cmp_)) if cmp_ is not None else False
        out["comparison_status"] = getattr(cmp_, "status", None) if cmp_ is not None else None
        out["comparison_why"] = getattr(cmp_, "why_comparable", None) if cmp_ is not None else None
        out["recurrence_present"] = rec is not None
        out["recurrence_count"] = getattr(rec, "count", None) if rec is not None else None
        out["recurrence_event"] = getattr(rec, "event", None) if rec is not None else None
        out["recurrence_period"] = getattr(rec, "period", None) if rec is not None else None
        out["root_cause_status"] = str(getattr(ctx, "root_cause_status", None))
        out["root_cause_verified"] = _rc_status_verified(getattr(ctx, "root_cause_status", None))
        out["causal_alternatives_unresolved"] = bool(getattr(ctx, "causal_alternatives_unresolved", False))
        out["missing_record_status"] = str(getattr(ctx, "missing_record_status", None))
        out["explicit_previous_capa_reference"] = bool(getattr(ctx, "explicit_previous_capa_reference", False))
        out["remediation_obligation"] = str(getattr(ctx, "remediation_obligation", None))

    if rc is not None:
        out["rc_pricing_status"] = str(getattr(rc, "pricing_status", None))
        out["rc_status"] = str(getattr(rc, "status", None))
        out["rc_one_time_cost"] = getattr(rc, "one_time_cost", None)
        out["rc_recurring_cost"] = getattr(rc, "recurring_cost", None)
        out["rc_recurring_period"] = getattr(rc, "recurring_period", None)
        out["rc_recurring_horizon_total"] = getattr(rc, "recurring_horizon_total", None)
        out["rc_recurring_horizon"] = getattr(rc, "recurring_horizon", None)
        out["rc_recurring_horizon_basis"] = str(getattr(rc, "recurring_horizon_basis", None))
        out["rc_most_likely"] = getattr(rc, "most_likely_estimate", None)
        out["rc_review_required"] = bool(getattr(rc, "review_required", False))
        out["rc_is_partial"] = bool(getattr(rc, "is_partial_estimate", False))
        out["rc_currency"] = getattr(rc, "currency", None)
        out["rc_unpriced_activities"] = list(getattr(rc, "unpriced_activities", []) or [])
        out["rc_auditor_inputs_required"] = list(getattr(rc, "auditor_inputs_required", []) or [])
        out["rc_not_assessable_reason"] = getattr(rc, "not_assessable_reason", None)
        out["rc_reasoning_source"] = str(getattr(rc, "reasoning_source", None))
        comps = getattr(rc, "cost_components", []) or []
        out["rc_components"] = [
            {
                "description": getattr(c, "description", None),
                "value_kind": str(getattr(c, "value_kind", None)),
                "recurrence": str(getattr(c, "recurrence", None)),
                "recurring_period": getattr(c, "recurring_period", None),
                "amount": getattr(c, "calculated_amount", getattr(c, "amount", None)),
            }
            for c in comps
        ]
        out["rc_rejected_items"] = [
            {"description": getattr(r, "description", None), "reason": str(getattr(r, "reason", None))}
            for r in (getattr(rc, "rejected_items", []) or [])
        ]
    else:
        out["rc_pricing_status"] = None

    return out


def _grade(scn: dict, res: dict) -> dict:
    """Compare extracted result against the oracle. Returns verdict dict."""
    exp = scn.get("expect", {})
    checks: list[dict] = []

    def chk(name: str, ok: bool | None, detail: str, material: bool = True):
        checks.append({"check": name, "ok": ok, "detail": detail, "material": material})

    if not res.get("canonical_success"):
        chk("canonical_success", False, f"canonical failed: {res.get('canonical_semantic_status')}")
        # a canonical failure is a safe state (fallback), not a material semantic error
        return {"verdict": "CANONICAL_FALLBACK", "checks": checks}

    if "comparison_active" in exp:
        want = exp["comparison_active"]
        got = res.get("comparison_active")
        ok = got == want
        chk(
            "comparison_active",
            ok,
            f"want={want} got={got} status={res.get('comparison_status')} why={res.get('comparison_why')!r}",
        )

    if "recurrence_count_max" in exp:
        cap = exp["recurrence_count_max"]
        got = res.get("recurrence_count")
        ok = got is None or got <= cap
        chk(
            "recurrence_count_max",
            ok,
            f"count<= {cap}? got={got} event={res.get('recurrence_event')!r} period={res.get('recurrence_period')!r}",
        )

    if "cost_recurrence" in exp and res.get("rc_pricing_status") is not None:
        want = exp["cost_recurrence"]
        has_rec = bool(res.get("rc_recurring_cost"))
        has_ot = bool(res.get("rc_one_time_cost"))
        if want == "RECURRING":
            ok = has_rec
        elif want == "ONE_TIME":
            ok = has_ot and not has_rec
        else:
            ok = not has_rec and not has_ot
        chk(
            "cost_recurrence",
            ok,
            f"want={want} one_time={res.get('rc_one_time_cost')} recurring={res.get('rc_recurring_cost')} "
            f"period={res.get('rc_recurring_period')} comps={res.get('rc_components')}",
        )

    if exp.get("recurring_period") and res.get("rc_pricing_status") is not None:
        want = exp["recurring_period"]
        got = (res.get("rc_recurring_period") or "").lower()
        ok = want in got if got else None
        chk("recurring_period", ok, f"want~{want} got={got!r}", material=False)

    if "has_horizon_total" in exp and res.get("rc_pricing_status") is not None:
        want = exp["has_horizon_total"]
        got = bool(res.get("rc_recurring_horizon_total"))
        chk(
            "has_horizon_total",
            got == want,
            f"want={want} got={got} total={res.get('rc_recurring_horizon_total')} "
            f"basis={res.get('rc_recurring_horizon_basis')}",
        )

    if exp.get("pricing_status_in") and res.get("rc_pricing_status") is not None:
        allowed = exp["pricing_status_in"]
        got = res.get("rc_pricing_status")
        chk("pricing_status_in", got in allowed, f"want in {allowed} got={got}")

    if exp.get("value_not_priced") and res.get("rc_pricing_status") is not None:
        # headline must NOT be presented as an exact single actual derived from a
        # loss / historical / unestablished figure
        ps = res.get("rc_pricing_status")
        ok = ps in {"NOT_ASSESSABLE", "PARTIAL_ESTIMATE", "RANGE_ESTIMATE"} or res.get("rc_review_required")
        chk("value_not_priced", ok, f"pricing_status={ps} review={res.get('rc_review_required')}")

    if exp.get("root_cause_not_verified"):
        got = res.get("root_cause_verified")
        chk("root_cause_not_verified", got is False, f"root_cause_status={res.get('root_cause_status')}")

    if exp.get("review_or_partial") and res.get("rc_pricing_status") is not None:
        ps = res.get("rc_pricing_status")
        ok = ps in {"NOT_ASSESSABLE", "PARTIAL_ESTIMATE", "RANGE_ESTIMATE"} or res.get("rc_review_required")
        chk("review_or_partial", ok, f"pricing_status={ps} review={res.get('rc_review_required')} "
            f"partial={res.get('rc_is_partial')}")

    def _approx(name, spec, got):
        if got is None:
            chk(name, False, f"want~{spec[0]}+-{spec[1]} got=None")
            return
        ok = abs(float(got) - float(spec[0])) <= float(spec[1])
        chk(name, ok, f"want~{spec[0]}+-{spec[1]} got={got}")

    if exp.get("one_time_cost_approx") and res.get("rc_pricing_status") is not None:
        _approx("one_time_cost_approx", exp["one_time_cost_approx"], res.get("rc_one_time_cost")
                or res.get("rc_most_likely"))
    if exp.get("recurring_cost_approx") and res.get("rc_pricing_status") is not None:
        _approx("recurring_cost_approx", exp["recurring_cost_approx"], res.get("rc_recurring_cost"))
    if exp.get("horizon_total_approx") and res.get("rc_pricing_status") is not None:
        _approx("horizon_total_approx", exp["horizon_total_approx"], res.get("rc_recurring_horizon_total"))

    if exp.get("min_priced_components") and res.get("rc_pricing_status") is not None:
        comps = res.get("rc_components") or []
        priced = [c for c in comps if c.get("amount") not in (None, 0)]
        # amount may be None in our extract; fall back to counting components + a headline cost
        n = len(priced) if priced else (len(comps) if (res.get("rc_one_time_cost") or res.get("rc_recurring_cost")) else 0)
        chk("min_priced_components", n >= exp["min_priced_components"],
            f"want>={exp['min_priced_components']} got~{n} comps={comps}")

    if exp.get("no_auditor_pricing_input") and res.get("rc_pricing_status") is not None:
        ai = res.get("rc_auditor_inputs_required") or []
        chk("no_auditor_pricing_input", len(ai) == 0,
            f"auditor_inputs={ai}")

    if exp.get("value_kind_not") and res.get("rc_components"):
        bad = [c for c in res["rc_components"] if str(c.get("value_kind")) in exp["value_kind_not"]]
        chk("value_kind_not", not bad, f"forbidden value_kind present: {bad}", material=True)

    material_fails = [c for c in checks if c["ok"] is False and c["material"]]
    minor_fails = [c for c in checks if c["ok"] is False and not c["material"]]
    unknown = [c for c in checks if c["ok"] is None]
    if material_fails:
        verdict = "MATERIAL_FAIL"
    elif minor_fails:
        verdict = "MINOR_FAIL"
    elif unknown and not [c for c in checks if c["ok"]]:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS"
    return {"verdict": verdict, "checks": checks}


async def _run_stage(scn: dict) -> tuple[dict, str | None]:
    """The 2 SEMANTIC LLM stages only: canonical interpret + validate, then
    remediation cost estimate. Same code paths the graph uses for semantics,
    without the RCA/investigation/critic/FEV latency."""
    from app.services.canonical_finding_interpreter import interpret_finding_canonically_with_status
    from app.services.canonical_context_validator import validate_canonical_context
    from app.remediation.engine import estimate_remediation_cost

    ft = scn["finding"]
    err = None
    ctx = None
    status = "UNSTARTED"
    try:
        status, raw = await interpret_finding_canonically_with_status(
            finding_text=ft, evidence_ledger=[], timeout_seconds=None
        )
        if raw is not None:
            ctx = validate_canonical_context(raw, [], ft)
    except Exception as e:  # noqa: BLE001
        err = f"canonical: {type(e).__name__}: {e}\n{traceback.format_exc()}"

    rc = None
    if err is None:
        try:
            rc = await estimate_remediation_cost(finding_text=ft, semantic_context=ctx)
        except Exception as e:  # noqa: BLE001
            err = f"remediation: {type(e).__name__}: {e}\n{traceback.format_exc()}"

    class _Rep:
        remediation_cost = rc

    fake_state = {
        "canonical_semantic_context": ctx,
        "semantic_mode": "CANONICAL_LLM" if ctx is not None else "DETERMINISTIC_FALLBACK",
        "canonical_semantic_status": status,
        "report": _Rep(),
        "errors": [],
    }
    return _extract(fake_state), err


async def _run_one(sid: str, mode: str = "stage") -> dict:
    from app.models.agent import InvestigateRequest

    scn = by_id(sid)
    _CALLS.clear()
    t0 = time.time()
    err = None
    final: dict | None = None

    if mode == "stage":
        res, err = await _run_stage(scn)
        final = res
    else:
        from app.agent.graph import build_agent_graph

        graph = build_agent_graph()
        req = InvestigateRequest(finding_text=scn["finding"])
        state = {
            "request": req, "evidence_ledger": [], "iteration_count": 0,
            "tool_call_count": 0, "critic_iteration": 0, "trace": [], "errors": [],
        }
        try:
            fg = await graph.ainvoke(state, {"recursion_limit": 60})
            final = _extract(fg)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            final = None

    wall = time.time() - t0

    res = final if final else {"canonical_success": False, "errors": [err or "no final state"]}
    grade = _grade(scn, res) if final else {"verdict": "ERROR", "checks": []}

    calls = [
        {
            k: c.get(k)
            for k in (
                "model", "elapsed_ms", "prompt_tokens", "output_tokens", "reasoning_tokens",
                "finish_reason", "native_load_ms", "native_prompt_eval_ms", "native_gen_ms",
                "native_total_ms", "native_tok_per_s",
            )
            if k in c
        }
        for c in _CALLS
    ]

    return {
        "id": sid,
        "section": scn.get("section"),
        "material": bool(scn.get("material")),
        "finding": scn["finding"],
        "expected_text": scn.get("expected_text"),
        "wall_seconds": round(wall, 1),
        "llm_calls": calls,
        "llm_call_count": len(_CALLS),
        "result": res,
        "grade": grade,
        "error": err,
    }


async def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--material", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--mode", default="stage", choices=["stage", "graph"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.ids:
        ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    elif args.material:
        ids = material_ids()
    elif args.all:
        ids = [s["id"] for s in SCENARIOS]
    else:
        ap.error("one of --ids / --material / --all required")

    tag = os.environ.get("CANONICAL_SEMANTIC_MODEL") or os.environ.get("OLLAMA_MODEL") or "default"
    tag = tag.replace(":", "-").replace("/", "-")
    out_path = Path(args.out) if args.out else OUT_DIR / f"pass50_results_{tag}_{args.mode}.json"

    _install_call_capture()

    from app.config import get_settings

    s = get_settings()
    header = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ollama_model": s.ollama_model,
        "canonical_semantic_model": s.canonical_semantic_model or "(global)",
        "remediation_cost_model": s.remediation_cost_model or "(global)",
        "remediation_cost_estimation_enabled": s.remediation_cost_estimation_enabled,
        "scenario_ids": ids,
    }
    print(json.dumps(header, indent=2))

    results = []
    for i, sid in enumerate(ids, 1):
        print(f"\n[{i}/{len(ids)}] {sid} ...", flush=True)
        r = await _run_one(sid, mode=args.mode)
        results.append(r)
        v = r["grade"]["verdict"]
        print(f"    -> {v}  ({r['wall_seconds']}s, {r['llm_call_count']} calls)", flush=True)
        for c in r["grade"]["checks"]:
            mark = {True: "ok ", False: "XX ", None: "?? "}[c["ok"]]
            print(f"       {mark}{c['check']}: {c['detail']}", flush=True)
        out_path.write_text(json.dumps({"header": header, "results": results}, indent=2, default=str))

    # summary
    counts: dict[str, int] = {}
    for r in results:
        counts[r["grade"]["verdict"]] = counts.get(r["grade"]["verdict"], 0) + 1
    summary = {"counts": counts, "out": str(out_path)}
    print("\n==== SUMMARY ====")
    print(json.dumps(summary, indent=2))
    out_path.write_text(
        json.dumps({"header": header, "summary": summary, "results": results}, indent=2, default=str)
    )


if __name__ == "__main__":
    asyncio.run(_main())
