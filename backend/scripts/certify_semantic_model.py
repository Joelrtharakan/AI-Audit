"""Semantic-model capability certification runner (spec §31-§40).

Runs the held-out benchmark (tests/certification/semantic_capability_benchmark.py)
through the COMPLETE agent graph for one or more configured models and classifies
every case as CORRECT / SAFE_ABSTENTION / CAUGHT_ERROR / SILENT_MATERIAL_ERROR.

Usage:
    python scripts/certify_semantic_model.py                 # default model
    python scripts/certify_semantic_model.py qwen3:8b qwen3:14b
    python scripts/certify_semantic_model.py --ids A1,E1,H1  # subset

Never installs / pulls a model. An unavailable model is reported MODEL_UNAVAILABLE
and the run continues with the others.

Autonomous eligibility (spec §38): ZERO SILENT_MATERIAL_ERROR required.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.certification.semantic_capability_benchmark import BENCHMARK, material_dimensions  # noqa: E402

_OUT = Path("/private/tmp/claude-501") / "cert"
_MATERIAL = material_dimensions()


def _ev_items(evidence):
    from app.models.agent import EvidenceItem, EvidenceStatus
    out = []
    for claim, status in (evidence or []):
        out.append(EvidenceItem(claim=claim, status=getattr(EvidenceStatus, status), source="benchmark"))
    return out


def _model_available(model: str) -> bool:
    import httpx
    from app.config import get_settings
    base = (get_settings().ollama_base_url or "http://localhost:11434").rstrip("/")
    try:
        r = httpx.get(f"{base}/api/tags", timeout=4.0)
        names = {m["name"] for m in r.json().get("models", [])}
        return model in names
    except Exception:
        return False


def _preflight(model: str) -> dict:
    """Operational preflight -- does NOT touch benchmark cases / oracle / scoring.
    Returns {ok: bool, reason: str, meta: {...}}. When ok is False the caller must
    NOT execute the 22 cases."""
    import hashlib
    import httpx

    from app.config import get_settings
    s = get_settings()
    meta: dict = {
        "provider": s.llm_provider,
        "configured_ollama_model": s.ollama_model,
        "certified_model": model,
        "ollama_base_url": s.ollama_base_url,
        "llm_fallback_enabled": s.llm_fallback_enabled,
        "canonical_semantic_prompt_version": s.canonical_semantic_prompt_version,
        "remediation_cost_prompt_version": s.remediation_cost_prompt_version,
        "analysis_prompt_version": s.analysis_prompt_version,
        "benchmark_cases": [c["id"] for c in BENCHMARK],
        "benchmark_sha256": hashlib.sha256(
            Path(__file__).resolve().parent.parent
            .joinpath("tests/certification/semantic_capability_benchmark.py")
            .read_bytes()).hexdigest()[:16],
    }

    if s.llm_provider != "ollama":
        return {"ok": False, "reason": f"PROVIDER_MISMATCH: configured provider is {s.llm_provider!r}, not 'ollama'", "meta": meta}
    if s.llm_fallback_enabled:
        return {"ok": False, "reason": "FALLBACK_ENABLED: LLM_FALLBACK_ENABLED must be false so no model can silently replace the configured one", "meta": meta}
    if model != s.ollama_model:
        return {"ok": False, "reason": f"MODEL_MISMATCH: certifying {model!r} but OLLAMA_MODEL is {s.ollama_model!r}", "meta": meta}

    base = (s.ollama_base_url or "http://localhost:11434").rstrip("/")
    try:
        r = httpx.get(f"{base}/api/tags", timeout=4.0)
        r.raise_for_status()
        installed = {m["name"] for m in r.json().get("models", [])}
    except Exception as exc:
        return {"ok": False, "reason": f"PROVIDER_UNAVAILABLE: {base}/api/tags -> {type(exc).__name__}: {exc}", "meta": meta}
    meta["installed_models"] = sorted(installed)
    if model not in installed:
        return {"ok": False, "reason": f"MODEL_UNAVAILABLE: {model!r} not installed in the configured Ollama ({sorted(installed)})", "meta": meta}

    return {"ok": True, "reason": "OK", "meta": meta}


async def _run_case(case: dict) -> dict:
    from app.agent.graph import build_agent_graph
    from app.models.agent import InvestigateRequest
    from app.services.canonical_semantic_models import comparison_is_active

    g = build_agent_graph()
    st = {
        "request": InvestigateRequest(finding_text=case["finding"]),
        "evidence_ledger": _ev_items(case.get("evidence")),
        "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
        "trace": [], "errors": [],
    }
    t0 = time.time()
    try:
        f = await g.ainvoke(st, {"recursion_limit": 80})
    except Exception as e:
        return {"id": case["id"], "verdict": "RUN_ERROR", "detail": repr(e), "wall_s": round(time.time() - t0, 1)}
    wall = round(time.time() - t0, 1)
    rep = f.get("report")
    rc = getattr(rep, "remediation_cost", None) if rep else None
    ctx = f.get("canonical_semantic_context")
    rc_status = f.get("root_cause_result") or f.get("root_cause")

    _cmp = getattr(ctx, "comparison", None) if ctx else None
    got = {
        "comparison_active": bool(comparison_is_active(_cmp)) if ctx else None,
        "comparison_raw": ({k: getattr(_cmp, k, None) for k in
                            ("left", "right", "reference", "status", "why_comparable",
                             "comparison_basis", "direction", "magnitude", "unit")}
                           if _cmp is not None else None),
        "pricing_status": str(getattr(rc, "pricing_status", None)),
        "one_time": getattr(rc, "one_time_cost", None),
        "recurring": getattr(rc, "recurring_cost", None),
        "recurring_period": getattr(rc, "recurring_period", None),
        "horizon_total": getattr(rc, "recurring_horizon_total", None),
        "review_required": bool(getattr(rc, "review_required", False)),
        "not_assessable_reason": getattr(rc, "not_assessable_reason", ""),
        "rejected": [getattr(r, "reason_code", "") for r in (getattr(rc, "rejected_items", []) or [])],
        "rca_status": str(getattr(rc_status, "status", getattr(rc_status, "root_cause_status", "?"))),
        "components": [{"d": c.description, "amt": c.calculated_amount, "rec": c.recurrence}
                       for c in (getattr(rc, "cost_components", []) or [])],
        # richer structured diagnostics (observability -- never chain-of-thought)
        "components_full": [
            {k: getattr(c, k, None) for k in (
                "component_id", "description", "value_kind", "quantity", "quantity_unit",
                "quantity_basis", "quantity_derivation", "unit_cost", "unit_cost_basis",
                "currency", "amount_type", "recurrence", "recurring_period", "calculated_amount")}
            for c in (getattr(rc, "cost_components", []) or [])],
        "calc_proposals": [
            {k: getattr(p, k, None) for k in (
                "calculation_id", "operation", "produces", "frequency", "recurring_period",
                "horizon", "horizon_unit", "horizon_basis", "target_component_id")}
            for p in (getattr(rc, "calculation_proposals", []) or getattr(rc, "calculation_traces", []) or [])],
        "recurring_horizon": getattr(rc, "recurring_horizon", None),
        "recurring_horizon_basis": getattr(rc, "recurring_horizon_basis", None),
        "uncertainty_reasons": list(getattr(rc, "uncertainty_reasons", []) or [])[:6],
        "semantic_mode": str(f.get("semantic_mode")),
    }

    exp = case["expect"]
    fails, abstained = [], False
    _abstain_states = ("NOT_ASSESSABLE", "PARTIAL_ESTIMATE")
    _abstained = got["pricing_status"] in _abstain_states or any(
        k in got["rejected"] for k in ("RECURRENCE_NOT_ESTABLISHED", "RECURRENCE_INCONSISTENT",
                                       "UNIT_OR_RATE_BASIS_UNRESOLVED"))

    if "comparison_active" in exp and got["comparison_active"] != exp["comparison_active"]:
        fails.append(f"comparison_active {got['comparison_active']}!={exp['comparison_active']}")
    if "cost_recurrence" in exp:
        want = exp["cost_recurrence"]
        recs = {c["rec"] for c in got["components"]}
        if want == "RECURRING" and "RECURRING" not in recs:
            fails.append(f"cost_recurrence want RECURRING, comps={recs}")
        if want == "ONE_TIME" and recs and recs != {"ONE_TIME"}:
            fails.append(f"cost_recurrence want ONE_TIME, comps={recs}")
    if "one_time_approx" in exp:
        v, tol = exp["one_time_approx"]
        g_ = got["one_time"]
        if g_ is None or abs(float(g_) - v) > tol:
            fails.append(f"one_time {g_}!={v}")
    if "recurring_approx" in exp:
        v, tol = exp["recurring_approx"]
        g_ = got["recurring"]
        if g_ is None or abs(float(g_) - v) > tol:
            fails.append(f"recurring {g_}!={v}")
    if "recurring_period_contains" in exp and (exp["recurring_period_contains"] not in (got["recurring_period"] or "")):
        fails.append(f"recurring_period {got['recurring_period']!r} lacks {exp['recurring_period_contains']!r}")
    if "horizon_total_approx" in exp:
        v, tol = exp["horizon_total_approx"]
        g_ = got["horizon_total"]
        if g_ is None or abs(float(g_) - v) > tol:
            fails.append(f"horizon_total {g_}!={v}")
    if exp.get("no_finite_total") and got["horizon_total"] is not None:
        fails.append(f"no_finite_total violated: horizon_total={got['horizon_total']}")
    if exp.get("no_finite_total") and got["one_time"]:
        fails.append(f"no_finite_total: recurring cost priced as one_time={got['one_time']}")
    if "pricing_status_in" in exp and got["pricing_status"] not in exp["pricing_status_in"]:
        fails.append(f"pricing_status {got['pricing_status']} not in {exp['pricing_status_in']}")
    if exp.get("rca_not_established") is True and "NOT_ESTABLISHED" not in got["rca_status"].upper():
        fails.append(f"rca {got['rca_status']} expected NOT_ESTABLISHED")
    if exp.get("rca_not_established") is False and "NOT_ESTABLISHED" in got["rca_status"].upper():
        fails.append(f"rca {got['rca_status']} expected ESTABLISHED/POSSIBLE")
    if exp.get("review_required") and not got["review_required"]:
        fails.append("review_required expected True")

    # classify
    if not fails:
        verdict = "CORRECT"
    elif exp.get("safe_if_unknown") and _abstained and got["review_required"]:
        verdict = "SAFE_ABSTENTION"
    elif _abstained:
        verdict = "CAUGHT_ERROR"
    elif case["dimension"] in _MATERIAL or case["dimension"] in ("A", "35"):
        verdict = "SILENT_MATERIAL_ERROR"
    else:
        verdict = "MINOR_ERROR"

    # Generic semantic-capability error class (spec §18) -- never tied to a
    # finding or an industry; it says which capability a future model must
    # improve.
    _fj = " ".join(fails).lower()
    if "comparison_active" in _fj:
        err_class = "RELATIONSHIP_ERROR"
    elif "horizon" in _fj:
        err_class = "HORIZON_ERROR"
    elif "recurring" in _fj or "cost_recurrence" in _fj or "no_finite_total" in _fj:
        err_class = "RECURRENCE_ERROR"
    elif "one_time" in _fj or "recurring_period" in _fj:
        err_class = "QUANTITY_ERROR"
    elif "rca" in _fj:
        err_class = "CAUSALITY_ERROR"
    elif "pricing_status" in _fj or "review_required" in _fj:
        err_class = "EVIDENCE_ERROR"
    else:
        err_class = "OTHER" if fails else ""

    return {"id": case["id"], "domain": case["domain"], "dimension": case["dimension"],
            "verdict": verdict, "error_class": err_class, "fails": fails,
            "got": got, "wall_s": wall}


async def certify(model: str, ids: set[str] | None) -> dict:
    import os
    os.environ["CANONICAL_SEMANTIC_MODEL"] = model
    os.environ["REMEDIATION_COST_MODEL"] = model
    os.environ["REMEDIATION_COST_ESTIMATION_ENABLED"] = "true"
    # force fresh settings
    from app.config import get_settings
    get_settings.cache_clear()

    pf = _preflight(model)
    print("---- CERTIFICATION PREFLIGHT ----", flush=True)
    for k, v in pf["meta"].items():
        print(f"  {k}: {v}", flush=True)
    print(f"  preflight: {pf['reason']}", flush=True)
    print("---------------------------------", flush=True)
    if not pf["ok"]:
        return {"model": model, "status": "CERTIFICATION_BLOCKED", "reason": pf["reason"], "meta": pf["meta"]}

    cases = [c for c in BENCHMARK if (ids is None or c["id"] in ids)]
    results = []
    for c in cases:
        r = await _run_case(c)
        results.append(r)
        print(f"[{model}] {r['id']:6} {r.get('dimension',''):3} -> {r['verdict']:22} "
              f"({r.get('wall_s')}s) {'; '.join(r.get('fails', []))[:120]}", flush=True)
        _OUT.mkdir(parents=True, exist_ok=True)
        json.dump({"model": model, "results": results},
                  open(_OUT / f"cert_{model.replace(':', '_')}.json", "w"), indent=2, default=str)

    tally: dict[str, int] = {}
    err_classes: dict[str, int] = {}
    for r in results:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
        if r["verdict"] == "SILENT_MATERIAL_ERROR" and r.get("error_class"):
            err_classes[r["error_class"]] = err_classes.get(r["error_class"], 0) + 1
    return {"model": model, "status": "DONE", "n": len(results), "tally": tally,
            "silent_error_classes": err_classes, "results": results}


def evaluate_certification_gates(tally: dict, n: int) -> dict:
    """Governance-only interpretation of a certification tally into four
    INDEPENDENT gates. Does NOT touch the oracle, the scoring, or any case
    classification -- it only reads the already-computed tally.

    SEMANTIC_SAFETY_GATE       -- "does the system fail safely?"
        PASS iff no SILENT_MATERIAL_ERROR and no RUN_ERROR.

    AUTONOMOUS_CAPABILITY_GATE -- "does the model meet the full held-out
                                  semantic-capability bar?"
        PASS iff SEMANTIC_SAFETY_GATE PASS *and* every case is CORRECT or
        SAFE_ABSTENTION (i.e. ZERO CAUGHT_ERROR). A CAUGHT_ERROR is a genuine
        capability failure that the deterministic layer caught -- safe, but not
        capable. SILENT_MATERIAL_ERROR == 0 is necessary, NOT sufficient.

    HUMAN_REVIEWED_PRODUCTION_GATE
        PASS (CONDITIONAL) iff SEMANTIC_SAFETY_GATE PASS. "CONDITIONAL" always,
        because integration (ASP.NET/DB/approval) and environment blockers are
        outside this script.

    REGRESSION_GATE
        Not evaluated here -- determined by the backend regression suite.
    """
    sme = int(tally.get("SILENT_MATERIAL_ERROR", 0))
    run_err = int(tally.get("RUN_ERROR", 0)) + int(tally.get("MODEL_UNAVAILABLE", 0))
    caught = int(tally.get("CAUGHT_ERROR", 0))
    correct = int(tally.get("CORRECT", 0))
    safe_abs = int(tally.get("SAFE_ABSTENTION", 0))

    safety = (sme == 0 and run_err == 0)
    fully_capable = safety and caught == 0 and (correct + safe_abs) == n and n > 0

    gates = {
        "SEMANTIC_SAFETY_GATE": "PASS" if safety else "FAIL",
        "REGRESSION_GATE": "NOT EVALUATED BY THIS SCRIPT (run the backend regression suite)",
        "HUMAN_REVIEWED_PRODUCTION_GATE": "PASS (CONDITIONAL)" if safety else "FAIL",
        "AUTONOMOUS_CAPABILITY_GATE": "PASS" if fully_capable else "FAIL",
    }
    if not safety:
        final = "BLOCKED — SEMANTIC SAFETY GATE FAILED"
    elif fully_capable:
        final = ("HUMAN-REVIEWED PRODUCTION READY (CONDITIONAL); "
                 "AUTONOMOUS CAPABILITY GATE PASS — confirm remaining integration/environment blockers")
    else:
        final = ("HUMAN-REVIEWED PRODUCTION READY (CONDITIONAL); "
                 "AUTONOMOUS DEPLOYMENT NOT ELIGIBLE "
                 f"({caught} material case(s) classified CAUGHT_ERROR — capability bar not met)")
    gates["FINAL_STATUS"] = final
    return gates


async def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    ids = None
    for a in sys.argv[1:]:
        if a.startswith("--ids"):
            ids = set(a.split("=", 1)[1].split(",")) if "=" in a else set(sys.argv[sys.argv.index(a) + 1].split(","))
    models = argv or ["qwen3:8b"]
    summary = []
    for m in models:
        s = await certify(m, ids)
        summary.append(s)
        print(f"\n=== {m}: {s.get('status')} {s.get('tally', {})} ===\n", flush=True)
    print("\n================ CERTIFICATION SUMMARY ================")
    for s in summary:
        if s.get("status") != "DONE":
            print(f"{s['model']:12} {s['status']} {s.get('reason', '')}")
            continue
        t = s["tally"]
        print(f"{s['model']:12} n={s['n']:3} {t}")
        gates = evaluate_certification_gates(t, s["n"])
        s["gates"] = gates
        for k in ("SEMANTIC_SAFETY_GATE", "REGRESSION_GATE",
                  "HUMAN_REVIEWED_PRODUCTION_GATE", "AUTONOMOUS_CAPABILITY_GATE"):
            print(f"{'':4}{k:32} : {gates[k]}")
        print(f"{'':4}{'FINAL STATUS':32} : {gates['FINAL_STATUS']}")
        if s.get("silent_error_classes"):
            print(f"{'':12} silent-error capability gaps: {s['silent_error_classes']}")
    json.dump(summary, open(_OUT / "cert_summary.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    asyncio.run(main())
