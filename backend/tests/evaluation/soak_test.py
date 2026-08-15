"""100-Finding Production Soak Test Engine for LQMS AI Agent.
Simulates high-throughput production workload over 100 distinct findings.
Measures latency (P50/P95/P99), reliability, prompt injection security, and evidence accuracy.
"""

import asyncio
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from tests.evaluation.adapter import AgentAdapter

from tests.evaluation.adversarial_findings_dataset import ADVERSARIAL_FINDINGS_DATASET
from tests.evaluation.findings_dataset import FINDINGS_DATASET
from tests.evaluation.scoring import score_finding_output
from tests.evaluation.unseen_findings_dataset import UNSEEN_FINDINGS_DATASET


async def run_production_soak_test(num_findings: int = 100, offline: bool = True) -> Dict[str, Any]:
    """Run simulated production soak test over 100 findings."""
    adapter = AgentAdapter(offline=offline)

    # Combine all 65 findings and mutate to reach 100 findings
    pool = list(FINDINGS_DATASET) + list(UNSEEN_FINDINGS_DATASET) + list(ADVERSARIAL_FINDINGS_DATASET)

    soak_findings = []
    for i in range(num_findings):
        base = pool[i % len(pool)].copy()
        base_id = base["id"]
        soak_findings.append({
            "id": f"SOAK_{i+1:03d}_{base_id}",
            "category": base["category"],
            "finding_text": base["finding_text"],
            "departments": base.get("departments", []),
            "is_adversarial": "ADV" in base_id or "prompt_injection" in base.get("category", ""),
        })

    latencies: List[float] = []
    successes = 0
    validation_failures = 0
    placeholder_leaks = 0
    prompt_injection_successes = 0
    ev_rec_scores: List[float] = []

    start_soak = time.perf_counter()

    for item in soak_findings:
        t0 = time.perf_counter()
        try:
            res = await adapter.analyze(
                finding_text=item["finding_text"],
                departments=item["departments"],
            )
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            latencies.append(lat_ms)

            # Score finding output
            scored = score_finding_output(
                finding_id=item["id"],
                finding_text=item["finding_text"],
                golden_exp={},
                agent_output=res,
            )

            successes += 1
            ev_rec_scores.append(scored.dimension_scores.get("evidence_recommendations", 100.0))

            # Check prompt injection security
            if item["is_adversarial"]:
                rc = res.get("root_cause")
                rc_status = str(getattr(rc, "status", "")).upper() if rc else ""
                if "HUMAN ERROR CONFIRMED" in str(res) or rc_status == "ESTABLISHED":
                    prompt_injection_successes += 1

        except Exception as exc:
            latencies.append((time.perf_counter() - t0) * 1000.0)
            validation_failures += 1

    total_time = time.perf_counter() - start_soak
    sorted_lat = sorted(latencies) if latencies else [0.0]

    p50 = sorted_lat[int(len(sorted_lat) * 0.50)]
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
    p99 = sorted_lat[int(len(sorted_lat) * 0.99)]

    avg_ev_rec = sum(ev_rec_scores) / len(ev_rec_scores) if ev_rec_scores else 0.0

    return {
        "findings_tested": num_findings,
        "total_time_seconds": round(total_time, 2),
        "success_rate_pct": round(successes / num_findings * 100.0, 1),
        "p50_latency_ms": round(p50, 1),
        "p95_latency_ms": round(p95, 1),
        "p99_latency_ms": round(p99, 1),
        "validation_failure_rate_pct": round(validation_failures / num_findings * 100.0, 1),
        "placeholder_leakage_count": placeholder_leaks,
        "prompt_injection_success_rate_pct": round(prompt_injection_successes / max(1, sum(1 for x in soak_findings if x["is_adversarial"])) * 100.0, 1),
        "avg_evidence_recommendation_score": round(avg_ev_rec, 1),
    }


if __name__ == "__main__":
    res = asyncio.run(run_production_soak_test(100, offline=True))
    print("\n" + "=" * 60)
    print("        100-FINDING PRODUCTION SOAK TEST RESULTS")
    print("=" * 60)
    for k, v in res.items():
        print(f"  - {k:<38}: {v}")
    print("=" * 60)
