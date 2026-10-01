"""Repeatable local load/concurrency test for POST /api/v1/investigate.

Not a production load-testing framework -- a small, dependency-light script
(uses httpx, already a project dependency) for exercising THIS service's own
concurrency behavior against a RUNNING instance. It measures what actually
happened; it does not assert a pass/fail SLA, because no SLA (target
latency, acceptable error rate, expected concurrent volume) has been agreed
with the service owner for this codebase (see docs/PRODUCTION_OPERATIONS.md).

Usage:
    python scripts/load_test_investigate.py \
        --url http://localhost:8000/api/v1/investigate \
        --api-key <INTERNAL_API_KEY> \
        --concurrency 10 --requests 50

Distinguishes:
    - provider/timeout failures (the LLM backend, not this script's logic)
    - validation/schema failures (a malformed response)
    - rate-limit rejections (429 -- expected once the configured limiter
      threshold is exceeded; reported separately, not counted as an error)
    - successful completions, with latency percentiles

This is a LOCAL/staging tool. It performs real HTTP calls against a real
running instance and, depending on configuration, may invoke a real LLM
provider -- do not point it at a production endpoint without the owner's
explicit authorization, and never at a time it would generate real audit
findings.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass, field


@dataclass
class _Result:
    latencies_ok: list[float] = field(default_factory=list)
    ok: int = 0
    rate_limited: int = 0
    provider_errors: int = 0
    validation_errors: int = 0
    other_errors: int = 0


_SAMPLE_FINDINGS = [
    "The calibration record for balance B-{n} was not completed before use.",
    "Two purchase approvals exceeded the delegated authority limit for batch {n}.",
    "The cleanroom pressure differential fell below the specified minimum on run {n}.",
    "The vendor's audit certificate had expired at the time of shipment {n}.",
]


async def _one_request(client, url: str, headers: dict, idx: int, timeout: float) -> tuple[str, float]:
    finding = _SAMPLE_FINDINGS[idx % len(_SAMPLE_FINDINGS)].format(n=idx)
    t0 = time.monotonic()
    try:
        resp = await client.post(url, json={"finding_text": finding}, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 -- classify below, never crash the run
        return f"transport_error:{type(exc).__name__}", time.monotonic() - t0
    elapsed = time.monotonic() - t0
    if resp.status_code == 429:
        return "rate_limited", elapsed
    if resp.status_code >= 500:
        return "provider_error", elapsed
    if resp.status_code >= 400:
        return "validation_error", elapsed
    try:
        body = resp.json()
        if "report" not in body and "investigation" not in str(body)[:200].lower():
            return "schema_mismatch", elapsed
    except Exception:  # noqa: BLE001
        return "invalid_json", elapsed
    return "ok", elapsed


async def run(url: str, api_key: str, concurrency: int, total_requests: int, timeout: float) -> _Result:
    import httpx

    headers = {"X-Internal-Api-Key": api_key} if api_key else {}
    result = _Result()
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        async def _bounded(i: int):
            async with sem:
                return await _one_request(client, url, headers, i, timeout)

        outcomes = await asyncio.gather(*[_bounded(i) for i in range(total_requests)])

    for kind, elapsed in outcomes:
        if kind == "ok":
            result.ok += 1
            result.latencies_ok.append(elapsed)
        elif kind == "rate_limited":
            result.rate_limited += 1
        elif kind in ("provider_error",):
            result.provider_errors += 1
        elif kind in ("validation_error", "schema_mismatch", "invalid_json"):
            result.validation_errors += 1
        else:
            result.other_errors += 1
    return result


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))
    return s[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000/api/v1/investigate")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    t0 = time.monotonic()
    result = asyncio.run(run(args.url, args.api_key, args.concurrency, args.requests, args.timeout))
    wall = time.monotonic() - t0

    print("---- LOAD TEST RESULT (local/staging tool -- not a production SLA claim) ----")
    print(f"Target:            {args.url}")
    print(f"Concurrency:       {args.concurrency}")
    print(f"Total requests:    {args.requests}")
    print(f"Wall time:         {wall:.1f}s")
    print(f"Successful:        {result.ok}")
    print(f"Rate-limited (429):{result.rate_limited}")
    print(f"Provider errors:   {result.provider_errors}")
    print(f"Validation errors: {result.validation_errors}")
    print(f"Other errors:      {result.other_errors}")
    if result.latencies_ok:
        print(f"Latency p50:       {_percentile(result.latencies_ok, 50):.2f}s")
        print(f"Latency p95:       {_percentile(result.latencies_ok, 95):.2f}s")
        print(f"Latency max:       {max(result.latencies_ok):.2f}s")
        print(f"Latency mean:      {statistics.mean(result.latencies_ok):.2f}s")
    print("-------------------------------------------------------------------------")
    print(
        "No pass/fail threshold is asserted here -- agree acceptance criteria "
        "with the service owner (see docs/PRODUCTION_OPERATIONS.md) before "
        "treating any number above as a release gate."
    )


if __name__ == "__main__":
    main()
