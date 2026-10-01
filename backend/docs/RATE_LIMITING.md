# Request Throttling — Configuration and Deployment Notes

## What exists today

`app/services/rate_limiter.py` implements an in-process sliding-window
limiter applied to `POST /api/v1/investigate` and `POST /api/v1/analyze-finding`
(the only two expensive, LLM-driven endpoints). It runs as a FastAPI
dependency **after** `require_internal_api_key` — a request must already be
authenticated before it is ever counted, and throttling can never let an
unauthenticated request through.

Configuration (`app/config.py`):

| Setting | Default | Meaning |
|---|---|---|
| `RATE_LIMIT_ENABLED` | `true` | Master on/off switch. The test suite sets this to `false` (see `tests/conftest.py`) so the shared test API key doesn't trip the limiter across unrelated test files. |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | `30` | Max requests per 60s sliding window, per caller identity. |

Caller identity is currently the presented `X-Internal-Api-Key` value,
falling back to client host. Because this repository defines exactly **one**
shared service-to-service key (ASP.NET → FastAPI), every legitimate caller
today shares **one** bucket — the limiter caps aggregate investigation
throughput per process, not per end-user. Per-end-user throttling would
require ASP.NET to mint distinguishable per-user or per-tenant credentials
and pass them through; that is an ASP.NET-side decision, not something this
service can invent on its own.

## What this does NOT do (explicit limitation)

The counter lives in **one process's memory**. It is correctly scoped for a
single-worker deployment. It is **not** sufficient for:

- `uvicorn --workers N` (N > 1) — each worker enforces its own independent
  budget, so the effective aggregate limit is `RATE_LIMIT_REQUESTS_PER_MINUTE
  × N`.
- Multiple service instances behind a load balancer — same problem, one
  bucket per instance.

**Before deploying with more than one worker/instance**, replace the
in-process store with a shared backend (Redis is the standard choice; the
`SlidingWindowRateLimiter.check(key)` interface is deliberately narrow so a
Redis-backed implementation can be swapped in behind the same
`enforce_investigation_rate_limit` dependency without touching the routers).
This is an infrastructure decision requiring the deployment topology to be
known — it is called out here as an owned, outstanding item, not implemented
speculatively against an unknown target.

## Fail-safe policy if the (future) shared backend becomes unavailable

Not yet decided — this is a policy call for the service owner:

- **Fail open** (allow the request through) prioritizes availability over
  protection.
- **Fail closed** (reject with 503) prioritizes protection over availability.

Document the choice here once made; the current in-process limiter has no
external dependency to fail, so this decision only becomes relevant once a
shared backend is introduced.

## Tuning

`RATE_LIMIT_REQUESTS_PER_MINUTE` has no "correct" value baked in — it is a
business/capacity decision depending on: expected legitimate call volume from
ASP.NET, the LLM provider's own rate limits, and acceptable latency under
load. Start conservative and raise it based on observed legitimate traffic
patterns, not a guess made in this document.
