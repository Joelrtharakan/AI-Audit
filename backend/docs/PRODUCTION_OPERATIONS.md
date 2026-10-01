# Production Operations — Deployment, Monitoring, Backup, Load Testing

**Status: SPECIFICATION + PARTIAL VERIFICATION.** Sections are marked
`[VERIFIED]` (something in this session actually confirmed it),
`[IMPLEMENTED, UNVERIFIED]` (code/config exists but was not exercised against
real infrastructure), or `[DECISION NEEDED]` (requires the service owner to
decide before it can be implemented or tested). Nothing here is claimed as
tested against live infrastructure unless marked `[VERIFIED]`.

## 1. Deployment

`[DECISION NEEDED]` — no Dockerfile, container manifest, or deployment
script exists in this repository as inspected. Before deployment:

- Define the build artifact (container image vs. direct venv deployment).
- Separate `.env` per environment (development/test/staging/production) —
  `app/config.py` already reads from `.env` + environment variables, so this
  is a matter of deployment pipeline discipline, not code change.
- `ENVIRONMENT=production` is already enforced as a **hard gate**
  (`[VERIFIED]` via `tests/test_auth_fail_closed.py`, `tests/test_startup_config_validation.py`):
  a missing/sentinel `INTERNAL_API_KEY` in production fails closed (HTTP 500)
  rather than silently allowing unauthenticated access.
- Health checks: `GET /health` and `GET /health/llm` exist and are
  intentionally **not** gated by the API key or rate limiter (`[VERIFIED]`,
  `tests/test_rate_limiting.py::test_health_endpoint_is_never_throttled`) —
  suitable for a load-balancer/readiness probe as-is.
- Rollout/rollback: not defined in this repository — this is a deployment
  pipeline decision (e.g. blue/green, canary) requiring the actual hosting
  platform, which is outside this repo's scope.
- This service holds no database and runs no migrations itself — schema
  migration strategy belongs entirely to ASP.NET's database.

## 2. Monitoring and alerting

`[IMPLEMENTED, UNVERIFIED]` — `app/services/llm_metrics.py` and the
`GET /health/llm-metrics` endpoint already exist and track provider/model
identity, request outcomes, and latency; this was not extended this session.

Recommended signals to wire into the organization's actual monitoring stack
(Prometheus/Datadog/App Insights/etc. — not chosen here, since no monitoring
infrastructure is present in this repository):

| Signal | Existing source in this repo |
|---|---|
| Request volume/latency | `llm_metrics` + standard ASGI access logs |
| Error/timeout rate | `LLMError`/`AllLLMProvidersUnavailableError` exception paths in `app/routers/investigate.py` |
| Provider/model identity | `AnalysisProvenance.provider`, `app.services.llm.execution.LLMExecutionConfig` |
| Rate-limit events | New: `enforce_investigation_rate_limit` raises `HTTPException(429)` — wire a log/metric on that exception |
| Auth failures | `require_internal_api_key` raises 401/500 — wire a log/metric on that exception |
| Capability-gate denials | `AutonomyDecision.blocking_reasons` (this session's autonomy-provenance work) |
| Human-review status | `report.review.required` / `.reason_codes` |
| Persistence/integration failures | Not observable from this repo — ASP.NET-side |

`[DECISION NEEDED]`: alert thresholds (e.g. "page on-call if error rate >
X%") require the service owner's operational risk tolerance — not invented
here.

**Privacy**: confirmed (`[VERIFIED]`, `tests/test_security.py`) that
provider API keys never leak into the OpenAPI schema. Extend the same
discipline to any new structured logging: never log full finding/evidence
text or the internal API key at a broadly-readable log level.

## 3. Backup and recovery

`[DECISION NEEDED]` — this service is stateless except for:
- The in-memory OAuth session store (`app/auth/session.py`) — sessions are
  ephemeral by design (expire on their own); losing them on restart forces
  re-login, not data loss.
- The file-based capability-certificate store (`results/capability_certificates/`)
  — regenerable at any time by re-running `scripts/certify_semantic_model.py`;
  not a backup-critical asset.

All durable audit data lives in ASP.NET's database, outside this repository.
Backup frequency, retention, encryption, RTO/RPO, and restore testing are
therefore **entirely an ASP.NET/database-team responsibility** — this
service has nothing to back up beyond its own source code (already in git).

## 4. Data retention

`[DECISION NEEDED]` — no retention policy is defined anywhere in this
repository for audit records, evidence, AI analyses, or logs. This is a
data-owner decision (likely driven by the organization's actual audit/
regulatory retention requirements) and must not be invented here.

## 5. Load and concurrency testing

`[IMPLEMENTED, UNVERIFIED]` — a repeatable local load-test script,
`scripts/load_test_investigate.py` (see below), issues concurrent requests
against a running instance of this service and reports latency/error-rate
percentiles. This was **not** run against a production-like environment in
this session (no provisioned staging environment was available) — running it
is the concrete acceptance step for whoever owns that environment.

What IS verified in this repository already: `tests/conftest.py`'s
`_reset_llm_execution_context` fixture and the existing `ContextVar`-based
single-provider-per-request execution config (`app/services/llm/execution.py`)
prevent cross-request state leakage **within one process's test suite** —
this is necessary but explicitly **not sufficient** evidence for live,
multi-worker concurrency safety under real load; the distinction is called
out per this charter's own instruction not to conflate the two.

### Acceptance thresholds

`[DECISION NEEDED]` — no SLA (target p95 latency, max acceptable error rate,
expected concurrent-request volume) is defined in this repository. Do not
invent one; agree it with the service owner before using the load-test
script's output as a pass/fail gate.

## 6. Live-LLM CI lane

`.github/workflows/backend-ci.yml` runs only the deterministic suite (Ollama
blackholed, `CANONICAL_SEMANTIC_LLM_PRIMARY=false`) — by design, so CI never
depends on a reachable model endpoint or incurs LLM latency/cost on every PR.

A live-LLM lane is **not** included in that workflow because:
- It requires a real, reachable Ollama (or other provider) endpoint as a CI
  secret/service, which is an infrastructure decision (self-hosted runner
  with a GPU/Ollama instance, or a hosted provider API key) not available in
  this session.
- `scripts/certify_semantic_model.py` already IS that lane's core logic —
  it takes provider/model as explicit runtime input (never hardcoded),
  refuses to run against a mismatched/misconfigured model
  (`_preflight` fail-closed checks), and produces a
  `ModelCapabilityCertificate` rather than a bare pass/fail.

**Recommended addition, once a live endpoint is available**: a manually-
triggered (`workflow_dispatch`) job that runs
`python scripts/certify_semantic_model.py <model>` against a documented,
explicit endpoint, uploads the resulting certificate JSON as an artifact, and
never overwrites the `results/capability_certificates/` history. This is
specified here, not added to the checked-in workflow, because wiring it
would require a real endpoint secret that does not exist in this repository
or session — adding the job without one would either fail every run or
silently no-op, both worse than documenting the gap.
