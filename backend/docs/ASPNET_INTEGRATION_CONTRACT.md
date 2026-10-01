# ASP.NET ⇄ FastAPI AI Service — Integration Contract and Acceptance Test Plan

**Status: SPECIFICATION ONLY.** No ASP.NET source, database schema, or
integration environment exists in this repository. Nothing in this document
has been executed against the real ASP.NET/database integration. It defines
the contract and the acceptance test the responsible ASP.NET/integration team
must run and sign off before this AI service is used in production.

## 1. Roles (unchanged, restated for this contract)

- **ASP.NET** is the system of record: it owns audit records, evidence
  storage, user identity/roles, approvals, and all persistence.
- **FastAPI** (this repository) is a stateless AI analysis service. It has no
  concept of "approved," "published," or "finalized" — those states do not
  exist anywhere in this codebase (confirmed: `app/routers/` exposes only
  `/investigate`, `/analyze-finding`, `/health*`, and OAuth login routes — no
  approval/finalization endpoint). Any such workflow must live in ASP.NET;
  this service returns an opinion for a human to review, never a record with
  legal effect.

## 2. Request/response contract

- **Versioning**: `AnalysisProvenance.canonical`, `.synthesis`, `.remediation`
  each carry `prompt_version`/`schema_version` fields already. ASP.NET should
  store these verbatim per analysis and treat a schema-version change as a
  signal to re-validate its own deserialization, not silently accept a
  structurally different payload.
- **Request**: `InvestigateRequest` (`app/models/agent.py`) — `finding_text`
  plus optional evidence. ASP.NET must validate finding text size and
  encoding before sending (see §4).
- **Response**: `InvestigateResponse` wrapping `InvestigationReport` +
  `AnalysisProvenance` (now including the `autonomy` field from the
  capability-certification framework — see prior session's provenance-wiring
  work). ASP.NET's own DTOs must be validated against this schema on receipt;
  do not assume forward/backward compatibility without an explicit schema
  version check.

## 3. Fields ASP.NET MUST preserve without reinterpretation

Per the existing epistemic-safety architecture, these fields must be
persisted **exactly as returned**, never re-derived or "cleaned up" by
ASP.NET:

- `report.review.required` (always `true` in the current architecture) and
  `report.review.reasons`.
- `report.root_cause.status` and `leading_hypothesis` — do not upgrade
  `NOT_ESTABLISHED` to an established cause in the UI or database merely
  because a human later types a note.
- `report.remediation_cost.*` (`pricing_status`, `estimate_classification`,
  `is_partial_estimate`, `recurring_horizon`, etc.) — the epistemic labels
  computed by the deterministic validator/calculator.
- `provenance.autonomy.eligible` / `.status` / `.blocking_reasons` — an AI
  analysis returned with `eligible=false` must never be silently treated as
  eligible by ASP.NET's own logic.
- Every `source_reference_ids` / evidence identifier — ASP.NET's own record
  IDs and this service's `E<n>`/`FINDING` identifiers are **different
  namespaces**; do not conflate them. If ASP.NET re-keys evidence into its
  own ID scheme, it must keep an explicit mapping table and never let a
  cross-namespace ID collision silently resolve to the wrong evidence item.

## 4. Required behaviors ASP.NET must implement (not verifiable from this repo)

1. Validate all incoming/outgoing payloads against the versioned schema
   before use; reject on mismatch rather than coercing.
2. Treat every AI response as a **proposal**, not a record — approval/
   publication remains an explicit, separately-authorized human action in
   ASP.NET, gated by ASP.NET's own RBAC (see the proposed role model below).
3. **Idempotency**: define an idempotency key for `/investigate` submissions
   (e.g. hash of finding text + evidence set + timestamp bucket) so a network
   retry does not create two persisted analyses for one real submission.
4. **Transactions**: persist `InvestigationReport` + `AnalysisProvenance`
   atomically — a partial write (report saved, provenance lost, or vice
   versa) must not be possible.
5. **Timeout/failure handling**: on a FastAPI timeout, 5xx, or schema-invalid
   response, ASP.NET must record the finding as "AI analysis unavailable /
   review required," never fabricate a placeholder analysis.
6. **Traceability**: persist a link from the finding → evidence set → this
   AI analysis → eventual auditor decision, so any of the four can be
   traced from any other.
7. **Logging**: never log the internal API key, OAuth tokens, or the full
   evidence/finding text at a log level accessible to unauthorized staff.

## 5. Proposed RBAC model (proposal only — requires business sign-off)

This FastAPI service has no user-role concept; it is a single service-to-
service credential. The following roles are a **proposed** model for
ASP.NET's actual approval/access workflow — not something this codebase
enforces or assumes is already approved:

| Role | Can view AI analysis | Can request new analysis | Can approve/publish | Can administer config |
|---|---|---|---|---|
| Auditor | ✅ | ✅ | ❌ | ❌ |
| Reviewer/Approver | ✅ | ❌ | ✅ | ❌ |
| Administrator | ✅ | ✅ | ✅ | ✅ |
| AI service (this FastAPI) | n/a — stateless | n/a | ❌ (no such endpoint exists) | ❌ |

This service's own `/api/v1/provider` (LLM provider switch) is now gated by
the same `INTERNAL_API_KEY` as `/investigate` — it should additionally be
restricted, at the ASP.NET/administrative layer, to Administrator-role
callers only once ASP.NET has a role concept to enforce that with.

## 6. Required round-trip acceptance test (to be executed by the integration team)

1. Submit a representative finding + evidence set through the real ASP.NET
   workflow.
2. Obtain the AI analysis via the real service boundary (not a local
   FastAPI TestClient).
3. Persist `InvestigationReport` + `AnalysisProvenance` in the real database.
4. Read the persisted record back.
5. Diff every field listed in §3 between what FastAPI returned and what was
   read back — byte-identical for epistemic/status fields, no silent
   reinterpretation.
6. Confirm every evidence reference in the persisted report still resolves
   in ASP.NET's evidence store.
7. Confirm `review.required` is still `true` and no approval/finalization
   flag was set automatically.
8. Submit the same finding twice within the idempotency window; confirm
   exactly one analysis is persisted (or a defined, documented conflict
   response).
9. Kill the FastAPI connection mid-request; confirm no partial/misleading
   record is left in the database.
10. Record pass/fail for each step. **This test has not been run** — no
    ASP.NET/database environment was available in this session.
