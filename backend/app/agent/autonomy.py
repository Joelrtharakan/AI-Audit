"""Runtime AUTONOMOUS CAPABILITY controller (spec §11, §33-§34).

Reuses the existing architecture end to end:
    - `app.services.llm.execution.LLMExecutionConfig` identifies the ONE
      provider+model an investigation actually ran with (no fan-out).
    - `scripts/certify_semantic_model.py` runs the EXISTING, unmodified
      held-out benchmark and produces a `ModelCapabilityCertificate`
      (see `app.models.autonomy`).
    - `app.agent.invariants.evaluate_all_invariants` / the causal graph /
      the evidence ledger supply the runtime structural, epistemic and
      causal-consistency signals this controller consumes -- it never
      re-derives them from finding text.

FAIL-CLOSED DEFAULT (spec §34): with no certificate, no matching identity, or
any missing runtime signal, the decision is NOT_CERTIFIED / NOT_ELIGIBLE.
Nothing here can upgrade `review_required` to False -- that field is fixed at
True and exists only so callers get an explicit, typed confirmation that the
existing human-review gate is untouched.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.models.autonomy import (
    TASK_CAPABILITY_REQUIREMENTS,
    AutonomyBlockingReason,
    AutonomyDecision,
    AutonomyStatus,
    CapabilityDimension,
    CapabilityResult,
    ExecutionConfigIdentity,
    ModelCapabilityCertificate,
)

_STORE_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "capability_certificates"


class CertificateStore:
    """File-based store keyed by the certificate's EXACT identity hash.

    Looking a certificate up is looking it up BY identity -- there is no
    "closest match" or "same model, different version is probably fine"
    logic. A provider/model/version/schema/contract/benchmark change simply
    produces a different key, which is not present, which IS the
    invalidation behavior required by spec §13/§14 -- no separate
    invalidation code path is needed.
    """

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or _STORE_DIR

    def _path(self, identity: ExecutionConfigIdentity) -> Path:
        return self.directory / f"{identity.identity_key()}.json"

    def save(self, certificate: ModelCapabilityCertificate) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(certificate.identity)
        path.write_text(json.dumps(certificate.model_dump(), indent=2, default=str))
        return path

    def load(self, identity: ExecutionConfigIdentity) -> ModelCapabilityCertificate | None:
        path = self._path(identity)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return ModelCapabilityCertificate.model_validate(data)
        except Exception:
            # A corrupt/unreadable certificate file is treated exactly like
            # no certificate at all -- fail-closed, never a crash and never
            # a silent "assume valid".
            return None


_default_store = CertificateStore()


def get_certificate_store() -> CertificateStore:
    return _default_store


def required_capabilities_for(tasks: list[str]) -> list[CapabilityDimension]:
    """Structural lookup ONLY -- from the fixed task/workflow table (spec
    §8), never from inspecting finding text (spec §35)."""
    seen: list[CapabilityDimension] = []
    for t in tasks:
        for dim in TASK_CAPABILITY_REQUIREMENTS.get(t, []):
            if dim not in seen:
                seen.append(dim)
    return seen


def evaluate_autonomy(
    identity: ExecutionConfigIdentity,
    required_tasks: list[str],
    *,
    evidence_sufficient: bool = True,
    structural_safety_ok: bool = True,
    epistemic_consistent: bool = True,
    causal_consistent: bool = True,
    provenance_ok: bool = True,
    store: CertificateStore | None = None,
) -> AutonomyDecision:
    """The autonomy controller (spec §33). Operates ONLY on structured state:
    an execution-config identity, a list of task names, and pre-computed
    boolean runtime signals the caller derives from the canonical state /
    evidence ledger / causal graph / provenance record using the EXISTING
    deterministic validators. Never touches raw finding text.

    Default is NOT_ELIGIBLE / NOT_CERTIFIED (spec §34, §16) -- an eligible
    result requires every one of: a certificate for this EXACT identity,
    that certificate's SEMANTIC_SAFETY_GATE == PASS, every required
    capability dimension == PASS on that certificate, and every runtime
    signal True.
    """
    store = store or _default_store
    required = required_capabilities_for(required_tasks)
    blocking: list[str] = []
    capability_results: dict[str, str] = {}
    safety_results = {
        "evidence_sufficient": evidence_sufficient,
        "structural_safety_ok": structural_safety_ok,
        "epistemic_consistent": epistemic_consistent,
        "causal_consistent": causal_consistent,
        "provenance_ok": provenance_ok,
    }

    cert = store.load(identity)
    if cert is None:
        return AutonomyDecision(
            eligible=False,
            status=AutonomyStatus.NOT_CERTIFIED.value,
            required_capabilities=[d.value for d in required],
            capability_results={},
            safety_results=safety_results,
            blocking_reasons=[AutonomyBlockingReason.MODEL_NOT_CERTIFIED.value],
            certificate_identity=identity,
            certificate_id=None,
            review_required=True,
        )

    if cert.semantic_safety_gate != "PASS":
        blocking.append(AutonomyBlockingReason.SEMANTIC_SAFETY_FAILED.value)

    for dim in required:
        result = cert.capability(dim)
        capability_results[dim.value] = result.value
        if result != CapabilityResult.PASS:
            if AutonomyBlockingReason.REQUIRED_CAPABILITY_NOT_CERTIFIED.value not in blocking:
                blocking.append(AutonomyBlockingReason.REQUIRED_CAPABILITY_NOT_CERTIFIED.value)

    if not evidence_sufficient:
        blocking.append(AutonomyBlockingReason.EVIDENCE_INSUFFICIENT.value)
    if not causal_consistent:
        blocking.append(AutonomyBlockingReason.CAUSAL_EVIDENCE_INSUFFICIENT.value)
    if not epistemic_consistent:
        blocking.append(AutonomyBlockingReason.EPISTEMIC_CONSISTENCY_FAILED.value)
    if not structural_safety_ok or not provenance_ok:
        blocking.append(AutonomyBlockingReason.STRUCTURED_OUTPUT_INVALID.value)

    eligible = not blocking
    status = (
        AutonomyStatus.ELIGIBLE.value if eligible
        else AutonomyStatus.NOT_ELIGIBLE.value
    )

    return AutonomyDecision(
        eligible=eligible,
        status=status,
        required_capabilities=[d.value for d in required],
        capability_results=capability_results,
        safety_results=safety_results,
        blocking_reasons=blocking,
        certificate_identity=identity,
        certificate_id=cert.certificate_id,
        review_required=True,
        generated_at=time.time(),
    )


_digest_cache: dict[str, str] = {}


def _cached_model_digest(settings) -> str:
    """Same live Ollama /api/tags digest lookup `scripts/certify_semantic_
    model.py`'s preflight already performs during certification -- cached
    per (base_url, model) for the life of the process so a per-request
    provenance derivation never adds a network round trip on the hot path.
    Fails closed to "" (never raises, never blocks the pipeline)."""
    if settings.llm_provider != "ollama":
        return ""
    key = f"{settings.ollama_base_url}|{settings.ollama_model}"
    if key in _digest_cache:
        return _digest_cache[key]
    digest = ""
    try:
        import httpx
        base = (settings.ollama_base_url or "http://localhost:11434").rstrip("/")
        r = httpx.get(f"{base}/api/tags", timeout=2.0)
        r.raise_for_status()
        for m in r.json().get("models", []):
            if m.get("name") == settings.ollama_model:
                digest = (m.get("digest") or "")[:24]
                break
    except Exception:
        digest = ""
    _digest_cache[key] = digest
    return digest


def _canonical_prompt_sha256(settings, filename: str = "canonical_finding_interpretation_system_prompt.txt") -> str:
    """Full SHA-256 of the exact bytes of the canonical interpretation system
    prompt. Fail-closed: an unreadable prompt yields a fixed sentinel, which
    matches no certificate (certificates are only saved for a readable prompt)."""
    import hashlib

    try:
        path = settings.prompts_dir / filename
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return "PROMPT_UNREADABLE"


def build_current_execution_identity(*, model_version: str | None = None) -> ExecutionConfigIdentity:
    """Builds the `ExecutionConfigIdentity` for the CURRENTLY CONFIGURED
    provider/model, using the EXACT SAME hash formulas
    `scripts/certify_semantic_model.py` uses when it saves a certificate --
    so a certificate generated for a configuration is found by a runtime
    lookup for that same configuration (spec §2, §13). `model_version` lets
    a caller that already resolved the live digest (the certification run
    itself) supply it directly instead of a fresh lookup.
    """
    import hashlib

    from app.config import get_settings

    s = get_settings()
    model = s.ollama_model if s.llm_provider == "ollama" else (s.llm_model or "")
    resolved_version = model_version if model_version is not None else _cached_model_digest(s)
    # The prompt VERSION strings stay in the identity, and the exact bytes of the
    # canonical system prompt are bound in too: editing the prompt text without
    # bumping a version can no longer leave an old certificate matching.
    inference_fields = "|".join(str(x) for x in (
        s.canonical_semantic_prompt_version, s.remediation_cost_prompt_version, s.analysis_prompt_version,
        "canonical_prompt_sha256=" + _canonical_prompt_sha256(s),
        "remediation_prompt_sha256=" + _canonical_prompt_sha256(s, "remediation_cost_interpretation_system_prompt.txt"),
    ))
    root = Path(__file__).resolve().parent.parent.parent
    schema_bytes = (root / "app" / "services" / "canonical_semantic_models.py").read_bytes()
    benchmark_path = root / "tests" / "certification" / "semantic_capability_benchmark.py"
    benchmark_hash = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()[:16] if benchmark_path.exists() else ""
    return ExecutionConfigIdentity(
        provider=s.llm_provider,
        model=model,
        model_version=resolved_version,
        semantic_schema_version=hashlib.sha256(schema_bytes).hexdigest()[:16],
        contract_version=hashlib.sha256(inference_fields.encode()).hexdigest()[:16],
        inference_config_hash=hashlib.sha256(str(s.ollama_base_url or "").encode()).hexdigest()[:16],
        benchmark_hash=benchmark_hash,
    )
