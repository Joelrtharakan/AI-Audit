"""Phase 21/23: LLM-backed evidence -> claim interpretation.

Bridges the Phase 19 adaptive evidence loop (app.agent.nodes.
evidence_acquisition) to the pre-existing canonical claim model
(app.models.agent.EvidenceClaim) and conflict machinery (app.agent.
claim_extractor.detect_evidence_conflicts) -- this module does NOT
introduce a second claim representation or a second contradiction
representation.

    EvidenceItem
        -> EvidenceInterpreter.interpret()
            -> LLMProvider.generate()   (provider-agnostic; Ollama/Copilot/...)
            -> strict JSON schema validation
            -> EvidenceClaim[]          (rejected/malformed output -> [])
        -> derive_hypothesis_relevance()   (deterministic, no LLM authority)

The LLM is responsible ONLY for semantic interpretation of one evidence
item's text. It cannot assign CandidateHypothesis.status, create causal
edges, or decide root cause -- see derive_hypothesis_relevance below and
app.agent.nodes.evidence_acquisition.reconcile_hypothesis_from_evidence
(the single authoritative evidence->status evaluator, unchanged in
authority by this module).

Phase 23 root-cause fix: Phase 21/22 asked the LLM for ONE field
(`epistemic_class`, values OBSERVED/REPORTED/SUPPORTING/CONTRADICTING/
UNKNOWN) that had to simultaneously answer two different questions --
"what kind of evidence is this" (directness/verification level) and
"does it bear on this hypothesis" (relevance). Live testing against real
qwen3:8b showed the model consistently defaulted to the easier
descriptive answer (OBSERVED) rather than committing to the harder
relational judgment, so SUPPORTING/CONTRADICTING were rarely produced and
`derive_hypothesis_relevance` correctly (but unhelpfully) returned
INSUFFICIENT almost every time. The fix is NOT a stronger prompt alone --
it is separating the two questions into two fields
(EvidenceClaim.status/epistemic_class for verification level, the new
EvidenceClaim.hypothesis_relations for per-hypothesis relevance) so the
model is never forced to overload one answer with two purposes.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from app.agent.claim_extractor import _classify_polarity
from app.models.agent import (
    ClaimAttribution,
    EvidenceClaim,
    EvidenceProposition,
    EvidenceStatus,
    HypothesisRelation,
    QuantitativeAssertion,
    TemporalRelation,
)
from app.services.llm.json_parser import parse_llm_json
from app.services.text_grounding import significant_words

logger = logging.getLogger(__name__)

_VALID_RELATIONS = {"SUPPORTING", "CONTRADICTING", "NEUTRAL", "INSUFFICIENT"}
_RELEVANCE_CLASSES = {"SUPPORTING", "CONTRADICTING"}
# Backward-compat only: pre-Phase-23 callers/tests may construct EvidenceClaim
# directly with a top-level epistemic_class in this vocabulary (never
# produced by the interpreter itself anymore -- see _epistemic_class_for_status).
_LEGACY_RELEVANCE_EPISTEMIC_CLASSES = {"SUPPORTING", "CONTRADICTING"}
_VALID_PROPOSITION_TYPES = {e.value for e in EvidenceProposition}
_VALID_TEMPORAL_RELATIONS = {e.value for e in TemporalRelation}
_VALID_CAUSAL_BASES = {"INDEPENDENT_EVIDENCE", "TEMPORAL_ONLY", "NOT_APPLICABLE"}
_QUANT_OPS = {
    "GT": lambda l, r: l > r, "LT": lambda l, r: l < r, "EQ": lambda l, r: l == r,
    "NE": lambda l, r: l != r, "GE": lambda l, r: l >= r, "LE": lambda l, r: l <= r,
}

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def verify_quantitative(q: QuantitativeAssertion | None) -> bool:
    """Phase 24 Part F: independently verify a claim's OWN stated numeric
    comparison is arithmetically consistent. This NEVER invents a threshold
    or infers a direction -- it only checks whether the operator the LLM
    (or extraction) reported actually holds for the two numbers it also
    reported, both taken verbatim from the evidence text. A claim whose
    stated arithmetic doesn't hold up is not safe to rely on for a
    relational judgment, regardless of what relation was proposed."""
    if q is None:
        return True  # nothing to verify -- not a quantitative claim
    op = _QUANT_OPS.get(q.operator)
    if op is None:
        return False
    return bool(op(q.left, q.right))


def validate_relation(
    llm_relation: str,
    claim_status: EvidenceStatus,
    proposition_type: EvidenceProposition | None,
    quantitative_valid: bool = True,
    causal_basis: str = "NOT_APPLICABLE",
) -> tuple[str, str]:
    """Phase 24/25 Part D: the deterministic relation-validation firewall.
    LLM_RELATION + STRUCTURED_EVIDENCE -> VALIDATED_RELATION. Operates only
    on typed structures (status enum, proposition-type enum, arithmetic
    consistency, causal-basis enum) -- never on domain vocabulary. Returns
    (validated_relation, decision) where decision is one of ACCEPT /
    DOWNGRADE_TO_NEUTRAL / DOWNGRADE_TO_INSUFFICIENT / REJECT.

    Rules (domain-general, structural only):
      1. An unrecognized relation value is REJECTed outright -> INSUFFICIENT.
      2. A claim whose own stated arithmetic doesn't check out cannot be
         trusted for a relational judgment -> DOWNGRADE_TO_INSUFFICIENT.
      3. Part B/E's core rule: ABSENCE_OF_EVIDENCE (a search/record found
         nothing) can NEVER by itself license SUPPORTING/CONTRADICTING --
         only EVIDENCE_OF_ABSENCE (an evidence text that itself establishes
         the source is authoritative/exhaustive) can. This is the exact
         fix for the Phase 23 "missing_evidence -> SUPPORTING" defect,
         applied structurally (via the LLM's own proposition_type
         classification) rather than via a phrase list.
      4. A claim built on evidence of UNKNOWN status classified as
         SUPPORTING/CONTRADICTING is downgraded to INSUFFICIENT -- an
         epistemic status the system itself doesn't understand cannot
         safely ground a relational judgment either.
      5. Phase 25 Rule 6, the temporal-causality firewall: a relation whose
         ONLY basis is temporal sequence (causal_basis == TEMPORAL_ONLY) is
         downgraded to NEUTRAL, never REJECTed/INSUFFICIENT -- the temporal
         fact itself is real and verified, it simply cannot alone establish
         a causal SUPPORTING/CONTRADICTING relation. Independent causal
         evidence (causal_basis == INDEPENDENT_EVIDENCE) is unaffected --
         this rule does NOT overcorrect legitimate causal evidence that
         happens to include temporal information (Rule 6's explicit
         "do not overcorrect" instruction).
      6. Otherwise ACCEPT the LLM's proposed relation unchanged.
    """
    relation = str(llm_relation or "").strip().upper()
    if relation not in _VALID_RELATIONS:
        return "INSUFFICIENT", "REJECT"

    if relation in _RELEVANCE_CLASSES and not quantitative_valid:
        return "INSUFFICIENT", "DOWNGRADE_TO_INSUFFICIENT"

    if relation in _RELEVANCE_CLASSES and proposition_type == EvidenceProposition.ABSENCE_OF_EVIDENCE:
        return "INSUFFICIENT", "DOWNGRADE_TO_INSUFFICIENT"

    if relation in _RELEVANCE_CLASSES and claim_status == EvidenceStatus.UNKNOWN:
        return "INSUFFICIENT", "DOWNGRADE_TO_INSUFFICIENT"

    if relation in _RELEVANCE_CLASSES and causal_basis == "TEMPORAL_ONLY":
        return "NEUTRAL", "DOWNGRADE_TO_NEUTRAL"

    return relation, "ACCEPT"


def extract_numbers(text: str) -> list[float]:
    """Phase 23 Part F: deterministic numeric extraction for observability/
    traceability -- the LLM is instructed never to invent or compute
    numbers; this lets calling code see exactly which numeric literals a
    claim actually cites, verifiable independently of the model's prose."""
    out = []
    for m in _NUMBER_RE.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def _cap_status(evidence_item_status: Any) -> EvidenceStatus:
    """Phase 23: claim evidentiary weight is now derived PURELY from the
    evidence item's own verification level -- never from anything the LLM
    says (removes the Phase 21 dependency on an LLM-supplied
    epistemic_class for this axis entirely, closing off a second path by
    which a REPORTED item could have been upgraded).

    Phase 24: uses EXACT equality, never `str(x).endswith("VERIFIED")` --
    that substring check is a real, silent bug elsewhere in this codebase
    (str(EvidenceStatus.UNVERIFIED) == "EvidenceStatus.UNVERIFIED", and
    "UNVERIFIED".endswith("VERIFIED") is True in Python), which would have
    let genuinely UNVERIFIED evidence be capped as VERIFIED here."""
    if evidence_item_status == EvidenceStatus.VERIFIED:
        return EvidenceStatus.VERIFIED
    if evidence_item_status == EvidenceStatus.REPORTED:
        return EvidenceStatus.REPORTED
    return EvidenceStatus.UNKNOWN


def _epistemic_class_for_status(status: EvidenceStatus) -> str:
    """Backward-compat derived value for EvidenceClaim.epistemic_class
    (Phase 21 vocabulary) -- deterministic, never LLM-controlled."""
    if status == EvidenceStatus.VERIFIED:
        return "OBSERVED"
    if status == EvidenceStatus.REPORTED:
        return "REPORTED"
    return "UNKNOWN"


def _attribution_for(status: EvidenceStatus) -> ClaimAttribution:
    if status == EvidenceStatus.VERIFIED:
        return ClaimAttribution.SYSTEM_EVIDENCE
    if status == EvidenceStatus.REPORTED:
        return ClaimAttribution.PERSON_REPORTED
    return ClaimAttribution.UNKNOWN


class EvidenceInterpreter:
    """Provider-agnostic evidence interpreter. Depends only on the
    app.services.llm.base.LLMProvider abstraction -- never instantiates a
    concrete provider (e.g. Ollama) directly, so swapping the configured
    provider (LLM_PROVIDER env var / get_llm_provider()) requires no
    change here. Every provider capability difference (structured output
    support, temperature, timeouts, ...) is handled inside the provider
    implementation, never here (Part N)."""

    def __init__(self, llm_provider: Any = None, node_name: str = "evidence_interpretation"):
        self._llm_provider = llm_provider
        self._node_name = node_name

    def _provider(self) -> Any:
        if self._llm_provider is not None:
            return self._llm_provider
        from app.services.llm.factory import get_llm_provider

        return get_llm_provider()

    async def interpret(
        self,
        evidence_item: Any,
        # Backward-compat positional signature preserved exactly (Phase
        # 21/22 call sites/tests: interpret(item, statement, hyp_id)).
        hypothesis_statement: str = "",
        hypothesis_id: str = "",
        question: str = "",
        *,
        # New in Phase 23 Part K: classify one evidence item against
        # SEVERAL hypotheses in a single call. Takes priority over
        # hypothesis_statement/hypothesis_id when given.
        hypotheses: list[dict] | None = None,
    ) -> list[EvidenceClaim]:
        """Return EvidenceClaim[] extracted from `evidence_item`, each
        carrying a `hypothesis_relations` entry for every hypothesis in
        `hypotheses` (or the single legacy hypothesis_id/hypothesis_statement
        pair). Never raises -- any provider failure, malformed output, or
        missing provenance results in an empty/partial list (fail closed),
        never a fabricated claim (Part O: provider failure != evidence)."""
        if hypotheses is None:
            hypotheses = [{"id": hypothesis_id, "statement": hypothesis_statement}] if hypothesis_id else []
        hyp_ids = {str(h.get("id")) for h in hypotheses if h.get("id")}

        evidence_id = getattr(evidence_item, "evidence_id", None)
        claim_text = getattr(evidence_item, "claim", "") or ""
        metadata = {
            "provider": type(self._provider()).__name__ if self._llm_provider is not None else None,
            "evidence_id": evidence_id, "hypothesis_ids": sorted(hyp_ids),
        }
        if not evidence_id or not claim_text.strip() or not hyp_ids:
            # No provenance to attach claims to, nothing to interpret, or no
            # hypothesis to classify against -- fail closed rather than
            # fabricate or guess (Part L: evidence_id/hypothesis_id required).
            return []

        prompt = self._build_prompt(evidence_item, hypotheses, question)

        import time
        t_start = time.monotonic()
        try:
            provider = self._provider()
            response = await provider.generate(
                node=self._node_name,
                prompt=prompt,
                temperature=0.0,
                response_format="json",
                max_output_tokens=800,
            )
            data = parse_llm_json(response.content)
        except Exception as exc:  # provider unavailable/timeout/malformed -- fail safe
            logger.warning(
                "event=evidence_interpretation_failure evidence_id=%s hypothesis_ids=%s "
                "failure_type=%s latency_ms=%d",
                evidence_id, sorted(hyp_ids), type(exc).__name__,
                int((time.monotonic() - t_start) * 1000),
            )
            return []
        latency_ms = int((time.monotonic() - t_start) * 1000)

        raw_claims = data.get("claims")
        if not isinstance(raw_claims, list):
            logger.warning("event=evidence_interpretation_malformed evidence_id=%s reason=no_claims_array", evidence_id)
            return []

        claims: list[EvidenceClaim] = []
        rejected = 0
        for idx, raw in enumerate(raw_claims):
            claim = self._validate_and_build_claim(raw, evidence_item, hyp_ids, idx)
            if claim is not None:
                claims.append(claim)
            else:
                rejected += 1

        relation_counts: dict[str, int] = {}
        for c in claims:
            for r in c.hypothesis_relations:
                relation_counts[r.relation] = relation_counts.get(r.relation, 0) + 1
        logger.info(
            "event=evidence_interpretation_complete evidence_id=%s hypothesis_ids=%s "
            "claim_count=%d accepted_claim_count=%d rejected_claim_count=%d relation_counts=%s "
            "latency_ms=%d provider=%s",
            evidence_id, sorted(hyp_ids), len(raw_claims), len(claims), rejected, relation_counts,
            latency_ms, metadata["provider"],
        )
        return claims

    def _build_prompt(self, evidence_item: Any, hypotheses: list[dict], question: str) -> str:
        from app.config import get_settings

        settings = get_settings()
        template = (settings.agent_prompts_dir / "evidence_interpretation.txt").read_text(encoding="utf-8")
        hypotheses_block = "\n".join(
            f"- id={h.get('id')}: {h.get('statement') or '(no statement)'}"
            + (f" [current status: {h.get('status')}]" if h.get("status") else "")
            for h in hypotheses
        ) or "(none provided)"
        return template.format(
            hypotheses_block=hypotheses_block,
            question=question or "(none stated)",
            evidence_source=getattr(evidence_item, "source", "") or "unknown",
            evidence_text=getattr(evidence_item, "claim", "") or "",
        )

    def _validate_and_build_claim(
        self, raw: Any, evidence_item: Any, known_hypothesis_ids: set[str], idx: int,
    ) -> EvidenceClaim | None:
        if not isinstance(raw, dict):
            return None
        text = str(raw.get("text") or "").strip()
        source_reference = str(raw.get("source_reference") or "").strip()

        if not text:
            return None
        if not source_reference:
            logger.info("event=claim_rejected evidence_id=%s reason=missing_source_reference",
                        getattr(evidence_item, "evidence_id", None))
            return None

        evidence_id = getattr(evidence_item, "evidence_id", None)
        if not evidence_id:
            # Dangling / provenance-less claim -- fail closed (Part L).
            return None

        # Grounding check: reject claims whose text has no meaningful overlap
        # with the source evidence text -- a cheap, domain-general defense
        # against hallucinated content the strict schema check alone can't
        # catch.
        evidence_words = significant_words(getattr(evidence_item, "claim", "") or "")
        claim_words = significant_words(text)
        if evidence_words and claim_words and not (claim_words & evidence_words):
            logger.info("event=claim_rejected evidence_id=%s reason=ungrounded", evidence_id)
            return None

        status = _cap_status(getattr(evidence_item, "status", EvidenceStatus.UNVERIFIED))
        attribution = _attribution_for(status)

        # Phase 24 Part C: structural proposition-type classification --
        # the LLM's own value, validated against the known enum (unknown/
        # missing -> None, never guessed).
        prop_type_raw = str(raw.get("proposition_type") or "").strip().upper()
        proposition_type = EvidenceProposition(prop_type_raw) if prop_type_raw in _VALID_PROPOSITION_TYPES else None

        # Phase 24 Part G: structural temporal relation, same treatment.
        temporal_raw = str(raw.get("temporal_relation") or "").strip().upper()
        temporal_relation = TemporalRelation(temporal_raw) if temporal_raw in _VALID_TEMPORAL_RELATIONS else None

        # Phase 24 Part F: quantitative assertion -- only trusted if BOTH
        # numbers and a recognized operator are present and internally
        # consistent (verify_quantitative). An internally-inconsistent
        # quantitative claim is dropped entirely (never kept in a form that
        # could be misread as verified arithmetic) and flagged for the
        # relation firewall below.
        quantitative: QuantitativeAssertion | None = None
        quantitative_valid = True
        raw_quant = raw.get("quantitative")
        if isinstance(raw_quant, dict) and raw_quant.get("left") is not None and raw_quant.get("right") is not None:
            try:
                candidate = QuantitativeAssertion(
                    left=float(raw_quant.get("left")),
                    operator=str(raw_quant.get("operator") or "").strip().upper(),
                    right=float(raw_quant.get("right")),
                    unit=str(raw_quant.get("unit") or "").strip() or None,
                )
            except (ValueError, TypeError):
                quantitative_valid = False
            else:
                if verify_quantitative(candidate):
                    quantitative = candidate
                else:
                    quantitative_valid = False
                    logger.info("event=quantitative_rejected evidence_id=%s reason=arithmetic_inconsistent "
                                "left=%s operator=%s right=%s",
                                evidence_id, raw_quant.get("left"), raw_quant.get("operator"), raw_quant.get("right"))

        # Phase 23 Part D, hardened Phase 24 Part D: validate
        # hypothesis_relations against the actual known hypothesis ids
        # (an entry naming an unknown id is dropped, never silently
        # trusted) AND run the surviving relation through the deterministic
        # firewall (validate_relation) -- the claim's own proposition_type/
        # status/quantitative-consistency can downgrade an LLM-proposed
        # SUPPORTING/CONTRADICTING to INSUFFICIENT.
        relations: list[HypothesisRelation] = []
        raw_relations = raw.get("hypothesis_relations")
        if isinstance(raw_relations, list):
            for rr in raw_relations:
                if not isinstance(rr, dict):
                    continue
                hid = str(rr.get("hypothesis_id") or "").strip()
                proposed_relation = str(rr.get("relation") or "").strip().upper()
                if not hid or hid not in known_hypothesis_ids:
                    if hid:
                        logger.info("event=relation_rejected evidence_id=%s reason=unknown_hypothesis_id hypothesis_id=%s",
                                    evidence_id, hid)
                    continue
                causal_basis = str(rr.get("causal_basis") or "").strip().upper()
                if causal_basis not in _VALID_CAUSAL_BASES:
                    causal_basis = "NOT_APPLICABLE"
                validated_relation, decision = validate_relation(
                    proposed_relation, status, proposition_type, quantitative_valid, causal_basis,
                )
                if decision != "ACCEPT":
                    logger.info(
                        "event=relation_validated evidence_id=%s hypothesis_id=%s proposed=%s "
                        "validated=%s decision=%s proposition_type=%s causal_basis=%s",
                        evidence_id, hid, proposed_relation, validated_relation, decision,
                        proposition_type.value if proposition_type else None, causal_basis,
                    )
                relations.append(HypothesisRelation(
                    hypothesis_id=hid, relation=validated_relation,
                    reason=str(rr.get("reason") or "").strip() or None,
                    validation_decision=decision,
                    causal_basis=causal_basis,
                ))

        return EvidenceClaim(
            claim_id=f"C_{evidence_id}_{idx}_{uuid.uuid4().hex[:6]}",
            text=text,
            subject=str(raw.get("subject") or "").strip() or None,
            predicate=str(raw.get("predicate") or "").strip() or None,
            source=getattr(evidence_item, "source", "") or "unknown",
            status=status,
            evidence_reference=evidence_id,
            evidence_ids=[evidence_id],
            hypothesis_ids=sorted({r.hypothesis_id for r in relations}),
            hypothesis_relations=relations,
            attribution=attribution,
            polarity=_classify_polarity(text),
            source_type="AI_INTERPRETED_EVIDENCE",
            epistemic_class=_epistemic_class_for_status(status),
            temporal_context=str(raw.get("temporal_context") or "").strip() or None,
            proposition_type=proposition_type,
            quantitative=quantitative,
            temporal_relation=temporal_relation,
            extraction_status="EXTRACTED",
        )


def derive_hypothesis_relevance(claims: list[EvidenceClaim], hypothesis_id: str | None = None) -> str | None:
    """Deterministic, domain-general aggregation of a claim's relation to
    ONE hypothesis into an EvidenceItem-relevance-style value. The LLM
    only tags individual claim<->hypothesis relations; THIS function --
    not the LLM -- decides what the evidence as a whole means for the
    hypothesis (Part J: "The interpreter only produces structured evidence
    interpretation. The reconciliation layer remains the only authority.").

    Phase 23: reads the explicit, per-hypothesis `hypothesis_relations`
    when present (filtered to `hypothesis_id` when given -- Part K: the
    same claim may vote differently for different hypotheses). Falls back
    to the legacy top-level `epistemic_class` for claims that predate this
    field (backward compatibility with pre-Phase-23 callers/tests that
    construct EvidenceClaim directly).

    Returns None when no claims exist (caller should leave the item's
    existing relevance, e.g. provider-set UNRESOLVED/UNAVAILABLE, alone).
    """
    if not claims:
        return None
    votes: set[str] = set()
    for c in claims:
        relations = getattr(c, "hypothesis_relations", None) or []
        matched = [r for r in relations if hypothesis_id is None or r.hypothesis_id == hypothesis_id]
        if matched:
            for r in matched:
                if r.relation in _RELEVANCE_CLASSES:
                    votes.add(r.relation)
        elif not relations and c.epistemic_class in _LEGACY_RELEVANCE_EPISTEMIC_CLASSES:
            votes.add(c.epistemic_class)
    if "SUPPORTING" in votes and "CONTRADICTING" in votes:
        return "CONFLICTING"
    if "SUPPORTING" in votes:
        return "SUPPORTING"
    if "CONTRADICTING" in votes:
        return "CONTRADICTING"
    # Claims exist but none votes SUPPORTING/CONTRADICTING for this
    # hypothesis (NEUTRAL/INSUFFICIENT only, or no relation data at all) --
    # evidence was retrieved but is insufficient to move this hypothesis,
    # not proof of anything (Part I: absence firewall).
    return "INSUFFICIENT"
