"""Post-generation hallucination/contamination guard for the /investigate
agent pipeline.

This is the code-level enforcement layer the prompt-only approach cannot
provide: even if an LLM call ignores its instructions and copies an entity,
number, or identifier from a different case (or invents one), this module
mechanically detects it by tracing every named-entity-shaped token and every
number in generated text back to THIS request's own finding text and evidence
ledger. Anything that doesn't trace back is stripped or the field is replaced
with a safe, generic fallback — never silently passed through.

Deliberately reuses the same entity/number primitives already proven out in
the analyze-finding pipeline's grounding_validator.py / text_grounding.py,
rather than inventing a second detector with different behavior.
"""

from __future__ import annotations

import re

from app.models.agent import EvidenceItem, EvidenceStatus
from app.services.text_grounding import entity_is_grounded, significant_words

# Named-entity-shaped tokens: IDs like "SOP-OPS-014", acronyms like "LIMS".
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)+\b|\b[A-Z]{2,}\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
# Spelled-out counts ("three operators") that a hallucination/contamination can
# introduce without tripping the digit check above. Deliberately excludes
# "one"/"two", which are common as generic words/pronouns and would produce
# too many false positives (e.g. "one of the records").
_WORD_NUMBER_RE = re.compile(r"\b(three|four|five|six|seven|eight|nine|ten)\b", re.IGNORECASE)

# Causal language that must never be used to describe an UNVERIFIED cause —
# this is the mechanical half of "reported + reported != established cause".
# Deliberately excludes the bare noun "cause"/"causes" (as in "root cause",
# "possible cause") -- only verb/connector forms that assert causation count.
CAUSAL_RE = re.compile(
    r"\b(caused|causing|led to|leads to|resulted? in|resulting in|due to|because of)\b",
    re.IGNORECASE,
)

SAFE_ROOT_CAUSE_FALLBACK = (
    "The available evidence establishes the observed condition but does not "
    "establish why it occurred."
)

# ---------------------------------------------------------------------------
# Unsupported-domain guard
# ---------------------------------------------------------------------------
# Entity/number grounding (above) cannot catch a model defaulting to its most
# common training data pattern -- "training"/"authorization" questions --
# using only generic vocabulary that contains no invented entity or number.
# Observed in production: a real (unmocked) LLM call for a pure sensor/
# equipment finding with zero mention of training anywhere still generated an
# investigation question about "mandatory training" and an evidence item
# "Training records for Cold Room Operations". These domain-trigger words are
# common enough defaults that if they appear in generated text but nowhere in
# this case's own finding/evidence, the content is almost certainly template
# leakage, not case-specific reasoning.
# Every stem here is a generic QMS root-cause trope a model defaults to
# regardless of finding content -- observed live: a pure calibration-label
# finding (no mention of any auditor interaction at all) still produced a
# hypothesis that the operator "miscommunicated the calibration status to
# the auditor". "communicat"/"supervis"/"workload" are exactly as prone to
# this as "train"/"authoriz" were.
_DOMAIN_TRIGGER_STEMS = (
    "train", "authoriz", "competenc", "communicat", "supervis", "workload",
    "policy", "sop", "procedure", "management", "careless", "human error",
    "resource", "staffing", "maintenance", "vendor", "supplier", "billing",
    "contract", "auditor"
)



# ---------------------------------------------------------------------------
# Prompt-placeholder leakage
# ---------------------------------------------------------------------------
# Observed in production: a weak model, given a JSON schema whose example
# values are fluent English sentences describing what to write, sometimes
# just echoes the instructional sentence itself instead of replacing it --
# e.g. literally outputting "A specific assessment pathway grounded in this
# finding's own affected items/records." as the impact area. Prompts now use
# <<bracketed>> non-natural-language placeholders specifically to make this
# harder, but this is a second, code-level line of defense: these are
# fragments of the actual instructional text used across the agent prompts,
# so if a model reproduces one anyway (verbatim or near-verbatim), it is
# almost certainly copied instruction text, not real analysis, regardless of
# which prompt version is in use.
_PLACEHOLDER_LEAK_FRAGMENTS = (
    "grounded in this finding's own affected items",
    "a second pathway addressing what changed",
    "pathway addressing whether affected outputs",
    "pathway addressing whether retrospective review",
    "the specific control/process area implicated by this finding's own facts",
    "a question about this finding's specific unresolved",
    "the specific record types that would resolve the open questions",
    "clear description of root cause or leading hypothesis",
    "one-sentence statement grounded in this finding's own entities",
    "hypothesis statement using only entities/process elements",
    "concise root cause statement or null if not_established",
)


def is_placeholder_leak(text: str | None, finding_text: str | None = None) -> bool:
    """True if `text` reproduces one of the agent's own prompt-instruction
    fragments verbatim/near-verbatim instead of real, case-specific content,
    OR literally contains the "<<...>>" placeholder-marker syntax itself,
    OR embeds the full finding text verbatim inside placeholder template phrases."""
    if not text:
        return False
    if "<<" in text or ">>" in text:
        return True
    lowered = text.lower()
    if any(fragment in lowered for fragment in _PLACEHOLDER_LEAK_FRAGMENTS):
        return True
    if finding_text and len(finding_text.strip()) > 15:
        norm_finding = finding_text.strip().lower()
        if norm_finding in lowered and len(lowered) > len(norm_finding) + 5:
            # Check for patterns like "records for '<FULL FINDING>'", "process for '<FULL FINDING>'"
            if any(prefix in lowered for prefix in ("for '", "for \"", "for `", "records for", "process for", "activity for")):
                return True
    return False



_DICT_KEY_STRIP_RE = re.compile(r"'(?:[a-zA-Z_][a-zA-Z0-9_]*)':\s*")
_DICT_QUOTE_STRIP_RE = re.compile(r"[{}\[\]']")


def clean_structured_leak(value) -> str:
    if isinstance(value, dict):
        parts = [clean_structured_leak(v) for v in value.values()]
        return "; ".join(p for p in parts if p)
    if isinstance(value, (list, tuple)):
        parts = [clean_structured_leak(v) for v in value]
        return "; ".join(p for p in parts if p)
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value) if value is not None else ""

    # Strip verbatim template prefixes like "Execution records for '...' " or "Operational process for '...' "
    text = re.sub(r"^(Execution|Operational|Primary)\s+(records|process|activity|steps)\s+for\s+['\"`].*?['\"`]\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\bfor\s+['\"`][^'\"]*during\s+the\s+audit[^'\"]*['\"`]", "for the observed nonconformity", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfor\s+['\"`][^'\"]*was\s+found[^'\"]*['\"`]", "for the observed nonconformity", text, flags=re.IGNORECASE)

    stripped = text.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
        stripped = _DICT_KEY_STRIP_RE.sub("", stripped)
        stripped = _DICT_QUOTE_STRIP_RE.sub("", stripped)
        stripped = re.sub(r"\s*,\s*", "; ", stripped).strip("; ").strip()
        return stripped

    return text



def mentions_unsupported_domain(text: str, source_text: str) -> bool:
    """True if `text` invokes a generic QMS root-cause domain (training,
    authorization, competency, communication, supervision, workload) that is
    not supported anywhere in `source_text` (this case's finding + evidence)."""
    if not text:
        return False
    lowered_text = text.lower()
    lowered_source = source_text.lower()

    synonyms = {
        "procedure": ["sop", "instruction", "bmr", "protocol", "sequence", "method", "policy"],
        "sop": ["procedure", "instruction", "bmr", "protocol", "sequence"],
        "policy": ["procedure", "sop", "requirement", "standard"],
        "train": ["competenc", "analyst", "operator", "instruction", "qualif"],
        "competenc": ["train", "qualification", "evaluat", "matrix"],
        "maintenanc": ["calibrat", "service", "work order", "pm-", "sensor"],
        "authoriz": ["approval", "override", "bypass", "payment", "sign-off", "permission"],
        "vendor": ["supplier", "payable", "procurement", "invoice"],
        "supplier": ["vendor", "payable", "procurement", "invoice"],
        "billing": ["invoice", "payment", "payable", "charge"],
    }

    for stem in _DOMAIN_TRIGGER_STEMS:
        if stem in lowered_text:
            # Check direct stem match in source
            if stem in lowered_source:
                continue
            # Check domain synonym matches in source
            domain_syns = synonyms.get(stem, [])
            if any(syn in lowered_source for syn in domain_syns):
                continue
            return True

    return False



def build_source_text(
    finding_text: str,
    evidence_ledger: list[EvidenceItem] | None = None,
    extra_trusted_text: list[str] | None = None,
) -> str:
    """Everything a generated claim is allowed to trace back to for THIS case:
    the finding text, every evidence ledger entry (regardless of status —
    REPORTED entities are still real entities, just not verified causes), and
    optionally other already-grounded text produced earlier in this same
    pipeline run (e.g. root_cause narrative, safe to trust once it has itself
    passed this guard)."""
    parts = [finding_text or ""]
    for item in evidence_ledger or []:
        parts.append(item.claim or "")
    parts.extend(extra_trusted_text or [])
    return " ".join(parts)


# App-generated system/administrative vocabulary (never a finding-specific
# claim) that would otherwise false-positive against the all-uppercase
# identifier heuristic below -- e.g. "DEGRADED MODE" in a degraded-mode
# narrative isn't a fabricated entity, it's this system labeling its own
# analysis quality. Deliberately small and generic, not case-specific.
_SYSTEM_LABEL_WORDS = {"DEGRADED", "MODE", "LLM", "CAPA", "ERP", "AP", "QMS", "RCA"}

# Ordinary short English function words that legitimately appear in ALL
# CAPS when this system writes its own shouting-case status language (e.g.
# "NOT ESTABLISHED FROM AVAILABLE EVIDENCE") -- general linguistic
# infrastructure, not domain vocabulary, kept local to this heuristic
# rather than reusing text_grounding._STOPWORDS so a change here can't
# silently affect extraction/grounding-validator behavior elsewhere.
_COMMON_SHORT_WORDS = {
    "FROM", "WITH", "INTO", "ONTO", "OVER", "UNDER", "THAT", "THIS", "WHEN",
    "WHERE", "WHICH", "WHAT", "WHO", "WHY", "HOW", "NOT", "FOR", "AND", "THE",
    "ARE", "WAS", "WERE", "HAS", "HAVE", "HAD", "DID", "DOES", "CAN", "MAY",
    "MUST", "WILL", "SHALL", "THAN", "THEN", "ONLY", "STILL", "YET",
}


def _looks_like_identifier(token: str) -> bool:
    """Filters out ordinary sentence-initial hyphenated words ("Out-of-range",
    "Workstation-level") that _ENTITY_RE's hyphen pattern also matches. A real
    identifier (SOP-OPS-014, R-12, LIMS-QA) either contains a digit or is
    entirely uppercase once hyphens/slashes are removed — ordinary English
    compound words are neither.

    A token with a digit or a hyphen/slash separator is identifier-shaped at
    any length (QC-REF-02, SOP-OPERATIONS-014) UNLESS every hyphen/slash
    segment is itself system-label/common-word vocabulary (e.g. "LLM-based"
    is not a fabricated identifier). A token that is ALL CAPS with no
    digit/separator is only identifier-shaped when SHORT and not a common
    function word: genuine acronyms/IDs matched this way (SOP, QC, LIMS,
    BMR, WFI) are a handful of characters. A long all-caps word is far more
    likely to be this system's own shouting-case status language (e.g.
    "ESTABLISHED", "AVAILABLE", "EVIDENCE") than a fabricated identifier --
    flagging those strips the system's own degraded-mode/uncertainty
    labeling out of the report."""
    if token.upper() in _SYSTEM_LABEL_WORDS or token.upper() in _COMMON_SHORT_WORDS:
        return False

    segments = re.split(r"[-/]", token)
    if len(segments) > 1 and all(
        seg.upper() in _SYSTEM_LABEL_WORDS or seg.upper() in _COMMON_SHORT_WORDS or not seg.isupper()
        for seg in segments
    ):
        # Every segment is either trusted vocabulary or not itself
        # all-caps (an ordinary word like "based") -- not identifier-shaped
        # unless it also carries a digit, which the check below still catches.
        if not any(c.isdigit() for c in token):
            return False

    stripped = token.replace("-", "").replace("/", "")
    if any(c.isdigit() for c in token) or "-" in token or "/" in token:
        return True
    return stripped.isupper() and len(stripped) <= 6


def ungrounded_entities(text: str, source_text: str) -> list[str]:
    """Returns every named-entity-shaped token or number in `text` that does
    not trace back to `source_text`."""
    if not text:
        return []
    violations: list[str] = []
    for entity in _ENTITY_RE.findall(text):
        if not _looks_like_identifier(entity):
            continue
        if not entity_is_grounded(entity, source_text):
            violations.append(entity)
    for number in _NUMBER_RE.findall(text):
        if number not in source_text:
            violations.append(number)
    for word_number in _WORD_NUMBER_RE.findall(text):
        if not re.search(rf"\b{re.escape(word_number)}\b", source_text, re.IGNORECASE):
            violations.append(word_number)
    return violations


def check_text_field(text: str | None, source_text: str) -> tuple[str | None, list[str]]:
    """For a single free-text field: returns (text, violations). Does not
    mutate — caller decides what to do (strip, fallback, downgrade status)."""
    if not text:
        return text, []
    return text, ungrounded_entities(text, source_text)


def filter_list_field(items: list[str], source_text: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Drops any list item (contributing factor, hypothesis statement, impact
    area, etc.) that contains an ungrounded entity/number. Returns the
    filtered list and the (item, violations) pairs that were dropped."""
    kept: list[str] = []
    dropped: list[tuple[str, list[str]]] = []
    for item in items:
        violations = ungrounded_entities(item, source_text)
        if violations:
            dropped.append((item, violations))
        else:
            kept.append(item)
    return kept, dropped


def has_verified_support(evidence_ledger: list[EvidenceItem]) -> bool:
    return any(e.status == EvidenceStatus.VERIFIED for e in evidence_ledger)


def has_causal_language(text: str | None) -> bool:
    return bool(text) and bool(CAUSAL_RE.search(text))


# ---------------------------------------------------------------------------
# Contradictory evidence
# ---------------------------------------------------------------------------

_NEGATION_RE = re.compile(
    r"\b(no|not|without|never|none|didn't|did not|hasn't|has not|wasn't|was not|"
    r"weren't|were not|couldn't|could not)\b",
    re.IGNORECASE,
)


def _is_negated(text: str) -> bool:
    return bool(_NEGATION_RE.search(text))


_STEM_SUFFIXES = ("ations", "ation", "ing", "ions", "ion", "ed", "es", "s")


def _stem(word: str) -> str:
    """Crude suffix-stripping stemmer -- just enough to match "completed" /
    "completion" / "complete" as the same root for contradiction detection.
    Not linguistically rigorous; only used to widen the overlap check below."""
    for suffix in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _stemmed_words(text: str) -> set[str]:
    return {_stem(w) for w in significant_words(text)}


def detect_evidence_contradictions(evidence_ledger: list[EvidenceItem]) -> list[tuple[str, str]]:
    """Detects pairs of evidence items that talk about the same subject but
    assert opposite polarity -- e.g. "supervisor stated training was
    completed" vs "training system showed no completion record". Returns the
    conflicting claim pairs so the caller can force a neutral conclusion
    ("cannot currently be established") instead of silently picking a side."""
    pairs: list[tuple[str, str]] = []
    items = [e for e in evidence_ledger if e.claim]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            claim_a, claim_b = items[i].claim, items[j].claim
            words_a, words_b = _stemmed_words(claim_a), _stemmed_words(claim_b)
            if not words_a or not words_b:
                continue
            overlap = words_a & words_b
            union = words_a | words_b
            # Jaccard (overlap / union), not overlap / smaller-set: a min-ratio
            # check false-positived in production on two DIFFERENT, entirely
            # compatible facts that just happen to share a subject -- "three
            # operators performed the procedure" vs "the three operators had
            # no recorded training completion" hit 0.4 on min-ratio (sharing
            # only "three"/"operators") despite describing unrelated aspects.
            # Jaccard penalizes that shared-subject-but-different-topic case
            # much harder while still catching genuine same-topic conflicts.
            jaccard = len(overlap) / len(union) if union else 0.0
            if len(overlap) >= 2 and jaccard >= 0.28 and (_is_negated(claim_a) != _is_negated(claim_b)):
                pairs.append((claim_a, claim_b))
    return pairs


# ---------------------------------------------------------------------------
# Recurrence / effectiveness
# ---------------------------------------------------------------------------

_RECURRENCE_TRIGGER_RE = re.compile(
    r"\b(same issue|recurring|recurred|repeat finding|previous audit|previously identified|"
    r"previous corrective action|previously completed|prior corrective action|"
    r"recorded as completed)\b",
    re.IGNORECASE,
)
_EFFECTIVENESS_CLAIM_RE = re.compile(
    r"\b(effective(ness)?|prevented recurrence|successfully implemented|verified effective|"
    r"resolved the issue)\b",
    re.IGNORECASE,
)

SAFE_RECURRENCE_FALLBACK = (
    "The finding indicates a previous corrective action, but the available evidence "
    "establishes only that it was recorded — not whether it was implemented, "
    "verified, or effective in preventing recurrence."
)


def is_recurrence_finding(finding_text: str) -> bool:
    return bool(_RECURRENCE_TRIGGER_RE.search(finding_text or ""))


def claims_unsupported_effectiveness(text: str | None, finding_text: str) -> bool:
    """For a recurrence finding, 'the previous CAPA was recorded as completed'
    must never be read as 'the previous CAPA was effective' -- completion and
    effectiveness are different, separately-evidenced claims. True if `text`
    asserts effectiveness/success and the finding text itself never does."""
    if not text or not is_recurrence_finding(finding_text):
        return False
    return bool(_EFFECTIVENESS_CLAIM_RE.search(text)) and not _EFFECTIVENESS_CLAIM_RE.search(finding_text or "")
