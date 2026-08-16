"""Claim-level evidence extraction and conflict detection.

Deterministic, LLM-free module that decomposes finding text into individual
claims with proper attribution and detects conflicts between claims about the
same proposition.

Every extracted claim carries its own provenance so downstream reasoning can
distinguish 'the operator stated X' (PERSON_REPORTED) from 'the auditor
observed X' (AUDITOR_OBSERVED) and never collapse two conflicting reported
statements into a single VERIFIED fact.

Key design rules:
  - A reported statement NEVER becomes VERIFIED merely because it appears in
    the finding text. Only direct auditor observations are VERIFIED.
  - REPORTED != UNKNOWN. A claim someone asserted is REPORTED (provenance)
    and UNVERIFIED (truth status) — it is NOT re-extracted a second time as a
    separate "UNKNOWN" proposition. UNKNOWN is reserved for propositions the
    available evidence does not address at all, not for every reported
    statement's bare content.
  - The system NEVER automatically resolves a conflict by choosing one side.
  - Conflict detection is structural (polarity + subject overlap), not
    keyword-based.
"""

from __future__ import annotations

import re
from typing import Any

from app.models.agent import ClaimAttribution, EvidenceClaim, EvidenceConflict, EvidenceStatus
from app.services.text_grounding import significant_words

# ---------------------------------------------------------------------------
# Attribution patterns (reuses the same structural approach as
# attribution_extraction.py but produces EvidenceClaim objects)
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Connective splits: "but", "however", semicolons -- used to decompose
# compound sentences into individual claims.
_CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:,\s*but\s+|,\s*however\s+|;\s*|\s+but\s+|\s+however\s+)\s*",
    re.IGNORECASE,
)

# "<Speaker> stated/reported/claimed/... that <claim>"
_REPORT_VERB_RE = re.compile(
    r"^(?P<speaker>[A-Z][\w\s-]{0,50}?)\s+"
    r"(?:stated|reported|confirmed|indicated|noted|mentioned|claimed|said|"
    r"explained|advised|acknowledged)\s+"
    r"(?:that\s+)?(?P<claim>.+)$",
    re.IGNORECASE,
)

# "<Speaker> was/were unaware/not aware/not informed (that|of) <claim>"
_AWARENESS_GAP_RE = re.compile(
    r"^(?P<speaker>[A-Z][\w\s-]{0,50}?)\s+(?:was|were)\s+"
    r"(?P<claim>(?:unaware|not\s+aware|not\s+informed|not\s+notified|not\s+told).*)$",
    re.IGNORECASE,
)

# "<Speaker> did not know / didn't know (that) <claim>"
_DID_NOT_KNOW_RE = re.compile(
    r"^(?P<speaker>[A-Z][\w\s-]{0,50}?)\s+(?:did\s+not|didn't)\s+know\s+"
    r"(?P<claim>.*)$",
    re.IGNORECASE,
)

_ATTRIBUTION_PATTERNS = (_REPORT_VERB_RE, _AWARENESS_GAP_RE, _DID_NOT_KNOW_RE)

# Supervisor/manager detection
_SUPERVISOR_RE = re.compile(
    r"\b(supervisor|manager|lead|team\s+lead|section\s+head|department\s+head|"
    r"quality\s+manager|shift\s+lead)\b",
    re.IGNORECASE,
)

# Polarity detection: positive ("was completed") vs negative ("was not completed")
_NEGATIVE_POLARITY_RE = re.compile(
    r"\b(not\s+(?:received|completed|performed|conducted|done|provided|given|attended|"
    r"assigned|delivered|available|accessible|found|present|documented|recorded|"
    r"maintained|updated|followed|applied|verified|confirmed|communicated|informed|"
    r"notified|acknowledged)|"
    r"had\s+not\s+(?:received|completed|been)|"
    r"did\s+not\s+(?:receive|complete|attend|perform)|"
    r"never\s+(?:received|completed|performed|attended|been)|"
    r"missing|absent|unavailable|incomplete|unaware|not\s+aware|"
    # A leading "No <noun phrase> was/were <state-word>" negates the whole
    # clause even though it contains a positive-looking state verb (e.g. "No
    # training attendance record was available") -- without this, the
    # sentence-initial "No" is missed and the clause is misread as positive
    # because "was available" alone matches the positive pattern below.
    r"^no\s+[\w\s-]{1,60}?\s+(?:was|were|has\s+been|have\s+been)\s+(?:completed|performed|conducted|"
    r"done|provided|given|received|attended|delivered|assigned|available|accessible|found|present|"
    r"documented|recorded|maintained|updated|followed|applied|verified|confirmed|informed|notified))\b",
    re.IGNORECASE,
)

_POSITIVE_POLARITY_RE = re.compile(
    # The passive-voice auxiliary ("was/were/has been") is OPTIONAL: a claim
    # predicate stated in active voice ("the operator completed the
    # training") is just as much a positive-polarity claim as its passive
    # form ("the training was completed") -- requiring the auxiliary missed
    # every active-voice claim entirely.
    r"\b(?:(?:was|were|has\s+been|have\s+been|had\s+been)\s+)?"
    r"(?:completed|performed|conducted|provided|given|received|attended|"
    r"delivered|assigned|documented|recorded|"
    r"maintained|updated|followed|applied|verified|confirmed|informed|notified|acknowledged)\b|"
    r"\b(?:was|were|has\s+been|have\s+been|had\s+been)\s+(?:done|available|accessible|found|present)\b",
    re.IGNORECASE,
)


def _classify_attribution(speaker: str | None) -> ClaimAttribution:
    """Classify the attribution based on the speaker role."""
    if not speaker:
        return ClaimAttribution.UNKNOWN
    if _SUPERVISOR_RE.search(speaker):
        return ClaimAttribution.SUPERVISOR_REPORTED
    # Any named person/role reporting is PERSON_REPORTED
    return ClaimAttribution.PERSON_REPORTED


def _classify_polarity(text: str) -> str | None:
    """Classify the polarity of a claim text."""
    if _NEGATIVE_POLARITY_RE.search(text):
        return "negative"
    if _POSITIVE_POLARITY_RE.search(text):
        return "positive"
    return None


def _extract_subject(text: str) -> str | None:
    """Extract the primary subject (what the claim is about) from claim text.
    Structural only -- no NLP dependency."""
    # Look for "the <noun phrase> was/were/has/had..."
    m = re.search(
        r"\b(?:the|a|an)\s+([a-z][\w\s-]{1,40}?)\s+(?:was|were|has|had|is|are)\b",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return None


def extract_claims(
    finding_text: str,
    evidence_ledger: list[Any] | None = None,
) -> list[EvidenceClaim]:
    """Decompose finding text into individual claims with proper attribution.

    Rules:
      - Direct auditor observations → AUDITOR_OBSERVED, status VERIFIED
      - "<Person> stated/claimed/reported X" → PERSON_REPORTED, status REPORTED
      - "<Supervisor> claimed X" → SUPERVISOR_REPORTED, status REPORTED
      - Derived propositions (the underlying truth) → UNKNOWN status
      - Compound sentences are split on "but"/"however"/semicolons
    """
    claims: list[EvidenceClaim] = []
    claim_counter = 0

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(finding_text.strip()) if s.strip()]

    for sentence in sentences:
        # First check if the sentence is a compound with conflicting clauses
        clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(sentence) if c.strip()]

        if len(clauses) >= 2:
            # Process each clause independently -- this is what prevents
            # "operator said X but supervisor said Y" from becoming one
            # VERIFIED compound claim.
            for clause in clauses:
                claim_counter += 1
                claim = _extract_single_claim(clause, claim_counter)
                claims.append(claim)
        else:
            # Check for attribution patterns
            claim_counter += 1
            claim = _extract_single_claim(sentence, claim_counter)
            claims.append(claim)

    # NOTE: a REPORTED claim is NOT separately re-extracted as an "UNKNOWN"
    # derived proposition. "The operator stated X" already carries its own
    # provenance (REPORTED) and truth status (UNVERIFIED) on the single
    # canonical claim above — duplicating it as a second claim with status
    # UNKNOWN would represent the same proposition twice under two different
    # (and contradictory) evidence statuses.
    return claims


def _extract_single_claim(text: str, counter: int) -> EvidenceClaim:
    """Extract a single claim from a sentence or clause."""
    # Check for attribution patterns
    for pattern in _ATTRIBUTION_PATTERNS:
        m = pattern.match(text.strip())
        if m:
            speaker = m.group("speaker").strip()
            claim_text = m.group("claim").strip().rstrip(".")
            attribution = _classify_attribution(speaker)
            return EvidenceClaim(
                claim_id=f"C{counter}",
                text=text.strip(),
                subject=_extract_subject(claim_text),
                predicate=claim_text,
                source="finding_text",
                status=EvidenceStatus.REPORTED,
                confidence="MEDIUM",
                evidence_reference="Attributed statement in finding text",
                attribution=attribution,
                polarity=_classify_polarity(claim_text),
                speaker=speaker,
            )

    # No attribution pattern -- this is a direct auditor observation
    return EvidenceClaim(
        claim_id=f"C{counter}",
        text=text.strip(),
        subject=_extract_subject(text),
        predicate=text.strip(),
        source="finding_text",
        status=EvidenceStatus.VERIFIED,
        confidence="HIGH",
        evidence_reference="Auditor finding text",
        attribution=ClaimAttribution.AUDITOR_OBSERVED,
        polarity=_classify_polarity(text),
    )

# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def detect_evidence_conflicts(claims: list[EvidenceClaim]) -> list[EvidenceConflict]:
    """Detect conflicts between claims about the same proposition.

    Two claims conflict when they concern the same subject but have
    incompatible polarities (one positive, one negative) and both are
    REPORTED (neither is VERIFIED). The system NEVER automatically
    resolves a conflict.

    Returns a list of EvidenceConflict objects, each identifying the
    conflicting claims, the proposition they disagree about, and the
    fact that resolution is required.
    """
    conflicts: list[EvidenceConflict] = []
    conflict_counter = 0

    # Claims that reference a DIFFERENT temporal event (a similar finding
    # from a previous audit, a previous corrective action) describe history,
    # not a competing explanation for the CURRENT deviation -- they must
    # never be paired into a conflict with a claim about the present finding
    # just because of incidental word overlap (e.g. both mentioning
    # "monitoring"). Recurrence reasoning about these claims happens
    # separately (app.agent.recurrence_guard), not through conflict
    # detection.
    from app.agent.recurrence_guard import _PREVIOUS_AUDIT_TIME_RE, _PREVIOUS_CAPA_RE, _SIMILAR_FINDING_RE

    def _is_recurrence_reference(text: str) -> bool:
        return bool(
            _SIMILAR_FINDING_RE.search(text) or _PREVIOUS_AUDIT_TIME_RE.search(text) or _PREVIOUS_CAPA_RE.search(text)
        )

    non_recurrence_claims = [c for c in claims if not _is_recurrence_reference(c.text)]

    # Group claims by subject (approximate -- uses significant word overlap)
    reported_claims = [c for c in non_recurrence_claims if c.status == EvidenceStatus.REPORTED]

    for i, c1 in enumerate(reported_claims):
        for c2 in reported_claims[i + 1:]:
            if _claims_conflict(c1, c2):
                conflict_counter += 1
                # Determine the proposition they disagree about
                proposition = _derive_conflict_proposition(c1, c2)
                conflicts.append(EvidenceConflict(
                    conflict_id=f"CONF{conflict_counter}",
                    conflict_type="CONFLICTING_REPORTS",
                    status="UNRESOLVED",
                    claims=[c1.claim_id, c2.claim_id],
                    proposition=proposition,
                    resolution_required=True,
                    resolution_note=None,
                ))

    # Also check reported vs verified conflicts
    verified_claims = [c for c in non_recurrence_claims if c.status == EvidenceStatus.VERIFIED]
    for vc in verified_claims:
        for rc in reported_claims:
            if _claims_conflict(vc, rc):
                conflict_counter += 1
                proposition = _derive_conflict_proposition(vc, rc)
                conflicts.append(EvidenceConflict(
                    conflict_id=f"CONF{conflict_counter}",
                    conflict_type="CONTRADICTED_BY_EVIDENCE",
                    status="UNRESOLVED",
                    claims=[vc.claim_id, rc.claim_id],
                    proposition=proposition,
                    resolution_required=True,
                ))

    return _merge_same_proposition_conflicts(conflicts)


def _merge_same_proposition_conflicts(conflicts: list[EvidenceConflict]) -> list[EvidenceConflict]:
    """Merge conflicts that describe the SAME underlying proposition under
    different wording into one conflict with all claims attached.

    A REPORTED-vs-REPORTED pass and a VERIFIED-vs-REPORTED pass run
    independently above and can each form their own conflict object around
    a shared claim (e.g. "supervisor says training completed") -- one
    phrased as "whether X received training", the other as "whether the
    required activity was completed" -- which is the same disagreement, not
    two. Conflict identity must be the semantic proposition, not the
    surface wording that happened to generate it (Section 6).
    """
    merged: list[EvidenceConflict] = []
    for conf in conflicts:
        prop_words = significant_words(conf.proposition)
        match = None
        for existing in merged:
            # Sharing a claim ID is the stronger signal: the two detection
            # passes above (REPORTED-vs-REPORTED, then VERIFIED-vs-REPORTED)
            # only ever pair claims _claims_conflict() already judged to be
            # about the same subject -- if a claim recurs across two
            # resulting conflict objects, both are already independently
            # anchored to the same underlying disagreement even when their
            # generated proposition wording uses different vocabulary (e.g.
            # "training" vs. "the required activity").
            if set(conf.claims) & set(existing.claims):
                match = existing
                break
            existing_words = significant_words(existing.proposition)
            if prop_words and existing_words:
                overlap = len(prop_words & existing_words) / min(len(prop_words), len(existing_words))
                if overlap >= 0.4:
                    match = existing
                    break
        if match:
            for cid in conf.claims:
                if cid not in match.claims:
                    match.claims.append(cid)
        else:
            merged.append(conf)
    for idx, conf in enumerate(merged, start=1):
        conf.conflict_id = f"CONF{idx}"
    return merged


def _stem(w: str) -> str:
    """Lightweight 2-letter stemmer for common suffixes (s, ed, ing, er, ers)."""
    w = w.lower()
    for suffix in ("ing", "ers", "ed", "er", "es", "s"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            return w[:-len(suffix)]
    return w


def _claims_conflict(c1: EvidenceClaim, c2: EvidenceClaim) -> bool:
    """True if two claims concern the same subject/activity but have
    incompatible polarities (one says X happened/was completed, the other says
    X did not happen/was not completed)."""
    # Both must have a polarity classification
    if not c1.polarity or not c2.polarity:
        return False
    # Polarities must be opposite
    if c1.polarity == c2.polarity:
        return False
    # Subjects/predicates must overlap (stem-level comparison)
    words1 = {_stem(w) for w in significant_words(c1.text)}
    words2 = {_stem(w) for w in significant_words(c2.text)}
    if not words1 or not words2:
        return False
    # Reporting verbs/roles (operator, supervisor, ...) AND generic
    # institutional/location nouns (laboratory, department, ...) are both
    # excluded from the PRIMARY overlap check -- a shared role word or a
    # shared location word ALONE is never sufficient to call two claims
    # about the same disputed proposition (e.g. an observation set in a
    # "laboratory" and an unrelated statement from a "laboratory
    # supervisor" share nothing but their setting).
    ignore_role_stems = {_stem(w) for w in ("stated", "claimed", "reported", "said", "operator", "supervisor", "auditor", "trainer", "technician")}
    ignore_institutional_stems = {_stem(w) for w in (
        "laboratory", "department", "area", "site", "facility", "room", "building", "company", "organization",
        "team", "unit", "location", "audit", "finding", "manager", "staff", "personnel",
    )}
    meaningful1 = words1 - ignore_role_stems - ignore_institutional_stems
    meaningful2 = words2 - ignore_role_stems - ignore_institutional_stems
    if not meaningful1 or not meaningful2:
        return False
    if meaningful1 & meaningful2:
        return True

    # Weak fallback: even without substantive vocabulary overlap, two
    # opposite-polarity claims that both mention a shared ROLE word (e.g.
    # both mentioning "operator") are treated as weakly related -- role
    # words are reinstated here (only institutional/location words stay
    # excluded), since that's the one case a bare role-word match is a
    # meaningful signal rather than shared incidental context.
    weak1 = words1 - ignore_institutional_stems
    weak2 = words2 - ignore_institutional_stems
    if not weak1 or not weak2:
        return False
    min_len = min(len(meaningful1), len(meaningful2))
    return len(weak1 & weak2) / max(min_len, 1) >= 0.2


# Captures a negated verb clause and its direct object/tail (e.g. "had not
# received retraining after the monitoring procedure was revised") so a
# conflict proposition can be built from a claim's OWN specific wording
# instead of from incidental shared vocabulary between two unrelated
# sentences (which is what produced nonsense like "whether laboratory was
# completed"). Deliberately generic verb list, no domain words.
_NEGATION_CLAUSE_RE = re.compile(
    r"\b(?:had\s+not|did\s+not|was\s+not|were\s+not|has\s+not|have\s+not|never)\s+"
    r"(?:received|completed|performed|conducted|attended|been\s+(?:given|provided|informed|notified))\s+"
    r"[\w\s-]{1,80}?(?=[.,;]|$|\s+(?:because|since|but|however)\b)",
    re.IGNORECASE,
)
_NEGATION_LEAD_STRIP_RE = re.compile(
    r"^(?:had\s+not|did\s+not|was\s+not|were\s+not|has\s+not|have\s+not|never)\s+", re.IGNORECASE
)


def _derive_conflict_proposition(c1: EvidenceClaim, c2: EvidenceClaim) -> str:
    """Derive a COMPLETE semantic proposition the two claims disagree about,
    built from the claims' own wording -- never from raw shared-vocabulary
    intersection, which can produce a nonsense proposition out of two
    sentences that merely happen to share an incidental word (e.g. both
    mentioning "laboratory").

    Prefers the negatively-phrased claim (by construction, conflict
    detection only pairs claims of opposite polarity, so exactly one side
    is negative): extracts the specific negated clause ("had not received
    retraining after the monitoring procedure was revised"), flips it to a
    neutral affirmative ("received retraining after the monitoring
    procedure was revised"), and states it as a yes/no question attributed
    to the claim's own speaker when known ("Whether the technician received
    retraining after the monitoring procedure was revised.").
    """
    negative_claim = c1 if c1.polarity == "negative" else (c2 if c2.polarity == "negative" else None)
    if negative_claim is not None:
        source_text = negative_claim.predicate or negative_claim.text
        m = _NEGATION_CLAUSE_RE.search(source_text)
        if m:
            core = _NEGATION_LEAD_STRIP_RE.sub("", m.group(0).strip()).strip()
            if core:
                actor_phrase = None
                if negative_claim.speaker:
                    from app.services.semantic_subject import strip_leading_article
                    stripped = strip_leading_article(negative_claim.speaker)
                    if stripped:
                        actor_phrase = stripped.lower() if stripped.lower().startswith("the ") else f"the {stripped.lower()}"
                subject_phrase = actor_phrase or "the responsible person"
                return f"Whether {subject_phrase} {core}"

    # Fallback: shared significant vocabulary, but only when it actually
    # forms a coherent multi-word topic (a single incidental shared word
    # like "laboratory" is not a proposition) -- otherwise fall back to the
    # claim subject rather than emit a malformed one-word proposition.
    words1 = significant_words(c1.text)
    words2 = significant_words(c2.text)
    ignore = {"stated", "claimed", "reported", "said", "operator", "supervisor", "auditor", "trainer", "technician", "completed", "received", "performed"}
    common = (words1 & words2) - ignore
    if len(common) >= 2:
        topic = " ".join(sorted(common))
        return f"Whether {topic} was completed/performed as required"
    subject = c1.subject or c2.subject or "the required activity"
    return f"Whether {subject} was completed/performed as required"
