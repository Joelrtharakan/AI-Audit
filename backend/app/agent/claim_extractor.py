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

# "<Speaker> stated/reported/claimed/... (that) <claim>"
_REPORT_VERB_RE = re.compile(
    r"^(?P<speaker>[A-Z][\w\s-]{0,50}?)\s+"
    r"(?:stated|reported|confirmed|indicated|noted|mentioned|claimed|said|"
    r"explained|advised|acknowledged|attributed\s+(?:this|it|the\s+issue|the\s+failure)?\s*to|cited|"
    r"states|reports|confirms|indicates|notes|mentions|claims|says|"
    r"explains|advises|acknowledges|"
    r"state|report|confirm|indicate|note|mention|claim|say|"
    r"explain|advise|acknowledge)\s+"
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
    r"assigned|delivered|distributed|available|accessible|found|present|documented|recorded|"
    r"maintained|updated|followed|applied|verified|confirmed|communicated|informed|"
    r"notified|acknowledged|opened|accessed|viewed|retrieved|approved|granted|authorized)|"
    r"had\s+not\s+(?:received|completed|been)|"
    r"did\s+not\s+(?:receive|complete|attend|perform)|"
    r"never\s+(?:received|completed|performed|attended|been|given|granted|approved)|"
    r"was\s+never\s+(?:available|accessible|found|present|opened|given|granted)|"
    r"missing|absent|unavailable|incomplete|unaware|not\s+aware|"
    r"\bno\s+[\w\s-]{1,60}?\s*(?:was|were|has\s+been|have\s+been|could\s+be)?\s*(?:completed|performed|conducted|"
    r"done|provided|given|received|attended|delivered|distributed|assigned|available|accessible|found|present|"
    r"documented|recorded|maintained|updated|followed|applied|verified|confirmed|informed|notified|located))\b",
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
    r"delivered|distributed|assigned|documented|recorded|opened|accessed|viewed|retrieved|approved|"
    r"granted|authorized|"
    r"maintained|updated|followed|applied|verified|confirmed|informed|notified|acknowledged|"
    r"dispatched|dispatch|delivered|delivery|distributed|distribution|sent|transmitted|transmission)\b|"
    r"\b(?:was|were|has\s+been|have\s+been|had\s+been)\s+(?:done|available|accessible|found|present)\b|"
    # "<Record/system/log> shows/show/records/indicates/confirms <completion/
    # approval/delivery/receipt/dispatch/distribution>"
    r"\b(?:shows?|records?|indicates?|confirms?|establishes?)\s+(?:[\w\s]{0,20}?\s+)?(?:completion|approval|delivery|receipt|dispatch|distribution|transmission)\b",
    re.IGNORECASE,
)


_AUDITOR_SPEAKER_RE = re.compile(
    r"\b(?:auditor|audit\s+observation|audit\s+finding|audit|inspection|investigation)\b",
    re.IGNORECASE,
)
_SYSTEM_RECORD_SPEAKER_RE = re.compile(
    r"\b(?:system|audit|server|dispatch|lms|training|distribution|calibration|maintenance|temperature|inspection|access|database|equipment|bank|accounting|scada|error)\s+(?:system\s+|training\s+|error\s+)?(?:logs?|records?|trail|history|memo|certificate|subledger|statement)\b|"
    r"\b(?:dispatch\s+system|scada\s+system|server\s+error\s+logs?|bank\s+credit\s+memo)\b",
    re.IGNORECASE,
)
_EVIDENCE_AVAILABILITY_RE = re.compile(
    r"\b(?:not\s+available\s+to\s+the\s+ai|not\s+available\s+for\s+inspection|unavailable\s+to\s+the\s+ai|report\s+was\s+not\s+available)\b",
    re.IGNORECASE,
)


def _classify_attribution(speaker: str | None) -> tuple[ClaimAttribution, EvidenceStatus]:
    """Classify the attribution and initial evidence status based on the speaker role."""
    if not speaker:
        return ClaimAttribution.AUDITOR_OBSERVED, EvidenceStatus.VERIFIED
    if _AUDITOR_SPEAKER_RE.search(speaker):
        return ClaimAttribution.AUDITOR_OBSERVED, EvidenceStatus.VERIFIED
    if _SYSTEM_RECORD_SPEAKER_RE.search(speaker):
        return ClaimAttribution.SYSTEM_EVIDENCE, EvidenceStatus.VERIFIED
    if _SUPERVISOR_RE.search(speaker):
        return ClaimAttribution.SUPERVISOR_REPORTED, EvidenceStatus.REPORTED
    # Any named person/role reporting is PERSON_REPORTED
    return ClaimAttribution.PERSON_REPORTED, EvidenceStatus.REPORTED


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
    group_idx = -1

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(finding_text.strip()) if s.strip()]

    for sentence in sentences:
        # A sentence that opens with a discourse connector ("However, ...")
        # continues the PREVIOUS sentence's proposition rather than starting
        # a new one -- keep it in the same sentence_group so anaphoric
        # references ("it was never received") can still be paired with the
        # earlier sentence's claim even without repeating its noun.
        if _LEADING_DISCOURSE_CONNECTOR_RE.match(sentence) and group_idx >= 0:
            pass
        else:
            group_idx += 1

        # First check if the sentence is a compound with conflicting clauses
        clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(sentence) if c.strip()]

        if len(clauses) >= 2:
            # Process each clause independently -- this is what prevents
            # "operator said X but supervisor said Y" from becoming one
            # VERIFIED compound claim.
            for clause in clauses:
                claim_counter += 1
                claim = _extract_single_claim(clause, claim_counter)
                claim.sentence_group = group_idx
                claims.append(claim)
        else:
            # Check for attribution patterns
            claim_counter += 1
            claim = _extract_single_claim(sentence, claim_counter)
            claim.sentence_group = group_idx
            claims.append(claim)

    # NOTE: a REPORTED claim is NOT separately re-extracted as an "UNKNOWN"
    # derived proposition. "The operator stated X" already carries its own
    # provenance (REPORTED) and truth status (UNVERIFIED) on the single
    # canonical claim above — duplicating it as a second claim with status
    # UNKNOWN would represent the same proposition twice under two different
    # (and contradictory) evidence statuses.
    return claims


_LEADING_DISCOURSE_CONNECTOR_RE = re.compile(
    r"^(?:however|but|nevertheless|yet|still|in\s+contrast|conversely|"
    r"on\s+the\s+other\s+hand)\s*,\s*", re.IGNORECASE,
)


def _strip_leading_discourse_connector(text: str) -> str:
    """Strip a leading discourse connector ("However, ", "But, ",
    "Nevertheless, ") from a sentence/clause before attribution-pattern
    matching. Without this, "However, three operators stated that..."
    never matches `_REPORT_VERB_RE` (anchored with `^`, and the comma
    right after "However" breaks the speaker capture group) -- the
    attributed statement silently falls through to the "no attribution
    pattern" branch and gets misclassified as a direct AUDITOR_OBSERVED
    VERIFIED claim instead of a REPORTED one, which then makes it
    invisible to conflict detection (VERIFIED-vs-VERIFIED pairs are never
    checked, only REPORTED-vs-REPORTED and VERIFIED-vs-REPORTED)."""
    return _LEADING_DISCOURSE_CONNECTOR_RE.sub("", text.strip())


def _extract_single_claim(text: str, counter: int) -> EvidenceClaim:
    """Extract a single claim from a sentence or clause."""
    matchable_text = _strip_leading_discourse_connector(text.strip())

    # Check if this clause is explicitly describing evidence availability
    if _EVIDENCE_AVAILABILITY_RE.search(matchable_text):
        return EvidenceClaim(
            claim_id=f"C{counter}",
            text=text.strip(),
            subject=_extract_subject(matchable_text),
            predicate=matchable_text,
            source="finding_text",
            status=EvidenceStatus.VERIFIED,
            confidence="HIGH",
            evidence_reference="Evidence availability constraint",
            attribution=ClaimAttribution.EVIDENCE_AVAILABILITY,
            polarity=_classify_polarity(matchable_text),
            source_type="MISSING_REFERENCED_DOCUMENT",
            statement_status="VERIFIED",
            underlying_event_status="NOT_APPLICABLE",
            availability="UNAVAILABLE",
            document_available=False,
        )

    # Check for attribution patterns
    for pattern in _ATTRIBUTION_PATTERNS:
        m = pattern.match(matchable_text)
        if m:
            speaker = m.group("speaker").strip()
            claim_text = m.group("claim").strip().rstrip(".")
            attribution, initial_status = _classify_attribution(speaker)
            if attribution == ClaimAttribution.SYSTEM_EVIDENCE:
                src_type = "SYSTEM_RECORD"
                stmt_status = "VERIFIED"
                evt_status = "VERIFIED"
            elif attribution == ClaimAttribution.AUDITOR_OBSERVED:
                src_type = "AUDIT_OBSERVATION"
                stmt_status = "VERIFIED"
                evt_status = "UNKNOWN"
            else:
                src_type = "REPORTED_STATEMENT"
                stmt_status = "REPORTED"
                evt_status = "UNKNOWN"

            return EvidenceClaim(
                claim_id=f"C{counter}",
                text=text.strip(),
                subject=_extract_subject(claim_text),
                predicate=claim_text,
                source="finding_text",
                status=initial_status,
                confidence="HIGH" if initial_status == EvidenceStatus.VERIFIED else "MEDIUM",
                evidence_reference="Auditor finding text" if initial_status == EvidenceStatus.VERIFIED else "Attributed statement in finding text",
                attribution=attribution,
                polarity=_classify_polarity(claim_text),
                speaker=speaker,
                source_type=src_type,
                statement_status=stmt_status,
                underlying_event_status=evt_status,
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
        source_type="AUDIT_OBSERVATION",
        statement_status="VERIFIED",
        underlying_event_status="UNKNOWN",
    )

# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

# Verb-concept vocabularies for structural conflict-proposition
# classification (Section 13, Conflict-Center hardening). Deliberately
# generic English verbs, not tied to any single domain -- "delivered" fires
# for email, SOP notifications, shipments, etc. equally.
_DELIVERY_VERBS = {"deliver", "delivered", "send", "sent", "transmit", "transmitted", "dispatch", "dispatched", "distribute", "distributed"}
_RECEIPT_ACCESS_VERBS = {"receive", "received", "get", "got", "obtain", "obtained", "access", "accessed", "accessible", "open", "opened", "view", "viewed", "retrieve", "retrieved"}
_COMPLETION_VERBS = {"complete", "completed", "perform", "performed", "conduct", "conducted", "finish", "finished", "carry", "carried", "execute", "executed"}
_MISSING_MARKERS_RE = re.compile(
    r"\bnever\b|\bnot\s+(?:performed|completed|conducted|done|given|carried|received|attended)\b",
    re.IGNORECASE,
)
# A verb like "received"/"delivered"/"accessed" is only about TRANSMISSION
# of a communicable artifact (a notification/email/document/message) when
# its object is one of these -- "received TRAINING" is a completion-of-
# activity idiom, not a transmission-of-content one, even though it shares
# the verb "received". Distinguishing by object noun (not verb alone) is
# what keeps this from misclassifying the training-conflict case.
_COMMUNICATION_ARTIFACT_WORDS = {
    "notification", "notifications", "email", "emails", "message", "messages",
    "memo", "memos", "letter", "letters", "document", "documents", "notice",
    "notices", "communication", "communications", "alert", "alerts", "bulletin",
    "bulletins", "attachment", "attachments", "copy", "copies",
}
_ACTIVITY_OBJECT_WORDS = {
    "training", "retraining", "instruction", "briefing", "orientation",
    "induction", "onboarding", "assessment", "certification", "qualification",
    "inspection", "calibration", "maintenance", "review", "audit",
}


def classify_conflict_proposition_type(c1: EvidenceClaim, c2: EvidenceClaim) -> str:
    """Structural (verb+object-shape, not per-domain-keyword) classification
    of WHAT the two conflicting claims disagree about -- e.g. whether
    something was delivered vs. actually received (DELIVERY_VS_RECEIPT,
    covers email/notification/document-access shapes alike) vs. whether an
    activity was completed at all (COMPLETION_VS_MISSING_RECORD) vs. a
    record asserting one thing against a person's contradicting statement
    with no transmission/completion verb involved (RECORD_VS_STATEMENT,
    e.g. an approval record vs. "approval was never given").

    DELIVERY_VS_RECEIPT requires BOTH a transmission verb (delivered/sent/
    received/accessed/etc.) AND a communicable-artifact object
    (notification/email/document/message/etc.) with NO activity-type object
    present -- "had not received TRAINING" shares the verb "received" with
    "never received the NOTIFICATION" but is a completion dispute, not a
    transmission one; checking the object noun (not just the verb) is what
    tells these apart.

    This exists so downstream investigation-question generation can tell a
    "did the recipient ever get it" conflict apart from a "was the task
    ever done" conflict -- treating every conflict as a completion dispute
    produces a fabricated NOT_COMPLETED hypothesis for conflicts that were
    never about completion at all (e.g. system-delivered-but-not-received)."""
    combined = f"{c1.text} {c2.text}".lower()
    stems = {_stem(w) for w in re.findall(r"[a-z]+", combined)}

    def _has_any(words: set[str]) -> bool:
        return bool(stems & {_stem(w) for w in words})

    has_transmission_verb = _has_any(_DELIVERY_VERBS) or _has_any(_RECEIPT_ACCESS_VERBS)
    has_comm_artifact = _has_any(_COMMUNICATION_ARTIFACT_WORDS)
    has_activity_object = _has_any(_ACTIVITY_OBJECT_WORDS)
    if has_transmission_verb and has_comm_artifact and not has_activity_object:
        return "DELIVERY_VS_RECEIPT"
    if (_has_any(_COMPLETION_VERBS) or has_transmission_verb) and _MISSING_MARKERS_RE.search(combined):
        return "COMPLETION_VS_MISSING_RECORD"
    statuses = {c1.status, c2.status}
    if EvidenceStatus.VERIFIED in statuses and EvidenceStatus.REPORTED in statuses:
        return "RECORD_VS_STATEMENT"
    return "CONFLICTING_REPORTS"


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
                    proposition_type=classify_conflict_proposition_type(c1, c2),
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
                    proposition_type=classify_conflict_proposition_type(vc, rc),
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
    # Two clauses from the SAME original sentence (joined with but/however/;
    # or a discourse-connector-led follow-on sentence) are already
    # grammatically linked by the finding's own author -- that structural
    # adjacency alone is sufficient, even when the second clause uses a
    # bare pronoun instead of repeating the first clause's noun ("the email
    # was delivered; it was never received" never re-states "email").
    if c1.sentence_group is not None and c1.sentence_group == c2.sentence_group:
        return True
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
    if meaningful1 & meaningful2:
        return True

    # Transmission/communication conflict: one claim mentions a communication artifact or delivery verb,
    # and the other mentions a communication artifact or receipt/access verb.
    comm_stems = {_stem(w) for w in _COMMUNICATION_ARTIFACT_WORDS}
    deliv_stems = {_stem(w) for w in _DELIVERY_VERBS}
    rec_stems = {_stem(w) for w in _RECEIPT_ACCESS_VERBS}
    if ((meaningful1 & comm_stems or words1 & deliv_stems) and
        (meaningful2 & comm_stems or words2 & rec_stems)):
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
