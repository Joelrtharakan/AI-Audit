"""Extracts structured financial facts from raw text, evidence items, and claims.

Enforces:
  - Epistemic preservation: REPORTED / UNVERIFIED evidence remains REPORTED / UNVERIFIED.
  - Separate extraction of verified event counts vs reported event counts vs potential additional events.
  - Per-event amount isolation.
  - Identification of conflict across multiple stated amounts.
  - Multi-currency isolation.
  - Detection of explicit zero recovery vs unknown recovery.
"""

from __future__ import annotations

import re
from app.financial.models import (
    FinancialAmountType,
    FinancialObservation,
    RecoveryStatus,
)
from app.models.agent import EvidenceClaim, EvidenceItem, EvidenceStatus

# ISO 4217 alpha-3 currency codes -- authoritative reference DATA (not
# per-currency application logic). Any 3-letter code found here flows
# through the exact same generic extraction/calculation path as any
# other; adding support for a currency never requires new code, only
# (if ever needed) a data addition to this set. Deliberately the
# complete standard list, not merely the handful used in examples/tests.
_ISO_4217_CODES = frozenset({
    "AED", "AFN", "ALL", "AMD", "ANG", "AOA", "ARS", "AUD", "AWG", "AZN",
    "BAM", "BBD", "BDT", "BGN", "BHD", "BIF", "BMD", "BND", "BOB", "BRL",
    "BSD", "BTN", "BWP", "BYN", "BZD", "CAD", "CDF", "CHF", "CLP", "CNY",
    "COP", "CRC", "CUP", "CVE", "CZK", "DJF", "DKK", "DOP", "DZD", "EGP",
    "ERN", "ETB", "EUR", "FJD", "FKP", "GBP", "GEL", "GHS", "GIP", "GMD",
    "GNF", "GTQ", "GYD", "HKD", "HNL", "HTG", "HUF", "IDR", "ILS", "INR",
    "IQD", "IRR", "ISK", "JMD", "JOD", "JPY", "KES", "KGS", "KHR", "KMF",
    "KPW", "KRW", "KWD", "KYD", "KZT", "LAK", "LBP", "LKR", "LRD", "LSL",
    "LYD", "MAD", "MDL", "MGA", "MKD", "MMK", "MNT", "MOP", "MRU", "MUR",
    "MVR", "MWK", "MXN", "MYR", "MZN", "NAD", "NGN", "NIO", "NOK", "NPR",
    "NZD", "OMR", "PAB", "PEN", "PGK", "PHP", "PKR", "PLN", "PYG", "QAR",
    "RON", "RSD", "RUB", "RWF", "SAR", "SBD", "SCR", "SDG", "SEK", "SGD",
    "SHP", "SLE", "SOS", "SRD", "SSP", "STN", "SYP", "SZL", "THB", "TJS",
    "TMT", "TND", "TOP", "TRY", "TTD", "TWD", "TZS", "UAH", "UGX", "USD",
    "UYU", "UZS", "VES", "VND", "VUV", "WST", "XAF", "XCD", "XOF", "XPF",
    "YER", "ZAR", "ZMW", "ZWL",
})


def _valid_iso_code(token: str | None) -> str | None:
    """Return the uppercased ISO 4217 code if `token` is a recognized
    3-letter code, else None. The single point of truth for "is this a
    real currency code" -- never a per-currency branch, just a set
    membership check against reference data."""
    if not token or len(token) != 3:
        return None
    up = token.upper()
    return up if up in _ISO_4217_CODES else None


# A generic 3-letter alphabetic token candidate, checked against
# _ISO_4217_CODES by _valid_iso_code() after matching (never in the
# regex itself) -- this is what makes currency support data-driven:
# recognizing a NEW ISO code requires adding it to the set above, never
# a new regex branch or extraction rule.
# Trailing (?![A-Za-z]) prevents matching the first 3 letters of a longer
# word (e.g. "AUD" inside "audit") -- an ISO code token must stand alone,
# never be a mere prefix of unrelated surrounding text.
_ANY_ISO_CODE_RE = r"[A-Za-z]{3}(?![A-Za-z])"

_STRICT_AMOUNT_PATTERN = re.compile(
    # code_prefix: an OPTIONAL leading explicit ISO code before the
    # symbol (e.g. "CNY \xa510,000", "USD $10,000") -- validated in code
    # against _ISO_4217_CODES, never assumed. When present, it resolves
    # an otherwise-ambiguous native symbol (\xa5 alone means nothing here;
    # it's added to the symbol class purely so it can be captured
    # alongside an explicit code) or flags a CONFLICT against a symbol
    # that itself unambiguously names a different currency -- neither
    # case silently picks one side.
    rf"(?:(?P<code_prefix>[A-Za-z]{{3}})\s+)?"
    rf"(?:(?P<symbol>₹|Rs\.?|\$|€|£|¥|{_ANY_ISO_CODE_RE})\s*(?P<number_sym>\d+(?:,\d+)*(?:\.\d+)?)\s*(?P<scale_sym>lakhs?|crores?|k|m|million|billion)?)"
    r"|"
    rf"(?:(?P<number_word>\d+(?:,\d+)*(?:\.\d+)?)\s*(?P<scale_word>lakhs?|crores?|k|m|million|billion)?\s*(?P<code_after>{_ANY_ISO_CODE_RE}|rupees?|dollars?|euros?))",
    re.IGNORECASE,
)

_RANGE_AMOUNT_PATTERN = re.compile(
    rf"(?P<symbol>₹|Rs\.?|\$|€|£|{_ANY_ISO_CODE_RE})?\s*"
    rf"(?P<min>\d+(?:,\d+)*(?:\.\d+)?)\s*(?:[-–—]|to)\s*(?P<symbol_max>₹|Rs\.?|\$|€|£|{_ANY_ISO_CODE_RE})?\s*"
    r"(?P<max>\d+(?:,\d+)*(?:\.\d+)?)\s*"
    r"(?P<scale>lakhs?|crores?|k|m|million|billion)?\s*"
    rf"(?P<code_after>{_ANY_ISO_CODE_RE}|rupees?|dollars?|euros?)?",
    re.IGNORECASE,
)

_OBSERVATION_PERIOD_RE = re.compile(
    r"(?:over|during|for|in|covering|across|spanning)\s+"
    r"(?:the\s+(?P<retro_modifier>past|last|previous|preceding)\s+|a\s+period\s+of\s+|an?\s+)?"
    r"(?P<count>one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)-?\s*(?P<unit>months?|years?|weeks?|days?|quarters?)"
    r"|(?:over|during|for|in|covering|across|spanning)\s+the\s+"
    r"(?P<bare_unit>past|last|previous|preceding)\s+(?P<bare_unit_noun>year|month|week|quarter)\b",
    re.IGNORECASE,
)

# Generic, domain-neutral vocabulary of "quantifiable unit" nouns a
# financial rate/amount can be attached to -- discrete occurrence nouns
# (event, delivery, batch, incident, transaction, nonconformity, unit,
# item, occurrence) AND time-duration nouns (hour, day, week, month).
# Deliberately a single shared list so a rate expressed as "per hour" or
# "10 hours" is recognized exactly like "per event" / "10 events" --
# never a special case for any specific unit.
_UNIT_NOUN_SINGULAR = (
    r"event|delivery|batch|incident|transaction|nonconformity|unit|item|"
    r"occurrence|defect|hour|day|week|month"
)
_UNIT_NOUN_PLURAL = (
    r"events?|defects?|deliveries|batches|incidents?|transactions?|"
    r"nonconformit(?:y|ies)|units?|items?|occurrences?|hours?|days?|weeks?|months?"
)

# Broad unit-noun compatibility classes: a rate/quantity pairing may only
# be multiplied together when both fall in the SAME class -- an hourly
# RATE ("INR 12,000/hour") must never be multiplied by an EVENT COUNT
# ("10 nonconformities"); the units are not interchangeable even though
# both are "a number of things." Deliberately broad (not per-word) so a
# rate stated "per incident" still links to a quantity stated in
# "events" (both name a generic occurrence), matching ordinary usage.
_UNIT_NOUN_CLASS = {
    "event": "OCCURRENCE", "incident": "OCCURRENCE", "occurrence": "OCCURRENCE",
    "defect": "OCCURRENCE", "nonconformity": "OCCURRENCE", "transaction": "OCCURRENCE",
    # Time units are NOT mutually interchangeable -- a rate stated per
    # hour must never multiply a quantity stated in days, and vice versa,
    # even though both are "a stated time duration." Each keeps its own
    # class; no automatic hour<->day<->week<->month conversion is assumed
    # (that would require an explicit, evidence-stated conversion ratio,
    # which this engine does not invent).
    "hour": "TIME_HOUR", "day": "TIME_DAY", "week": "TIME_WEEK", "month": "TIME_MONTH",
    # Discrete-object units are NOT mutually interchangeable either -- a
    # batch can contain multiple units, a delivery can contain multiple
    # batches, so "N units" must never multiply a "per batch" rate (or
    # vice versa) without an evidence-stated conversion ratio.
    "unit": "UNIT", "item": "ITEM", "batch": "BATCH", "delivery": "DELIVERY",
}


def _unit_noun_class(word: str | None) -> str | None:
    """Normalize a matched unit noun (singular or plural, e.g.
    "deliveries"/"nonconformities") to its broad compatibility class, or
    None if unmatched/unrecognized (treated as a wildcard by the caller)."""
    if not word:
        return None
    w = word.lower()
    if w in _UNIT_NOUN_CLASS:
        return _UNIT_NOUN_CLASS[w]
    if w == "nonconformities":
        return _UNIT_NOUN_CLASS["nonconformity"]
    if w.endswith("ies") and w[:-3] + "y" in _UNIT_NOUN_CLASS:
        return _UNIT_NOUN_CLASS[w[:-3] + "y"]
    # Standard English "-es" plural after ch/sh/x/s/z (e.g. "batches" ->
    # "batch", "matches" -> "match") -- stripping only a trailing "s"
    # leaves "batche", which never matches, silently falling through to
    # the unrecognized/wildcard case and defeating the very compatibility
    # check this class exists to enforce.
    if w.endswith("es") and w[:-2] in _UNIT_NOUN_CLASS:
        return _UNIT_NOUN_CLASS[w[:-2]]
    if w.endswith("s") and w[:-1] in _UNIT_NOUN_CLASS:
        return _UNIT_NOUN_CLASS[w[:-1]]
    return None


_PER_EVENT_RE = re.compile(
    rf"\bper\s+(?P<rate_unit_a>{_UNIT_NOUN_SINGULAR})\b"
    rf"|(?<=[\d,.])\s*/\s*(?P<rate_unit_b>{_UNIT_NOUN_SINGULAR})\b"
    rf"|\beach\b",
    re.IGNORECASE,
)

# "average" stated alongside a per-event/per-unit amount, with no
# explicit event count elsewhere in the statement, inherently implies an
# UNSPECIFIED population of 2+ occurrences (an average over exactly one
# instance is not a meaningful "average") -- treating it as a single
# verified event (the calculator's count-or-1 fallback) would fabricate
# a population size the evidence explicitly does not state.
_AVERAGE_WORD_RE = re.compile(r"\baverage\b", re.IGNORECASE)

_EVENT_COUNT_WORD_RE = re.compile(
    # The digit run must be a genuine, WHOLE standalone number -- either
    # properly comma-grouped in triplets (e.g. "1,000", "12,500") or plain
    # digits with no comma at all (e.g. "10", "1000") -- never a fragment
    # of a larger comma-grouped monetary amount (rejecting a bare "12" or
    # "000" out of "12,000 per hour" via the surrounding lookaround).
    # Immediately followed by "per"/"/" is a RATE denominator ("12,000 per
    # hour"), not a count of that unit, so it's excluded via the negative
    # lookahead rather than by refusing to match commas at all.
    rf"\b(?P<word_count>one|two|three|four|five|six|seven|eight|nine|ten|"
    rf"(?<!,)(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?!,\d))"
    rf"(?!\s*(?:per\b|/))\s+(?:[\w-]+\s+){{0,3}}?(?P<count_unit>{_UNIT_NOUN_PLURAL}|times?|occasions?)\b",
    re.IGNORECASE,
)

# A plural/multi-event population stated WITHOUT a specific countable
# number ("some nonconformities", "several incidents") -- distinct from
# _EVENT_COUNT_WORD_RE (a specific number) and from a single bare noun
# (which implies one instance, not an unresolved population). Generalizes
# across domains via the same shared unit-noun vocabulary used elsewhere,
# never finding-specific vocabulary.
_VAGUE_QUANTITY_RE = re.compile(
    rf"\b(?:some|several|multiple|many|numerous|various)\s+(?:[\w-]+\s+){{0,3}}?"
    rf"(?:{_UNIT_NOUN_PLURAL})\b",
    re.IGNORECASE,
)

# A statement framed as backward-looking / historical context, as
# distinct from the CURRENT finding's own facts -- structural framing
# language, never tied to any specific domain or finding. Deliberately
# narrow: generic duration phrasing like "over a period of 4 months"
# describes how long the CURRENT finding's own condition persisted, NOT a
# reference to separate past incidents, so it must not match here.
_HISTORICAL_MARKER_RE = re.compile(
    r"\bhistorically\b|\bin\s+the\s+past\b|\bpreviously\b|"
    r"\bprior\s+(?:incidents?|occurrences?|events?)\b|"
    r"\bpast\s+(?:incidents?|occurrences?|events?)\b|"
    r"\bhistor(?:y|ical)\s+(?:shows?|indicates?|data|records?|trend)\b|"
    r"\bsimilar\s+(?:incidents?|events?|occurrences?|nonconformit(?:y|ies))\s+"
    r"(?:occurred|have\s+occurred|were\s+recorded|were\s+reported)\b|"
    r"\bover\s+the\s+(?:past|last)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:years?|months?)\b|"
    r"\breview\s+period\b",
    re.IGNORECASE,
)

_POTENTIAL_ADDITIONAL_RE = re.compile(
    r"(?P<pot_count>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+additional\s+(?:potentially\s+affected\s+|exposed\s+)?(?:deliveries|events?|batches|units?|shipments?)",
    re.IGNORECASE,
)

_WORD_TO_INT = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}

_RECOVERY_RE = re.compile(
    r"(?:recovered|recovery|reimbursed|refunded|recouped|returned)\s*(?:from\s+[\w\s]+\s+)?(?:of\s+)?(?P<amount>(?:₹|Rs\.?|\$|€|£|INR|USD)?\s*\d+(?:,\d+)*(?:\.\d+)?)",
    re.IGNORECASE,
)

_ZERO_RECOVERY_RE = re.compile(
    r"\b(?:no\s+recovery|zero\s+recovery|unrecoverable|irrecoverable|no\s+amount\s+recovered|recovery\s+was\s+(?:nil|zero))\b",
    re.IGNORECASE,
)

_REWORK_RE = re.compile(r"\b(?:rework|repair|re-processing)\b", re.IGNORECASE)
_SCRAP_RE = re.compile(r"\b(?:scrap|discarded|written\s*off)\b", re.IGNORECASE)
_DOWNTIME_RE = re.compile(r"\b(?:downtime|idle\s+time|line\s+stoppage)\b", re.IGNORECASE)
_PENALTY_RE = re.compile(r"\b(?:penalty|fine|regulatory\s+sanction)\b", re.IGNORECASE)
_REMEDIATION_RE = re.compile(
    r"\b(?:remediation\s+cost|capa\s+cost|implementation\s+cost|fix\s+cost)\b"
    r"|\b(?:remediation|corrective\s+action|preventive\s+action|capa|the\s+fix|implementation)\b[^.;]{0,30}?\b(?:will\s+cost|is\s+(?:expected|estimated|projected)\s+to\s+cost|cost(?:s|ing)?|estimate\s+is|estimate\s+of)\b"
    r"|\bcost\s+(?:of|to)\s+(?:implement|remediate|fix)\b"
    r"|\bremediation\s+estimate\b",
    re.IGNORECASE,
)

# An explicit "total"/"combined" marker on a remediation-cost statement --
# the single generic signal this extractor trusts to mean "this figure
# is the sum of the program's components," as distinct from one
# component among several or one alternative among several. Structural
# only (the word "total"/"combined"/"overall", never tied to any
# specific program name), so a genuinely stated total is recognized
# without the engine ever inventing which components it covers.
_REMEDIATION_TOTAL_RE = re.compile(
    r"\b(?:total|combined|overall|aggregate)\b[^.;]{0,40}?\b(?:cost|cost\s+of\s+(?:remediation|implementation|the\s+(?:fix|program|capa)))\b"
    r"|\b(?:cost\s+of\s+the\s+)?(?:total|combined|overall)\s+(?:remediation|capa|corrective\s+action|implementation)\s+program\b",
    re.IGNORECASE,
)
_COMPENSATION_RE = re.compile(r"\b(?:compensation|customer\s+credit|settlement)\b", re.IGNORECASE)


def is_historical_marker(text: str) -> bool:
    """Public accessor for the historical-framing detector used elsewhere in the
    pipeline (e.g. the plan-investigation fallback) that needs the same structural
    signal this extractor uses for CURRENT_FINDING vs HISTORICAL population framing,
    without reaching into this module's private regex objects."""
    return bool(_HISTORICAL_MARKER_RE.search(text))


def is_remediation_marker(text: str) -> bool:
    """Public accessor for the remediation-cost framing detector -- see
    `is_historical_marker` for why this wraps a private regex instead of exposing it."""
    return bool(_REMEDIATION_RE.search(text))


# Deterministic parsing aliases ONLY (never arithmetic/calculation
# logic) -- a handful of unambiguous currency symbols and English words
# that map to a single ISO code. Everything else is resolved generically
# via _valid_iso_code() against the full ISO 4217 reference set, so
# recognizing a currency never requires adding an entry here.
_CURRENCY_ALIAS_MAP = {
    "₹": "INR", "RS": "INR", "RUPEE": "INR", "RUPEES": "INR",
    "$": "USD", "DOLLAR": "USD", "DOLLARS": "USD",
    "€": "EUR", "EURO": "EUR", "EUROS": "EUR",
    "£": "GBP", "POUND": "GBP", "POUNDS": "GBP",
}


def _normalize_currency(sym_or_code: str | None) -> str | None:
    """Resolve a matched symbol/code/word to its ISO 4217 currency, or
    None if it cannot be reliably determined. Never guesses and never
    defaults to any particular currency merely because resolution
    failed -- the caller is responsible for treating None as UNKNOWN
    rather than fabricating a value."""
    if not sym_or_code:
        return None
    s = sym_or_code.strip()
    su = s.upper().rstrip(".")
    if su in _CURRENCY_ALIAS_MAP:
        return _CURRENCY_ALIAS_MAP[su]
    return _valid_iso_code(su)


def _resolve_currency_with_prefix(code_prefix: str | None, symbol_or_code: str | None) -> str | None:
    """Resolve a matched currency, honoring an optional leading explicit
    ISO code (e.g. "CNY \xa510,000", "USD $10,000"):

    - No code_prefix: resolve `symbol_or_code` alone, exactly as before
      (preserves the existing "$" -> USD, "₹" -> INR, etc. behavior).
    - code_prefix present and valid, symbol/code unresolved (e.g. a bare
      native symbol like \xa5 with no alias): the explicit code wins --
      this is what makes "CNY \xa510,000" resolve to CNY and "JPY \xa510,000"
      resolve to JPY, without ever guessing what \xa5 alone means.
    - code_prefix present and valid, symbol/code ALSO resolves but to a
      DIFFERENT currency (e.g. "USD ₹10,000"): a genuine conflict --
      return None rather than silently picking either side.
    - code_prefix present and valid, symbol/code resolves to the SAME
      currency (e.g. "USD $10,000"): confirmed, return it.
    """
    prefix_curr = _valid_iso_code(code_prefix) if code_prefix else None
    own_curr = _normalize_currency(symbol_or_code)
    if prefix_curr and own_curr and prefix_curr != own_curr:
        return None
    return prefix_curr or own_curr


def _has_valid_amount(text: str) -> bool:
    """True only if `text` contains a monetary match with a currency
    that actually resolves -- a bare structural match (e.g. "for 5 days"
    superficially matching the amount pattern via "for"/"day" as a
    3-letter code candidate) does not count. Used wherever "does this
    text already state an amount" gates bare-quantity/vague-quantity
    extraction, so a text with no REAL amount is never mistaken for one
    merely because a stray 3-letter word sits next to a number."""
    for m in _STRICT_AMOUNT_PATTERN.finditer(text):
        if _resolve_currency_with_prefix(m.group("code_prefix"), m.group("symbol") or m.group("code_after")) is not None:
            return True
    for m in _RANGE_AMOUNT_PATTERN.finditer(text):
        if _normalize_currency(m.group("symbol") or m.group("symbol_max") or m.group("code_after")) is not None:
            return True
    return False


def _parse_amount(val_str: str | None, scale: str | None) -> float | None:
    if not val_str:
        return None
    try:
        clean = val_str.replace(",", "").strip()
        amt = float(clean)
        if scale:
            sc = scale.lower()
            if "lakh" in sc:
                amt *= 100_000.0
            elif "crore" in sc:
                amt *= 10_000_000.0
            elif sc == "k":
                amt *= 1_000.0
            elif sc in ("m", "million"):
                amt *= 1_000_000.0
            elif "billion" in sc:
                amt *= 1_000_000_000.0
        return amt
    except (ValueError, TypeError):
        return None


_PER_YEAR_RATE_RE = re.compile(r"\bper\s+year\b|(?<=[\d,.])\s*/\s*year\b", re.IGNORECASE)


def _extract_observation_period_months(text: str) -> float | None:
    m = _OBSERVATION_PERIOD_RE.search(text)
    if not m:
        # "X per year" / "X/year" directly states an ANNUAL figure -- the
        # amount itself already spans a 12-month period; this is a
        # distinct signal from _OBSERVATION_PERIOD_RE's "over/during/for
        # a stated duration" framing, so it only applies as a fallback
        # when no other period phrase already matched.
        if _PER_YEAR_RATE_RE.search(text):
            return 12.0
        return None
    try:
        if m.group("bare_unit_noun"):
            # "over/during/for/in the past/last year|month|week|quarter"
            # -- an implicit single unit, never a fabricated one: the
            # finding itself states the (singular) span.
            cnt = 1.0
            unit = m.group("bare_unit_noun").lower() + "s"
        else:
            cnt = float(_parse_count_token(m.group("count")) or 0)
            unit = m.group("unit").lower()
        if cnt <= 0:
            return None
        if "year" in unit:
            return cnt * 12.0
        elif "month" in unit:
            return cnt
        elif "week" in unit:
            return cnt / 4.33
        elif "quarter" in unit:
            return cnt * 3.0
        elif "day" in unit:
            return cnt / 30.0
    except (ValueError, TypeError):
        pass
    return None


def _parse_count_token(val: str | None) -> int | None:
    if not val:
        return None
    val_low = val.lower().strip()
    if val_low in _WORD_TO_INT:
        return _WORD_TO_INT[val_low]
    try:
        return int(val_low.replace(",", ""))
    except ValueError:
        return None


def extract_financial_observations(
    finding_text: str,
    evidence_ledger: list[EvidenceItem] | None = None,
    evidence_claims: list[EvidenceClaim] | None = None,
) -> tuple[list[FinancialObservation], bool, list[str]]:
    """Extract structured FinancialObservation facts strictly maintaining source verification status.

    Returns (observations, has_conflict, currency_conflicts) -- has_conflict
    flags conflicting stated AMOUNTS; currency_conflicts separately lists
    any explicit-code-vs-symbol currency mismatches (e.g. "USD ₹10,000")
    encountered, each excluded from `observations`.
    """
    observations: list[FinancialObservation] = []
    currencies_found: set[str] = set()

    # 5th element: the ORIGINAL EvidenceStatus name (e.g. "BELIEF"),
    # preserved purely as rendering provenance -- v_stat (element 2)
    # remains the sole authoritative calculation bucket and is computed
    # exactly as before; nothing about eligibility/aggregation changes.
    sources: list[tuple[str, str, str, str | None, str | None]] = []
    if evidence_ledger:
        for idx, item in enumerate(evidence_ledger):
            v_stat = "VERIFIED" if item.status == EvidenceStatus.VERIFIED else ("REPORTED" if item.status == EvidenceStatus.REPORTED else "UNVERIFIED")
            ev_id = f"E{idx+1}"
            _orig_status = item.status.value if item.status is not None else None
            sources.append((item.claim, v_stat, ev_id, item.source_reference or item.source, _orig_status))

    # `evidence_claims` (app.models.agent.EvidenceClaim, produced by
    # extract_claims() from THIS SAME evidence_ledger -- see
    # understanding.py) is a claim-level restructuring of the ledger's
    # own sentences, never an independent additional evidence source.
    # Feeding both into `sources` duplicates every fact (e.g. a quantity
    # and a rate each appear twice), which defeats the cross-evidence
    # quantity/rate linking below: it sees two candidate quantities and
    # two candidate rates for what is really one pair, and correctly
    # refuses to guess which belongs with which -- silently losing a
    # calculation that a single evidence_ledger alone would have found.
    # Only fall back to evidence_claims when no evidence_ledger was
    # supplied at all.
    if evidence_claims and not evidence_ledger:
        for c in evidence_claims:
            _c_status = getattr(c, "status", None)
            v_stat = "VERIFIED" if _c_status == EvidenceStatus.VERIFIED else "UNVERIFIED"
            src_ref = getattr(c, "source_doc", None) or getattr(c, "evidence_reference", None) or getattr(c, "source", None)
            _orig_status = _c_status.value if _c_status is not None else None
            sources.append((c.text, v_stat, c.claim_id, src_ref, _orig_status))

    if not sources and finding_text:
        sources.append((finding_text, "REPORTED", "E0", "Audit Observation", "REPORTED"))

    obs_idx = 1
    has_conflict = False
    # Distinct from `has_conflict` (conflicting stated AMOUNTS): a
    # genuine explicit-code-vs-symbol currency mismatch (e.g.
    # "USD ₹10,000") where the code and the native symbol each validly
    # resolve but to DIFFERENT currencies -- surfaced separately so the
    # caller can report *why* no amount was extracted instead of that
    # looking identical to "no financial data present at all".
    currency_conflicts: list[str] = []
    # Observations stated as an "average" per-unit/per-event amount with
    # no LOCAL count of their own -- final VERIFIED-downgrade decision is
    # deferred until after cross-evidence quantity linking runs (see the
    # deferred pass below), since linking may still supply a compatible,
    # independently VERIFIED count from a separate evidence statement.
    _avg_without_count_candidates: list[FinancialObservation] = []

    # Bare quantity facts: a source statement that states a count of
    # events/items/batches/etc but contains NO monetary amount of its own
    # (e.g. "Eight verified nonconformities were identified" with the cost
    # stated separately elsewhere). Tracked so a PER_EVENT/PER_UNIT amount
    # extracted from a *different* source statement can be linked to its
    # actual population size instead of silently defaulting to a single
    # event -- never guessed when more than one such bare-quantity
    # statement exists (ambiguous which quantity applies to which amount).
    bare_quantity_facts: list[tuple[int, str, str, str, str | None]] = []  # (count, status, evidence_id, population, unit_class)
    has_ambiguous_population = False

    for text, v_stat, src_id, src_ref, orig_status in sources:
        # Scoped per-source (not module/call-wide): dedup only guards
        # against the SAME source statement matching the same amount
        # twice via overlapping regex passes -- it must never suppress a
        # genuinely distinct evidence claim that happens to state the
        # same numeric amount as another claim (e.g. two independent
        # historical-frequency claims that coincidentally share a
        # per-event cost but disagree on event count must both survive
        # extraction so the conflict between them can be detected).
        seen_amounts: set[tuple[float, str, bool]] = set()
        obs_period = _extract_observation_period_months(text)
        has_zero_rec = bool(_ZERO_RECOVERY_RE.search(text))
        _per_event_m = _PER_EVENT_RE.search(text)
        is_per_event = bool(_per_event_m)
        _rate_unit_class = (
            _unit_noun_class(_per_event_m.group("rate_unit_a") or _per_event_m.group("rate_unit_b"))
            if _per_event_m else None
        )
        is_stated_average = bool(_AVERAGE_WORD_RE.search(text))

        # A statement explicitly framed as backward-looking context (past
        # incidents distinct from the current finding) belongs to the
        # HISTORICAL population -- its amounts feed recurrence/annualization
        # analysis, never the current finding's own gross exposure. Two
        # independent signals: a fixed set of framing phrases, OR the
        # SAME retrospective modifier (past/last/previous/preceding)
        # already recognized structurally by the observation-period regex
        # -- reusing that structural signal instead of a second separate
        # word list for the same underlying concept (retrospection).
        _period_m = _OBSERVATION_PERIOD_RE.search(text)
        _period_unit_for_pop = (_period_m.group("unit") or _period_m.group("bare_unit_noun") or "").lower() if _period_m else ""
        _retro_modifier = (
            (_period_m.group("retro_modifier") or _period_m.group("bare_unit"))
            if (_period_m and _period_unit_for_pop.startswith(("month", "year", "quarter")))
            else None
        )
        fin_population = "HISTORICAL" if (_HISTORICAL_MARKER_RE.search(text) or _retro_modifier) else "CURRENT_FINDING"

        # Event counts: matched against a copy of the text with any
        # observation-period phrase masked out first -- "12 months" in
        # "over the past 12 months" is a DURATION, never a count of
        # events, but without masking it can win a bare regex search
        # against a genuine event count appearing later in the same
        # sentence (e.g. "Records covering 12 months identified 8
        # events" must count 8 events, not misread "12 months" as 12
        # events). Masking preserves the string length so unrelated
        # character-offset logic elsewhere is unaffected.
        #
        # Scoped to month/year/quarter units only: "N days"/"N hours" is
        # the conventional granularity for a rate-multiplier QUANTITY
        # ("10 hours" x "INR 12,000/hour"), not an annualization
        # observation period, so masking it would break that distinct,
        # already-covered semantic (duration x rate). Month/year/quarter
        # phrasing is unambiguously the annualization-period case.
        _period_masked_text = text
        if _period_m and _period_unit_for_pop.startswith(("month", "year", "quarter")):
            _s, _e = _period_m.span()
            _period_masked_text = text[:_s] + (" " * (_e - _s)) + text[_e:]

        # Reject an event-count match whose digit span OVERLAPS a monetary
        # amount match -- "INR 500 per day" must never also read "500" as
        # a count of "500 days" just because "day" is a recognized
        # countable unit noun immediately after the number. Without this,
        # the SAME numeral is double-purposed as both the rate's value
        # and a fabricated event count, corrupting the observation's own
        # event_count field directly (not merely the cross-clause
        # linking path already guarded below).
        # Only spans with a VALID currency count as genuine amount matches
        # -- with generic ISO-code recognition, a bare 3-letter word
        # immediately after a number (e.g. "8 inc[idents]") can match the
        # pattern's code_after group without being a real currency; such
        # a match must not be allowed to block legitimate event-count
        # extraction just because it superficially looks amount-shaped.
        _amount_spans = [
            m.span() for m in _STRICT_AMOUNT_PATTERN.finditer(text)
            if _resolve_currency_with_prefix(m.group("code_prefix"), m.group("symbol") or m.group("code_after")) is not None
        ] + [
            m.span() for m in _RANGE_AMOUNT_PATTERN.finditer(text)
            if _normalize_currency(m.group("symbol") or m.group("symbol_max") or m.group("code_after")) is not None
        ]

        def _overlaps_amount(span: tuple[int, int]) -> bool:
            return any(span[0] < a_end and a_start < span[1] for a_start, a_end in _amount_spans)

        # Blank out each recognized monetary amount's own span before
        # searching for an event count -- a comma-grouped monetary amount
        # (e.g. "INR 15,000") is itself a well-formed number and can
        # satisfy this pattern's own number+unit shape (e.g. "15,000
        # across 8 verified incidents" reads as if "15,000" were the
        # count of "incidents"), which would wrongly consume the match via
        # .search()'s leftmost-first semantics and hide the REAL count
        # token (the actual "8") appearing later in the same sentence.
        # Masking the amount's own characters (as spaces, preserving
        # offsets) means the digit run can never start matching there,
        # while the genuine count phrase elsewhere in the text is
        # untouched.
        _count_search_text = _period_masked_text
        for _a_start, _a_end in _amount_spans:
            _count_search_text = _count_search_text[:_a_start] + (" " * (_a_end - _a_start)) + _count_search_text[_a_end:]

        ev_cnt = None
        ev_m = _EVENT_COUNT_WORD_RE.search(_count_search_text)
        if ev_m and _overlaps_amount(ev_m.span()):
            ev_m = None
        if ev_m:
            ev_cnt = _parse_count_token(ev_m.group("word_count"))
            _qty_unit_class = _unit_noun_class(ev_m.group("count_unit"))
            if ev_cnt and not _has_valid_amount(text):
                bare_quantity_facts.append((ev_cnt, v_stat, src_id, fin_population, _qty_unit_class))
        elif _VAGUE_QUANTITY_RE.search(_period_masked_text) and not _has_valid_amount(text):
            has_ambiguous_population = True

        # Potential additional events
        pot_cnt = None
        pot_m = _POTENTIAL_ADDITIONAL_RE.search(text)
        if pot_m:
            pot_cnt = _parse_count_token(pot_m.group("pot_count"))

        # Check for range matches
        for rm in _RANGE_AMOUNT_PATTERN.finditer(text):
            sym = rm.group("symbol") or rm.group("symbol_max") or rm.group("code_after")
            if not sym:
                continue
            curr = _normalize_currency(sym)
            if curr is None:
                # The candidate token was a bare 3-letter word that is
                # NOT a recognized ISO 4217 code (e.g. incidental prose
                # adjacent to a number) -- never fabricate a currency.
                continue
            min_v = _parse_amount(rm.group("min"), rm.group("scale"))
            max_v = _parse_amount(rm.group("max"), rm.group("scale"))
            if min_v and max_v:
                obs = FinancialObservation(
                    observation_id=f"FIN-OBS-{obs_idx:03d}",
                    amount_min=min_v,
                    amount_max=max_v,
                    currency=curr,
                    amount_type=FinancialAmountType.POTENTIAL_EXPOSURE,
                    observation_period_months=obs_period,
                    event_count=ev_cnt,
                    potential_event_count=pot_cnt,
                    source_evidence_ids=[src_id] if src_id else [],
                    verification_status=v_stat,  # type: ignore[arg-type]
                    source_reference=src_ref,
                    financial_population=fin_population,  # type: ignore[arg-type]
                    source_evidence_status=orig_status,
                )
                observations.append(obs)
                obs_idx += 1
                currencies_found.add(curr)

        # Match single financial numbers
        for match in _STRICT_AMOUNT_PATTERN.finditer(text):
            num_str = match.group("number_sym") or match.group("number_word")
            scale_str = match.group("scale_sym") or match.group("scale_word")
            curr_str = match.group("symbol") or match.group("code_after")

            amt = _parse_amount(num_str, scale_str)
            if amt is None or amt <= 0:
                continue

            # Clause-bounded window: a recovery keyword in a *different*
            # clause (e.g. "Scrap cost of INR 25,000 was incurred;
            # recovery was nil.") must never attribute that amount to
            # recovery -- only a keyword in the SAME clause as the amount
            # governs its meaning, so the window is clipped at the nearest
            # clause boundary (.;) on either side before applying the
            # +/-35 character radius. "of which" is also treated as a
            # sub-clause boundary within a single comma-joined sentence
            # (e.g. "overpaid by X, of which Y was recovered.") -- without
            # it, a recovery keyword describing only the SECOND (post-"of
            # which") amount would incorrectly also attribute the FIRST
            # (pre-"of which") amount to recovery, since both share one
            # period-bounded clause.
            _owl = text.lower()
            _of_which_before = _owl.rfind("of which", 0, match.start())
            _of_which_after = _owl.find("of which", match.end())
            _clause_start = max(
                text.rfind(".", 0, match.start()),
                text.rfind(";", 0, match.start()),
                (_of_which_before + len("of which")) if _of_which_before != -1 else -1,
            ) + 1
            _clause_end_dot = text.find(".", match.end())
            _clause_end_semi = text.find(";", match.end())
            _clause_end_candidates = [c for c in (_clause_end_dot, _clause_end_semi, _of_which_after) if c != -1]
            _clause_end = min(_clause_end_candidates) if _clause_end_candidates else len(text)
            match_start = max(_clause_start, match.start() - 35)
            match_end = min(_clause_end, match.end() + 35)
            window = text[match_start:match_end].lower()
            is_recovery_amt = bool(re.search(r"\b(?:recover(?:ed|y)?|refund(?:ed)?|reimburs(?:ed)?|recoup(?:ed)?)\b", window))

            curr = _resolve_currency_with_prefix(match.group("code_prefix"), curr_str)
            if curr is None:
                # Either the token doesn't resolve to a recognized
                # currency (e.g. a bare 3-letter word in unrelated
                # prose), or an explicit code conflicts with a
                # differently-resolving symbol (e.g. "USD ₹10,000")
                # -- never fabricate or silently pick a side either way.
                _code_prefix = match.group("code_prefix")
                _prefix_curr = _valid_iso_code(_code_prefix) if _code_prefix else None
                _own_curr = _normalize_currency(curr_str)
                if _prefix_curr and _own_curr and _prefix_curr != _own_curr:
                    currency_conflicts.append(f"{_code_prefix} {curr_str}".strip())
                continue
            currencies_found.add(curr)

            if (amt, curr, is_recovery_amt) in seen_amounts:
                continue
            seen_amounts.add((amt, curr, is_recovery_amt))

            if is_recovery_amt:
                amt_type = FinancialAmountType.RECOVERY
            elif v_stat == "VERIFIED":
                amt_type = FinancialAmountType.DIRECT_LOSS
            else:
                amt_type = FinancialAmountType.POTENTIAL_EXPOSURE

            if not is_recovery_amt:
                if _REWORK_RE.search(text):
                    amt_type = FinancialAmountType.REWORK_COST
                elif _SCRAP_RE.search(text):
                    amt_type = FinancialAmountType.SCRAP_COST
                elif _DOWNTIME_RE.search(text):
                    amt_type = FinancialAmountType.DOWNTIME_COST
                elif _PENALTY_RE.search(text):
                    amt_type = FinancialAmountType.PENALTY
                elif _REMEDIATION_RE.search(text):
                    amt_type = FinancialAmountType.REMEDIATION_COST
                elif _COMPENSATION_RE.search(text):
                    amt_type = FinancialAmountType.CUSTOMER_COMPENSATION

            rec_stat = RecoveryStatus.VERIFIED_ZERO_RECOVERY if (has_zero_rec and v_stat == "VERIFIED") else RecoveryStatus.REQUIRES_VERIFICATION

            unit_amt = amt if is_per_event else None
            direct_amt = None if is_per_event else amt

            # A stated "average" per-event/per-unit amount with no
            # explicit count anywhere in the statement cannot be treated
            # as a single verified event's cost. The actual downgrade is
            # applied in a deferred pass AFTER cross-evidence quantity
            # linking runs (below) rather than here -- linking can still
            # supply a compatible, independently VERIFIED count from a
            # SEPARATE evidence statement, and deciding now (before
            # linking has had a chance to run) would incorrectly and
            # irreversibly mark the observation UNVERIFIED even when a
            # legitimate count is about to be linked to it.
            _obs_v_stat = v_stat
            _is_avg_without_local_count = bool(is_per_event and is_stated_average and not ev_cnt)

            obs = FinancialObservation(
                observation_id=f"FIN-OBS-{obs_idx:03d}",
                amount=direct_amt,
                unit_amount=unit_amt,
                currency=curr,
                amount_type=amt_type,
                observation_period_months=obs_period,
                event_count=ev_cnt,
                potential_event_count=pot_cnt,
                source_evidence_ids=[src_id] if src_id else [],
                verification_status=_obs_v_stat,  # type: ignore[arg-type]
                source_reference=src_ref,
                recovery_status=rec_stat,
                financial_population=fin_population,  # type: ignore[arg-type]
                rate_unit_class=_rate_unit_class if is_per_event else None,
                source_evidence_status=orig_status,
                is_aggregate_total=bool(_REMEDIATION_TOTAL_RE.search(text)),
            )
            observations.append(obs)
            obs_idx += 1
            if _is_avg_without_local_count:
                _avg_without_count_candidates.append(obs)

    if len(observations) > 1:
        verified_losses = [o for o in observations if o.verification_status == "VERIFIED" and o.amount_type in (FinancialAmountType.DIRECT_LOSS, FinancialAmountType.POTENTIAL_EXPOSURE)]
        if len(verified_losses) >= 2 and all(o.source_reference == verified_losses[0].source_reference for o in verified_losses):
            if any(o.amount != verified_losses[0].amount for o in verified_losses):
                has_conflict = True

        # Conflicting historical recurrence frequency: two or more VERIFIED
        # HISTORICAL observations stating the SAME per-event cost and
        # observation period but a DIFFERENT event count describe the same
        # underlying recurrence fact inconsistently -- silently picking
        # the first would fabricate a specific rate the evidence does not
        # actually agree on.
        _hist_freq_obs = [
            o for o in observations
            if o.verification_status == "VERIFIED"
            and o.financial_population == "HISTORICAL"
            and o.event_count is not None
            and o.unit_amount is not None
            and o.unit_amount > 0
            and o.observation_period_months is not None
        ]
        if len(_hist_freq_obs) >= 2:
            # Pairwise: two HISTORICAL facts describing the same
            # recurrence (matching on TWO of {cost/event, event count,
            # period}) but disagreeing on the third field are describing
            # the SAME underlying recurrence inconsistently -- a real
            # conflict, never silently combined or averaged. Checking all
            # three "which field differs" cases symmetrically (not just
            # event-count conflicts) so a stated cost-per-event
            # discrepancy or a stated period discrepancy is caught the
            # same way a stated event-count discrepancy already was.
            for _i in range(len(_hist_freq_obs)):
                for _j in range(_i + 1, len(_hist_freq_obs)):
                    _a, _b = _hist_freq_obs[_i], _hist_freq_obs[_j]
                    if _a.currency != _b.currency:
                        continue
                    _same_cost = _a.unit_amount == _b.unit_amount
                    _same_count = _a.event_count == _b.event_count
                    _same_period = _a.observation_period_months == _b.observation_period_months
                    _agreements = sum([_same_cost, _same_count, _same_period])
                    if _agreements == 2 and not (_same_cost and _same_count and _same_period):
                        has_conflict = True
                        break
                if has_conflict:
                    break

    # Link a single unambiguous bare quantity fact to a single unambiguous
    # per-unit/per-event amount that has no count of its own -- the
    # derived exposure's provenance is bounded by the WEAKER of the two
    # inputs (never upgraded), and linking is skipped entirely whenever
    # more than one candidate on either side would make the pairing a
    # guess rather than a determination.
    _unit_only_obs = [
        o for o in observations
        if o.unit_amount is not None and o.event_count is None and o.amount is None
    ]
    # Unit-class compatibility: an hourly RATE must never be multiplied by
    # an EVENT count, a per-batch RATE by a UNIT count, etc. -- wildcard
    # (None) on either side (e.g. a bare "each") is compatible with
    # anything, since it names no specific unit to conflict with.
    def _rate_and_qty_compatible(rate_cls: str | None, qty_cls: str | None) -> bool:
        return rate_cls is None or qty_cls is None or rate_cls == qty_cls

    if (
        len(bare_quantity_facts) == 1
        and len(_unit_only_obs) == 1
        and bare_quantity_facts[0][3] == _unit_only_obs[0].financial_population
        and _rate_and_qty_compatible(_unit_only_obs[0].rate_unit_class, bare_quantity_facts[0][4])
    ):
        _qty, _qty_status, _qty_evidence_id, _qty_population, _qty_unit_class = bare_quantity_facts[0]
        _target = _unit_only_obs[0]
        _rank = {"VERIFIED": 3, "REPORTED": 2, "UNVERIFIED": 1, "CONTRADICTED": 0}
        _linked_status = _target.verification_status if _rank.get(_target.verification_status, 0) <= _rank.get(_qty_status, 0) else _qty_status
        _target.event_count = _qty
        _target.verification_status = _linked_status  # type: ignore[assignment]
        if _qty_evidence_id and _qty_evidence_id not in _target.source_evidence_ids:
            _target.source_evidence_ids = [*_target.source_evidence_ids, _qty_evidence_id]
        _target.notes = (
            (_target.notes + " " if _target.notes else "")
            + f"Event count ({_qty}) derived from a separate evidence statement "
            f"({_qty_status}); linked exposure calculation is bounded by the weaker of "
            f"the two source provenances ({_linked_status})."
        )
    elif not bare_quantity_facts and has_ambiguous_population and len(_unit_only_obs) == 1:
        # A per-event/per-unit amount with NO specific count anywhere, but
        # a competing statement establishing the population is plural and
        # of UNKNOWN size ("some"/"several"/... nonconformities). Silently
        # treating this as a single verified event (the calculator's
        # cnt-or-1 fallback) would fabricate a specific population size
        # that the evidence explicitly does not support -- downgrade so
        # it can never surface as a VERIFIED gross exposure.
        _target = _unit_only_obs[0]
        if _target.verification_status == "VERIFIED":
            _target.verification_status = "UNVERIFIED"
            _target.notes = (
                (_target.notes + " " if _target.notes else "")
                + "Population size is stated as plural but unspecified elsewhere in the "
                "evidence; a single-event exposure cannot be assumed."
            )

    # Deferred "average without count" downgrade: applied only NOW, after
    # cross-evidence quantity linking above has had its chance to supply
    # obs.event_count from a separate, compatible-population evidence
    # statement. If a count was linked, the observation is left exactly
    # as the linking step set it (already correctly bounded by the weaker
    # of the two source provenances); only an observation that STILL has
    # no count at all is downgraded.
    for _avg_obs in _avg_without_count_candidates:
        if _avg_obs.event_count is None and _avg_obs.verification_status == "VERIFIED":
            _avg_obs.verification_status = "UNVERIFIED"  # type: ignore[assignment]

    return observations, has_conflict, currency_conflicts
