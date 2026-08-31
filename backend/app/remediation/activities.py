"""Canonical remediation ACTIVITY collection + the single deterministic
normalization step.

There is exactly ONE canonical activity collection per result. Every
downstream remediation field is derived from it:

    implementation_activities  = [a.description for a in canon]
    conditional_activities     = [a.description for a in canon if a.is_conditional]
    unpriced_activities        = [a.description for a in canon if not a.is_priced]
    evidence_improves_estimate = unique(a.pricing_evidence_required
                                        for a in canon if not a.is_priced
                                        and a.pricing_evidence_required)

The LLM proposes two parallel collections -- `activities` and `cost_components`.
`build_canonical_activities` reconciles BOTH into the single canonical
collection: a cost component is attached to the activity it prices (by
explicit `activity_ids`, by being the only activity, or by description
overlap); a component that prices no known activity BECOMES a canonical
activity so its cost -- and its pricing-evidence gap -- is never orphaned.

Deterministic, domain-agnostic (keys on the activity's grammatical role via a
small verb-class classifier, never on finding vocabulary), no LLM call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.remediation.scope import _pricing_need

# The five structural activity kinds -- identical to the KINDS `scope.py`
# assigns, so an LLM activity and a deterministic-scope activity of the same
# role price the same way.
KIND_IMMEDIATE_CORRECTION = "IMMEDIATE_CORRECTION"
KIND_SCOPE_ASSESSMENT = "SCOPE_ASSESSMENT"
KIND_CAUSAL_INVESTIGATION = "CAUSAL_INVESTIGATION"
KIND_SYSTEMIC_STRENGTHENING = "SYSTEMIC_STRENGTHENING"
KIND_EFFECTIVENESS_VERIFICATION = "EFFECTIVENESS_VERIFICATION"

CONFIRMED = "CONFIRMED"
CONDITIONAL = "CONDITIONAL"

PRICED = "PRICED"
UNPRICED = "UNPRICED"

PROV_LLM = "LLM"
PROV_SCOPE = "DETERMINISTIC_SCOPE"
PROV_COST_COMPONENT = "LLM_COST_COMPONENT"
PROV_CANONICAL = "CANONICAL_LLM"

# Canonical `SemRemediationAction.disposition` -> the structural KIND used for
# pricing. The canonical LLM's own classification is authoritative; this is a
# 1:1 translation, NOT a re-classification (no verb inspection).
_DISPOSITION_TO_KIND = {
    "IMMEDIATE_CORRECTION": KIND_IMMEDIATE_CORRECTION,
    "CONTAINMENT": KIND_IMMEDIATE_CORRECTION,
    "CORRECTIVE_ACTION": KIND_SYSTEMIC_STRENGTHENING,
    "CONDITIONAL_SYSTEMIC": KIND_SYSTEMIC_STRENGTHENING,
    "EFFECTIVENESS_CHECK": KIND_EFFECTIVENESS_VERIFICATION,
}

# Component <-> activity relationship states.
COMP_ATTACHED = "ATTACHED_TO_ACTIVITY"
COMP_INDEPENDENT = "INDEPENDENT_WORK"
COMP_UNRESOLVED_DRIVER = "UNRESOLVED_PRICING_DRIVER"


# --- text-intelligence primitives ----------------------------------------

# Semantic invariant:
#   An IMPLEMENTATION ACTIVITY describes WORK that must be performed.
#   A COST COMPONENT / PRICING DRIVER describes what PRICES an activity.
#
# A pricing-driver phrase is recognised STRUCTURALLY: its grammatical head is
# an economics / resource meta-noun (labour, effort, cost, fee, rate, hours,
# quotation, ...), or it is framed as "<cost-noun> for <work>" / "<work>
# cost". These meta-nouns are generic accounting vocabulary -- NOT a domain
# blacklist (cf. the finding-subject resolver's own `_FINANCIAL_META_NOUNS`).
_COST_HEAD_NOUNS = (
    r"labou?r|effort|efforts|cost|costs|costing|fee|fees|charge|charges|"
    r"expense|expenses|expenditure|expenditures|quotation|quotations|quote|"
    r"quotes|pricing|price|prices|rate|rates|budget|spend|spending|manpower|"
    r"staffing|personnel|resourc\w*|overhead|overheads|disbursement|"
    r"disbursements|outlay|hours|hour|man-?hours?|man-?days?|mandays|manhours|"
    r"headcount|fte|ftes|wages?|salar(?:y|ies)|premiums?|surcharge|levy|tariff|"
    r"materials?|parts?|supplies|consumables?|licen[cs]e|licen[cs]ing|"
    r"subscription|subscriptions"
)
_PRICING_MODIFIER = (
    r"internal|external|contractor|contractors|vendor|vendors|supplier|"
    r"suppliers|third-?party|professional|engineering|analyst|analysts|"
    r"consult\w*|specialist|in-?house|outsourced|additional|estimated|"
    r"projected|allocated|direct|indirect|fixed|variable|one-?time|recurring|"
    r"service|services|testing|training|documentation|equipment|tooling|"
    r"hardware|software|travel|logistics|freight|shipping|installation|"
    r"replacement|calibration|validation|inspection"
)
# Leads with (optional modifiers, then) a cost head noun: "Labour for X",
# "Contractor cost", "Analyst effort", "Internal engineering hours".
_PRICING_DRIVER_LEAD_RE = re.compile(
    rf"^\s*(?:the\s+|a\s+|an\s+|estimated\s+|approximate\s+)?"
    rf"(?:(?:{_PRICING_MODIFIER})[\s/-]+)*"
    rf"(?:{_COST_HEAD_NOUNS})\b",
    re.IGNORECASE,
)
# "<work> cost / effort / labour / fee" (head is the cost noun).
_PRICING_DRIVER_TRAIL_RE = re.compile(
    rf"\b(?:{_COST_HEAD_NOUNS})\s*$",
    re.IGNORECASE,
)
# "cost / price / budget / quotation of|for X"
_PRICING_OF_RE = re.compile(
    rf"^\s*(?:the\s+)?(?:{_COST_HEAD_NOUNS})\s+(?:of|for|to)\b",
    re.IGNORECASE,
)
# Strip the pricing frame to recover the WORK phrase it prices.
_STRIP_DRIVER_LEAD_RE = re.compile(
    rf"^\s*(?:the\s+|a\s+|an\s+|estimated\s+|approximate\s+)?"
    rf"(?:(?:{_PRICING_MODIFIER})[\s/-]+)*"
    rf"(?:{_COST_HEAD_NOUNS})\s+"
    r"(?:for|to|of|on|associated\s+with|related\s+to|required\s+for|needed\s+for|"
    r"incurred\s+(?:for|in|on)|spent\s+on|towards?|in\s+support\s+of)\s+"
    r"(?:the\s+|a\s+|an\s+)?"
    r"(?:completing|performing|conducting|carrying\s+out|doing|executing|"
    r"undertaking|delivering)?\s*(?:the\s+)?",
    re.IGNORECASE,
)
_STRIP_DRIVER_TRAIL_RE = re.compile(
    rf"\s+(?:{_COST_HEAD_NOUNS})\s*$", re.IGNORECASE,
)
# Base verbs that mark a phrase as WORK (imperative / action). A structural
# action-verb class -- not a domain vocabulary list.
_WORK_VERB_LEAD_RE = re.compile(
    r"^\s*(?:re-?)?(?:inspect|review|assess|reassess|evaluate|re-?evaluate|"
    r"examine|analy[sz]e|re-?analy[sz]e|audit|check|re-?check|monitor|repair|"
    r"replace|restore|rework|remediate|correct|rectify|fix|update|revise|"
    r"re-?review|modify|amend|adjust|implement|establish|develop|create|"
    r"introduce|deploy|configure|re-?configure|install|build|design|validate|"
    r"revalidate|test|re-?test|verify|reverify|re-?verify|confirm|calibrate|"
    r"recalibrate|qualif\w*|requalif\w*|train|retrain|investigate|determine|"
    r"identify|ascertain|obtain|procure|acquire|perform|re-?perform|execute|"
    r"conduct|carry|complete|finish|prepare|draft|re-?draft|document|record|"
    r"reconstruct|recover|re-?enable|re-?establish|quarantine|segregate|"
    r"reconcile|escalate|notify|strengthen|enhance|improve|upgrade|approve|"
    r"authori[sz]e|assign|schedule|remove|withdraw|revoke|isolate|contain|"
    r"disposition|close\s+out|follow\s+up)\b",
    re.IGNORECASE,
)
# The grammatical HEAD is a deverbal ACTION noun ("... reconstruction",
# "... recalibration", "... impact assessment") -- also WORK, not a resource.
_DEVERBAL_ACTION_HEAD_RE = re.compile(
    r"\b(?:inspection|review|reassessment|assessment|evaluation|re-?evaluation|"
    r"examination|analysis|re-?analysis|audit|check|recheck|repair|replacement|"
    r"restoration|rework|remediation|correction|rectification|update|revision|"
    r"modification|amendment|implementation|establishment|re-?establishment|"
    r"development|deployment|configuration|reconfiguration|installation|"
    r"validation|revalidation|verification|re-?verification|confirmation|"
    r"calibration|recalibration|qualification|requalification|testing|retest|"
    r"training|retraining|investigation|determination|identification|"
    r"reconstruction|recovery|quarantine|segregation|reconciliation|escalation|"
    r"notification|strengthening|enhancement|improvement|upgrade|approval|"
    r"authori[sz]ation|walkthrough|walk-?through|clean-?up|cleanup|"
    r"disposition|close-?out|follow-?up|round|rounds|check-?out)s?\s*$",
    re.IGNORECASE,
)


def looks_like_work(text: str | None) -> bool:
    """True when the phrase positively reads as an ACTION to be performed --
    it leads with an action verb, its head is a deverbal action noun, or it
    is a gerund phrase / the hedged conditional-systemic sentence. Everything
    else (a bare noun phrase naming a resource, service, thing) is NOT work
    and, when it comes from a cost component, must not become an activity."""
    s = (text or "").strip().rstrip(".")
    if not s:
        return False
    if _WORK_VERB_LEAD_RE.match(s):
        return True
    if _DEVERBAL_ACTION_HEAD_RE.search(s):
        return True
    if re.match(r"^\s*(?:re-?)?[a-z]+ing\b", s, re.IGNORECASE):  # "Reconstructing the records"
        return True
    if _ALREADY_HEDGED_RE.search(s) or "determine whether" in s.lower():
        return True  # the canonical conditional-systemic sentence
    return False

_CAUSAL_DIRECTIVE_RE = re.compile(
    r"\b(?:to\s+(?:address|correct|eliminate|fix|remove|remediate|resolve)\s+"
    r"(?:the\s+)?(?:underlying\s+|systemic\s+)?(?:root\s+cause|cause)|"
    r"(?:address|correct|eliminate)\s+(?:the\s+)?root\s+cause)\b",
    re.IGNORECASE,
)
_ALREADY_HEDGED_RE = re.compile(
    r"\b(?:subject\s+to|contingent|potential|once\s+the\s+cause|if\s+confirmed|"
    r"pending|provisional|may\s+(?:be\s+)?requir|assuming)\b",
    re.IGNORECASE,
)

# Verbs of "bring a NEW resource into existence / into service" -- a
# grammatical/semantic class, NOT a domain vocabulary list, disjoint from the
# verbs of an immediate correction (recover / reconstruct / re-verify /
# re-perform / quarantine / document / contain / assess).
_NEW_RESOURCE_VERB_RE = re.compile(
    r"\b(?:install|reinstall|deploy|redeploy|procure|purchase|acquire|buy|"
    r"lease|rent|build|construct|fabricate|erect|introduce|"
    r"roll\s+out|set\s+up|stand\s+up|commission|"
    r"automat(?:e|ing)|digiti[sz]e|computeri[sz]e|"
    r"hire|recruit|onboard|"
    r"re-?train|upskill|"
    r"upgrade|replace|retrofit|redesign|re-?engineer|moderni[sz]e|"
    r"re-?architect|provision)\b"
    # "implement / establish / create / develop / introduce <a NEW artefact>"
    # -- verb + a generic artefact-class noun object = bringing a new resource
    # into being. Excludes "implement the corrective control/change the
    # confirmed cause identifies" (that defers -> _DEFERS_TO_CAUSE_RE wins).
    r"|\b(?:implement|establish|create|develop|introduce|put\s+in\s+place|"
    r"roll\s+out)\s+(?:[a-z]+\s+){0,3}"
    r"(?:system|software|tool|toolset|platform|solution|application|mechanism|"
    r"programme|program|framework|dashboard|tracker|tracking|register|registry|"
    r"database|portal|module|workflow|pipeline|integration|automation|bot|"
    r"scanner|monitor|monitoring|alerting|checklist\s+system)\b",
    re.IGNORECASE,
)
_DEFERS_TO_CAUSE_RE = re.compile(
    r"\b(?:confirmed|established|identified|verified)\s+(?:root\s+)?cause\b|"
    r"\bonce\s+the\s+(?:root\s+)?(?:cause|mechanism|reason)\b|"
    r"\bthe\s+(?:causal\s+)?(?:investigation|assessment|analysis)\s+"
    r"(?:identif|determin|confirm|show|find|indicat|establish)|"
    r"\bwhether\s+.{0,90}?\b(?:is|are|would\s+be|may\s+be)\s+"
    r"(?:required|needed|necessary|warranted|appropriate|justified)\b|"
    r"\bsubject\s+to\s+conf|\bcontingent\s+on\b|\bpending\s+conf",
    re.IGNORECASE,
)

_EFFECTIVENESS_RE = re.compile(
    r"\b(?:effectiveness\s+(?:check|verification|review)|"
    r"verify\s+(?:that\s+)?the\s+(?:completed\s+)?(?:remediation|corrective\s+action|capa|fix)\b|"
    r"confirm\s+(?:that\s+)?the\s+(?:remediation|corrective\s+action|capa)\b[^.]*\beffective|"
    r"verify\s+(?:the\s+)?effectiveness)\b",
    re.IGNORECASE,
)
_CAUSAL_RE = re.compile(
    r"\b(?:investigate\s+(?:why|the\s+cause|the\s+reason)|"
    r"determine\s+(?:the|its)\s+(?:root\s+)?cause|"
    r"root[-\s]cause\s+(?:analysis|investigation)|"
    r"establish\s+(?:the\s+)?(?:underlying\s+)?cause|"
    r"identify\s+(?:the\s+)?(?:root\s+)?cause|"
    r"ascertain\s+(?:the\s+)?cause)\b",
    re.IGNORECASE,
)
_SCOPE_RE = re.compile(
    r"\b(?:determine\s+the\s+(?:extent|scope)|"
    r"assess\s+the\s+(?:full\s+)?(?:extent|scope|impact)|"
    r"extent\s+of\s+the\s+(?:deviation|condition|issue|non-?conformit|impact)|"
    r"identify\s+(?:all\s+)?other\s+[^.]*\baffected|"
    r"how\s+(?:far|widely)\s+the\s+[^.]*\bextend|"
    r"establish\s+(?:the\s+)?population\s+(?:affected|impacted))\b",
    re.IGNORECASE,
)
_SYSTEMIC_RE = re.compile(
    r"\b(?:strengthen|systemic|preventive\s+(?:action|control|measure)|"
    r"prevent\s+(?:its\s+|the\s+)?recurrence|prevent\s+recurrence|"
    r"reduce\s+the\s+(?:likelihood|risk|probability)\s+of\s+recurrence|"
    r"so\s+that\s+[^.]*\b(?:is|are)\s+(?:prevented|reliably\s+detected)|"
    r"address\s+(?:the\s+)?(?:underlying|systemic))\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "at", "by",
    "with", "that", "this", "its", "any", "all", "was", "were", "is", "are",
    "be", "been", "as", "from", "into", "current", "affected", "review",
    "reviewed", "against", "which", "their", "there",
}


def is_unsupported_concrete_intervention(text: str | None) -> bool:
    """A concrete "install / procure / retrain / replace / ..." prescription
    that does NOT defer to the confirmed cause or a completed assessment."""
    s = (text or "").strip()
    if not s:
        return False
    if _DEFERS_TO_CAUSE_RE.search(s) or _ALREADY_HEDGED_RE.search(s):
        return False
    return bool(_NEW_RESOURCE_VERB_RE.search(s))


def hedge_causal_directive(text: str, contingent: bool) -> str:
    if not contingent or not text:
        return text
    if _CAUSAL_DIRECTIVE_RE.search(text) and not _ALREADY_HEDGED_RE.search(text):
        lead = text[0].lower() + text[1:] if not text[:2].isupper() else text
        return f"Subject to confirming the underlying cause, {lead}"
    return text


def classify_activity_kind(description: str | None) -> str:
    s = (description or "").strip()
    if not s:
        return KIND_IMMEDIATE_CORRECTION
    if _EFFECTIVENESS_RE.search(s):
        return KIND_EFFECTIVENESS_VERIFICATION
    if _CAUSAL_RE.search(s):
        return KIND_CAUSAL_INVESTIGATION
    if _SCOPE_RE.search(s):
        return KIND_SCOPE_ASSESSMENT
    if _SYSTEMIC_RE.search(s) or is_unsupported_concrete_intervention(s):
        return KIND_SYSTEMIC_STRENGTHENING
    return KIND_IMMEDIATE_CORRECTION


def is_pricing_driver_phrase(text: str | None) -> bool:
    """True when `text` describes what PRICES work (a cost / resource driver)
    rather than a distinct unit of remediation work. Structural: the
    grammatical head is an economics/resource meta-noun, or the phrase is
    framed as '<cost-noun> for <work>' / '<work> cost'. A phrase that leads
    with an action (work) verb is never a pricing driver."""
    s = (text or "").strip().rstrip(".")
    if not s:
        return False
    if _WORK_VERB_LEAD_RE.match(s):
        return False
    if _PRICING_DRIVER_LEAD_RE.match(s) or _PRICING_OF_RE.match(s):
        return True
    # "<work> cost/effort/labour/fee" -- head is the cost noun, and there is
    # real content before it (not just an article).
    if _PRICING_DRIVER_TRAIL_RE.search(s):
        head_stripped = _STRIP_DRIVER_TRAIL_RE.sub("", s).strip()
        if len(head_stripped.split()) >= 1 and not _WORK_VERB_LEAD_RE.match(head_stripped):
            return True
    return False


def strip_pricing_frame(text: str | None) -> str:
    """Recover the WORK phrase a pricing-driver expression prices, or '' when
    the driver names no work ('Contractor cost', 'Analyst effort')."""
    s = (text or "").strip().rstrip(".")
    if not s:
        return ""
    m = _STRIP_DRIVER_LEAD_RE.match(s)
    if m:
        return s[m.end():].strip(" -:") .strip()
    t = _STRIP_DRIVER_TRAIL_RE.sub("", s).strip(" -:").strip()
    if t and t.lower() != s.lower():
        return t
    return ""


@dataclass
class CanonicalActivity:
    """One remediation activity in the canonical collection. Carries every
    dimension a downstream consumer needs so nothing is re-inferred.
    `conditionality` (does the activity's NECESSITY depend on the unconfirmed
    cause?) and `pricing_status` (can it be defensibly priced yet?) are
    INDEPENDENT axes."""

    id: str
    description: str
    kind: str
    conditionality: str                      # CONFIRMED | CONDITIONAL
    provenance: str                          # LLM | DETERMINISTIC_SCOPE | LLM_COST_COMPONENT
    pricing_status: str = UNPRICED           # PRICED | UNPRICED
    pricing_evidence_required: str = ""      # "" iff PRICED
    component_ids: list[str] = field(default_factory=list)
    is_hypothetical: bool = False

    @property
    def is_conditional(self) -> bool:
        return self.conditionality == CONDITIONAL

    @property
    def is_priced(self) -> bool:
        return self.pricing_status == PRICED

    @property
    def semantic_role(self) -> str:
        return self.kind


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower()).rstrip(".")


def _sig_tokens(s: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9][a-z0-9-]{2,}", (s or "").lower())
        if w not in _STOPWORDS
    }


def _match_by_overlap(
    canon: list[CanonicalActivity], desc: str, threshold: float = 0.6,
) -> CanonicalActivity | None:
    """Attach to an existing activity only on STRONG lexical overlap (one
    significant-token set nearly contains the other) -- never a loose guess."""
    ct = _sig_tokens(desc)
    if len(ct) < 2:
        return None
    best, best_score = None, 0.0
    for a in canon:
        at = _sig_tokens(a.description)
        if not at:
            continue
        score = len(ct & at) / min(len(ct), len(at))
        if score > best_score:
            best, best_score = a, score
    return best if best_score >= threshold else None


@dataclass
class UnresolvedPricingDriver:
    """A cost component the LLM produced that is a PRICING representation of
    work (not distinct work) and could not be attached to any activity. Kept
    for audit -- its full pricing detail already lives in the result's
    `cost_components`; this records the unresolved relationship explicitly so
    it never leaks into `implementation_activities`."""

    description: str
    component_id: str
    pricing_evidence_required: str
    provenance: str = PROV_COST_COMPONENT
    relationship: str = COMP_UNRESOLVED_DRIVER


def _from_scope(scope) -> list[CanonicalActivity]:
    out: list[CanonicalActivity] = []
    seen: set[str] = set()
    for i, sa in enumerate(getattr(scope, "activities", []) or []):
        k = _norm_key(sa.description)
        if not k or k in seen:
            continue
        seen.add(k)
        cond = CONDITIONAL if sa.conditionality == CONDITIONAL else CONFIRMED
        out.append(CanonicalActivity(
            id=f"S{i}",
            description=sa.description,
            kind=sa.kind,
            conditionality=cond,
            provenance=PROV_SCOPE,
            pricing_evidence_required=(
                sa.pricing_evidence_needed or _pricing_need(sa.description, sa.kind, None)
            ),
            is_hypothetical=(cond == CONDITIONAL),
        ))
    return out


def _from_canonical(
    canonical_activities: list, semantic_type: str | None, contingent: bool,
    conditional_systemic_sentence: str,
) -> list[CanonicalActivity]:
    """Base activity list from the VALIDATED canonical `remediation_activities`
    (`SemRemediationAction`). The canonical LLM already made the semantic
    role + conditionality decision -- this translates its `disposition` 1:1
    into the structural KIND and never re-inspects the wording. Epistemic
    safety only: a concrete systemic prescription presented as confirmed while
    the cause is unconfirmed is still forced conditional."""
    out: list[CanonicalActivity] = []
    seen: set[str] = set()
    for i, a in enumerate(canonical_activities or []):
        desc = (getattr(a, "activity", "") or "").strip()
        if not desc:
            continue
        k = _norm_key(desc)
        if not k or k in seen:
            continue
        seen.add(k)
        disp = str(getattr(a, "disposition", "") or "")
        kind = _DISPOSITION_TO_KIND.get(disp, KIND_SYSTEMIC_STRENGTHENING)
        depends = bool(getattr(a, "depends_on_root_cause", False))
        # A DIRECT correction (IMMEDIATE_CORRECTION / CONTAINMENT) of the
        # established condition is NEVER conditionalised by root-cause
        # uncertainty and "replace/restore the damaged X" is the correction --
        # not an "unsupported concrete intervention" (spec §4/§6/§10). The
        # unsupported-intervention guard applies ONLY to systemic activities.
        _is_direct = disp in ("IMMEDIATE_CORRECTION", "CONTAINMENT")
        conditional = (
            disp == "CONDITIONAL_SYSTEMIC"
            or depends
            or (contingent and not _is_direct and kind == KIND_SYSTEMIC_STRENGTHENING)
            or (contingent and not _is_direct and is_unsupported_concrete_intervention(desc))
        )
        if conditional and not _is_direct and kind == KIND_SYSTEMIC_STRENGTHENING and contingent \
                and is_unsupported_concrete_intervention(desc):
            desc = conditional_systemic_sentence or desc
        out.append(CanonicalActivity(
            id=str(getattr(a, "action_id", "") or f"K{i}"),
            description=desc,
            kind=kind,
            conditionality=CONDITIONAL if conditional else CONFIRMED,
            provenance=PROV_CANONICAL,
            pricing_evidence_required=(
                getattr(a, "pricing_evidence_needed", None)
                or _pricing_need(desc, kind, semantic_type)
            ),
            is_hypothetical=conditional,
        ))
    return out


def _from_llm_activities(
    llm_activities: list, semantic_type: str | None, contingent: bool,
    conditional_systemic_sentence: str,
) -> list[CanonicalActivity]:
    out: list[CanonicalActivity] = []
    seen: set[str] = set()
    injected_systemic = False

    def _add(act: CanonicalActivity) -> None:
        k = _norm_key(act.description)
        if not k or k in seen:
            return
        seen.add(k)
        out.append(act)

    for a in llm_activities or []:
        desc = (getattr(a, "description", "") or "").strip()
        if not desc:
            continue
        # The cost LLM's own semantic role for this activity (fallback path).
        # A DIRECT correction of the established condition (IMMEDIATE_CORRECTION /
        # CONTAINMENT) is NEVER conditionalised by root-cause uncertainty and is
        # never treated as an "unsupported concrete intervention" -- restoring a
        # damaged established item IS the correction (spec §2/§3/§5). The
        # unsupported-intervention / systemic-injection path applies only to
        # non-direct activities. Mirrors `_from_canonical` (pass 25).
        _disp = str(getattr(a, "disposition", "") or "")
        _is_direct = _disp in ("IMMEDIATE_CORRECTION", "CONTAINMENT")
        desc = desc if _is_direct else hedge_causal_directive(desc, contingent)
        hyp = bool(getattr(a, "is_hypothetical", False)) or bool(
            getattr(a, "depends_on_root_cause", False)
        )
        # An activity the LLM says it derived from an UNCONFIRMED root-cause
        # hypothesis is, by its own provenance, cause-dependent.
        if contingent and not _is_direct and str(getattr(a, "derived_from", "")) == "ROOT_CAUSE_HYPOTHESIS":
            hyp = True
        aid = str(getattr(a, "activity_id", "") or f"A{len(out)}")

        if contingent and not _is_direct and is_unsupported_concrete_intervention(desc):
            if not injected_systemic:
                _add(CanonicalActivity(
                    id="CSYS",
                    description=conditional_systemic_sentence,
                    kind=KIND_SYSTEMIC_STRENGTHENING,
                    conditionality=CONDITIONAL,
                    provenance=PROV_SCOPE,
                    pricing_evidence_required=_pricing_need(
                        conditional_systemic_sentence, KIND_SYSTEMIC_STRENGTHENING, semantic_type
                    ),
                    is_hypothetical=True,
                ))
                injected_systemic = True
            continue

        kind = _DISPOSITION_TO_KIND.get(_disp) if _disp else None
        if kind is None:
            kind = classify_activity_kind(desc)
        conditional = (
            hyp
            or (not _is_direct and bool(_ALREADY_HEDGED_RE.search(desc)))
            or (contingent and not _is_direct and kind == KIND_SYSTEMIC_STRENGTHENING)
        )
        _add(CanonicalActivity(
            id=aid,
            description=desc,
            kind=kind,
            conditionality=CONDITIONAL if conditional else CONFIRMED,
            provenance=PROV_LLM,
            pricing_evidence_required=_pricing_need(desc, kind, semantic_type),
            is_hypothetical=hyp or conditional,
        ))
    return out


def build_canonical_activities(
    *,
    llm_activities: list,
    interp_components: list,
    validated_component_results: list,
    priced_component_ids: set[str],
    scope,
    semantic_type: str | None,
    contingent: bool,
    use_scope_as_canonical: bool,
    conditional_systemic_sentence: str,
    canonical_activities: list | None = None,
) -> tuple[list[CanonicalActivity], list[UnresolvedPricingDriver]]:
    """The ONE deterministic step. Returns (canonical activities, unresolved
    pricing drivers).

    Single source of truth (spec §8/§22): when `canonical_activities` (the
    VALIDATED `remediation_activities` from the canonical interpretation) is
    supplied, it is the authoritative base -- the second remediation
    interpretation's activities are NOT used to define the work, only its cost
    components attach for pricing. Falls back to the LLM interpreter's
    activities, then the deterministic scope, only when no canonical activity
    list is available (provider unavailable / flag off)."""
    if canonical_activities is not None:
        canon = _from_canonical(
            canonical_activities, semantic_type, contingent, conditional_systemic_sentence
        )
    elif use_scope_as_canonical or (not llm_activities and not validated_component_results):
        canon = _from_scope(scope)
    else:
        canon = _from_llm_activities(
            llm_activities, semantic_type, contingent, conditional_systemic_sentence
        )

    canon, unresolved = _reconcile_components(
        canon,
        interp_components=interp_components,
        validated_component_results=validated_component_results,
        priced_component_ids=priced_component_ids,
        semantic_type=semantic_type,
        contingent=contingent,
        conditional_systemic_sentence=conditional_systemic_sentence,
    )

    # finalise the pricing-evidence requirement: none for a priced activity,
    # a role-derived requirement for an unpriced one.
    for a in canon:
        if a.is_priced:
            a.pricing_evidence_required = ""
        elif not a.pricing_evidence_required:
            a.pricing_evidence_required = _pricing_need(a.description, a.kind, semantic_type)
    return canon, unresolved


def _reconcile_components(
    canon: list[CanonicalActivity],
    *,
    interp_components: list,
    validated_component_results: list,
    priced_component_ids: set[str],
    semantic_type: str | None,
    contingent: bool,
    conditional_systemic_sentence: str,
) -> tuple[list[CanonicalActivity], list[UnresolvedPricingDriver]]:
    by_raw_id = {
        str(getattr(c, "component_id", "")): c for c in (interp_components or [])
    }
    by_canon_id = {a.id: a for a in canon}
    unresolved: list[UnresolvedPricingDriver] = []
    _sys_holder: list[CanonicalActivity | None] = [
        next((a for a in canon if a.kind == KIND_SYSTEMIC_STRENGTHENING and a.is_conditional), None)
    ]

    def _ensure_sys() -> CanonicalActivity:
        if _sys_holder[0] is None:
            sa = CanonicalActivity(
                id="CSYS",
                description=conditional_systemic_sentence,
                kind=KIND_SYSTEMIC_STRENGTHENING,
                conditionality=CONDITIONAL,
                provenance=PROV_SCOPE,
                pricing_evidence_required=_pricing_need(
                    conditional_systemic_sentence, KIND_SYSTEMIC_STRENGTHENING, semantic_type
                ),
                is_hypothetical=True,
            )
            canon.append(sa)
            by_canon_id[sa.id] = sa
            _sys_holder[0] = sa
        return _sys_holder[0]

    def _new_work_activity(desc: str, cid: str) -> CanonicalActivity | None:
        """Genuine independent remediation work -> a canonical activity."""
        desc = hedge_causal_directive(desc, contingent)
        if contingent and is_unsupported_concrete_intervention(desc):
            return _ensure_sys()
        kind = classify_activity_kind(desc)
        cond = contingent and kind == KIND_SYSTEMIC_STRENGTHENING
        na = CanonicalActivity(
            id=cid or f"C{len(canon)}", description=desc, kind=kind,
            conditionality=CONDITIONAL if cond else CONFIRMED,
            provenance=PROV_COST_COMPONENT, is_hypothetical=cond,
        )
        if _norm_key(na.description) in {_norm_key(a.description) for a in canon}:
            return next(a for a in canon if _norm_key(a.description) == _norm_key(na.description))
        canon.append(na)
        by_canon_id[na.id] = na
        return na

    for cr in validated_component_results or []:
        cid = str(getattr(cr, "component_id", "") or "")
        cdesc = (getattr(cr, "description", "") or "").strip()
        raw = by_raw_id.get(cid)
        act_ids = [str(x) for x in (getattr(raw, "activity_ids", []) or [])] if raw else []
        had_explicit_link = bool(act_ids)

        # 3a. explicit activity_ids.
        targets = [by_canon_id[a] for a in act_ids if a in by_canon_id]

        if not targets and cdesc:
            # A component becomes an implementation activity ONLY when it
            # positively reads as WORK. A pricing-driver phrase, or ANY bare
            # noun phrase that does not read as an action (a resource, a
            # service, a thing), is a pricing representation -- it is attached
            # to the work it prices or kept as an explicit unresolved pricing
            # relationship, never a phantom activity (spec RULE 6: ambiguity
            # resolves toward preserving the work/price distinction).
            component_is_work = looks_like_work(cdesc) and not is_pricing_driver_phrase(cdesc)
            _llm_acts = [a for a in canon if a.provenance == PROV_LLM]
            _genuine = [a for a in canon if a.provenance != PROV_COST_COMPONENT]

            if component_is_work:
                m = _match_by_overlap(canon, cdesc) or _new_work_activity(cdesc, cid)
                targets = [m] if m is not None else []
            else:
                # PRICING representation -- resolve to the work it prices.
                work = strip_pricing_frame(cdesc)
                m = (
                    _match_by_overlap(canon, work)
                    or _match_by_overlap(canon, cdesc)
                ) if (work or cdesc) else None
                if m is None and work:
                    m = _match_by_overlap(canon, work, threshold=0.5)
                if m is None and len(_llm_acts) == 1 and not had_explicit_link:
                    m = _llm_acts[0]
                if m is not None:
                    targets = [m]
                elif not _genuine:
                    # degenerate: no genuine activity anywhere -> this component
                    # is the only description of remediation work we have.
                    t = _new_work_activity(cdesc, cid)
                    targets = [t] if t is not None else []
                else:
                    # preserved, auditable, NOT an implementation activity.
                    unresolved.append(UnresolvedPricingDriver(
                        description=cdesc, component_id=cid,
                        pricing_evidence_required=_pricing_need(
                            work or cdesc,
                            classify_activity_kind(work or cdesc),
                            semantic_type,
                        ),
                    ))
                    continue

        for t in targets:
            if cid and t is not None and cid not in t.component_ids:
                t.component_ids.append(cid)

    # An activity's pricing_status: PRICED iff it has >=1 linked component and
    # EVERY linked component was priced. No component, or any unpriced
    # component -> UNPRICED (still visible). Conditionality is untouched --
    # the two axes are independent.
    for a in canon:
        if a.component_ids and all(c in priced_component_ids for c in a.component_ids):
            a.pricing_status = PRICED
        else:
            a.pricing_status = UNPRICED
    return canon, unresolved


# --- derivations: EVERY downstream field comes from these -----------------

def implementation_activities(canon: list[CanonicalActivity]) -> list[str]:
    return [a.description for a in canon]


def conditional_activities(canon: list[CanonicalActivity]) -> list[str]:
    return [a.description for a in canon if a.is_conditional]


def unpriced_activities(canon: list[CanonicalActivity]) -> list[str]:
    return [a.description for a in canon if not a.is_priced]


def evidence_improves_estimate(canon: list[CanonicalActivity]) -> list[str]:
    """The activity-to-evidence invariant, by construction: every item is the
    pricing requirement of a specific unpriced canonical activity."""
    out: list[str] = []
    seen: set[str] = set()
    for a in canon:
        if a.is_priced:
            continue
        e = (a.pricing_evidence_required or "").strip()
        if e and e.lower() not in seen:
            seen.add(e.lower())
            out.append(e)
    return out


# Back-compat: the previous public name.
def evidence_needed_for(canon: list[CanonicalActivity]) -> list[str]:
    return evidence_improves_estimate(canon)


def normalize_remediation_activities(
    *,
    llm_activities: list,
    scope,
    semantic_type: str | None,
    contingent: bool,
    use_scope_as_canonical: bool,
    conditional_systemic_sentence: str,
) -> list[CanonicalActivity]:
    """Back-compat wrapper (no cost components)."""
    canon, _ = build_canonical_activities(
        llm_activities=llm_activities,
        interp_components=[],
        validated_component_results=[],
        priced_component_ids=set(),
        scope=scope,
        semantic_type=semantic_type,
        contingent=contingent,
        use_scope_as_canonical=use_scope_as_canonical,
        conditional_systemic_sentence=conditional_systemic_sentence,
    )
    return canon
