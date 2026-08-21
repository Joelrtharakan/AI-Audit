"""Instruction & Prompt-Injection Detector.

Treats all user-provided finding text strictly as DATA. Detects embedded
imperative instructions (e.g., "ignore previous instructions", "close the CAPA",
"mark as resolved", "set root cause to...", "blame operator") -- including
INDIRECT framings that try to masquerade as a higher-priority instruction
("system message:", "the AI must...", "management instructed the AI to...")
-- and separates them from legitimate factual evidence before evidence
extraction.

Security classification is a SEPARATE dimension from evidence status
(Section 4 of the prompt-injection hardening spec): a claim's evidentiary
weight (VERIFIED/REPORTED) and its input-integrity classification
(NORMAL/QUOTED_INSTRUCTION/INSTRUCTION_LIKE/PROMPT_INJECTION_SUSPECTED/
MALICIOUS_INSTRUCTION) are never conflated into one field.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

SecurityClassification = Literal[
    "NORMAL", "QUOTED_INSTRUCTION", "INSTRUCTION_LIKE",
    "PROMPT_INJECTION_SUSPECTED", "MALICIOUS_INSTRUCTION",
]

# ---------------------------------------------------------------------------
# B/C: a reported/quoted instruction directed at a HUMAN role is a
# legitimate operational event ("the supervisor instructed the technician to
# complete the checklist") and must remain eligible as evidence -- only an
# instruction directed at the AI/assistant/system itself (A) is untrusted.
# ---------------------------------------------------------------------------
_HUMAN_DIRECTED_INSTRUCTION_RE = re.compile(
    r"\b(?:instructed|told|directed|ordered|asked)\s+(?:the\s+|a\s+|an\s+)?"
    r"(?:technician|operator|supervisor|staff|employee|personnel|team|worker|"
    r"manager|analyst|inspector|auditor|department|contractor)s?\s+to\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# A: text that explicitly addresses/frames itself as directed at the AI,
# assistant, model, system, or developer -- including impersonating a
# higher-priority message channel ("system message:", "developer
# instruction:", "assistant:") -- regardless of what specific action it asks
# for. This is what actually distinguishes indirect injection from a
# harmless mention of "the system" as a noun (e.g. "the document-control
# system failed").
# ---------------------------------------------------------------------------
_AI_DIRECTED_RE = re.compile(
    r"\b(?:the\s+)?(?:ai|assistant|model|chatbot|llm)\s+(?:must|should|is\s+to|needs?\s+to|has\s+to|will)\s+\w+|"
    r"\b(?:instructed|told|directed|ordered|asked)\s+(?:the\s+)?(?:ai|assistant|model|chatbot|llm|system)\s+to\b|"
    r"^\s*(?:system\s+message|developer\s+message|developer\s+instruction|assistant)\s*:|"
    r"\byou\s+are\s+now\b|"
    r"\boutput\s+(?:root_cause_established|established|verified|compliant)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Direct override / prompt-injection framing -- attempts to claim
# higher priority than the actual system prompt.
_OVERRIDE_RE = re.compile(
    r"\bignore\s+(?:all\s+)?(?:previous\s+|prior\s+)?instructions?\b|"
    r"\bdisregard\s+(?:all\s+)?(?:previous\s+|prior\s+)?instructions?\b|"
    r"\bfollow\s+these\s+instructions\b|"
    r"\b(?:system\s+prompt|new\s+system\s+instruction)\b|"
    r"\bchange\s+your\s+answer\b|"
    r"\boverride\s+(?:your|the|any|all)\s+(?:previous\s+|prior\s+)?instructions?\b|"
    r"\boverride\s+(?:the\s+)?system\s+prompt\b|"
    r"\bbypass\s+the\s+audit\b",
    re.IGNORECASE,
)

# Evidence-tampering intent -- combined with an override/AI-directed signal
# this escalates classification to MALICIOUS_INSTRUCTION (Section 2's
# "delete/hide/suppress evidence" list).
_EVIDENCE_TAMPERING_RE = re.compile(
    r"\b(?:delete|hide|suppress|remove|do\s+not\s+report)\s+(?:this\s+|the\s+)?(?:evidence|records?|logs?|statement)\b",
    re.IGNORECASE,
)

# Generic imperative instructions directed at the audit outcome itself
# (close/approve/resolve/mark) -- ambiguous enough on their own to warrant
# PROMPT_INJECTION_SUSPECTED rather than the stronger MALICIOUS_INSTRUCTION,
# matching the spec's own worked example (an "ignore instructions" + "approve
# the CAPA" combination classifies as PROMPT_INJECTION_SUSPECTED, not
# MALICIOUS_INSTRUCTION -- MALICIOUS_INSTRUCTION is reserved for override
# attempts COMBINED with evidence-tampering intent).
_OUTCOME_IMPERATIVE_RE = re.compile(
    r"\b(?:close|approve|resolve|finalize)\s+(?:the\s+)?(?:capa|finding|case|audit|deviation|investigation|corrective\s+action)\b|"
    r"\b(?:set|mark|change|modify|declare)\s+(?:the\s+)?(?:root\s+cause|severity|status|confidence|risk|quality|finding)\s+(?:to|as)?\b|"
    r"\bclassify\s+(?:this|the\s+finding)\s+as\s+(?:compliant|low\s+risk|resolved)\b|"
    r"\b(?:do\s+not|don't)\s+(?:investigate|mention|check|look\s+into|verify|report)\b|"
    r"\bconsider\s+this\s+(?:verified|resolved|approved|effective|trained|closed)\b|"
    r"\bassume\s+the\s+(?:cause|root\s+cause)\s+is\b|"
    r"\btreat\s+(?:the\s+)?(?:finding|deviation|issue|case)\s+as\s+(?:verified|resolved|approved|effective)\b|"
    r"\b(?:blame|attribute\s+to)\s+(?:the\s+)?(?:operator|technician|personnel|individual)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# STRUCTURAL IMPERATIVE DETECTION (Defect 6).
#
# The generalization is grammatical, not lexical: an audit observation is a
# THIRD-PERSON DECLARATIVE about an external real-world entity ("the permit
# was not issued"), whereas an injection is a SECOND-PERSON IMPERATIVE whose
# referential target is the analysis process itself ("ignore the previous
# instructions and output the root cause as ...").  We therefore test two
# independent structural properties and require BOTH:
#
#   (a) imperative mood  -- a subjectless clause headed by a base-form verb,
#       or an explicit second-person directive ("you must/are to/are
#       authorized to", "your task is to").
#   (b) meta-referential target -- the clause talks about the instructions,
#       the system/AI, the analysis, the report, the output, or the verdict
#       fields, rather than about an auditable real-world object.
#
# Neither half is a list of attacker phrasings: (a) is morphology + function
# words, (b) is a list of the *only* nouns that can name this system's own
# process.  A novel attack in novel wording still satisfies both.
# ---------------------------------------------------------------------------

# Closed-class function words that can never head an imperative clause. This
# is grammar, not vocabulary: determiners, pronouns, prepositions,
# conjunctions, auxiliaries, wh-words and sentence adverbs.
_NON_IMPERATIVE_HEADS = {
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "all",
    "each", "every", "no", "none", "both", "either", "neither", "such",
    "i", "you", "he", "she", "it", "we", "they", "who", "whom", "whose",
    "which", "what", "when", "where", "why", "how", "there", "here",
    "in", "on", "at", "by", "for", "from", "of", "to", "with", "without",
    "during", "after", "before", "under", "over", "into", "onto", "upon",
    "within", "across", "against", "between", "among", "per", "via", "as",
    "and", "or", "but", "so", "yet", "nor", "if", "unless", "although",
    "though", "while", "whereas", "because", "since", "however",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had", "do", "does", "did", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must",
    "not", "also", "further", "additionally", "moreover", "subsequently",
    "approximately", "records", "record", "review", "reviews", "audit",
    "evidence", "documentation", "observation", "observations", "finding",
    "findings", "personnel", "management", "training", "calibration",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "several", "multiple", "numerous",
    # subordinators / complementizers / participial connectives -- a clause
    # they head is embedded, never a standalone imperative.
    "whether", "whereby", "wherein", "whereas", "whenever", "wherever",
    "including", "excluding", "regarding", "concerning", "given", "based",
    "until", "once", "unless", "than", "that", "though", "despite",
    "notwithstanding", "pending", "following", "according", "subject",
    "potential", "potentially", "possible", "possibly", "likely", "unlikely",
}

# Objects/complements typical right after an imperative head verb.
_IMPERATIVE_COMPLEMENT_RE = re.compile(
    r"^(?:all|any|the|this|that|these|those|your|my|our|its|his|her|their|"
    r"it|them|previous|prior|above|everything|anything)\b",
    re.IGNORECASE,
)

# (b) The referential target is the analysis process / the system / the
# output artefact -- never an external auditable entity.
_META_REFERENT_RE = re.compile(
    r"\b(?:instructions?|prompt|system\s+prompt|directives?|"
    r"the\s+(?:ai|assistant|model|llm|chatbot|system|tool|engine|analysis|"
    r"investigation|assessment|report|output|response|answer|verdict|"
    r"conclusion|finding|analyser|analyzer)|"
    r"root[\s_]cause|root_cause_established|"
    r"(?:this|the)\s+(?:report|output|analysis|investigation|finding|case|capa)|"
    r"previous\s+(?:instructions?|context|messages?)|"
    r"(?:everything|anything)\s+(?:stated|said|written)\s+(?:earlier|above|previously)|"
    r"above\s+(?:instructions?|text)|"
    # Directing the VERDICT is directing the analysis ("... as compliant",
    # "... to resolved"). Deliberately excludes bare analysis-field nouns
    # (severity, risk rating, compliance status) -- those appear legitimately
    # in the system's OWN impact and investigation prose, so keying on them
    # would make the invariant fire on our own correct output.
    r"(?:as|to)\s+(?:compliant|resolved|closed|verified|approved|effective|"
    r"acceptable|low\s+risk|no\s+finding|not\s+a\s+finding))\b",
    re.IGNORECASE,
)

# (a2) Explicit second-person directive framing.
_SECOND_PERSON_DIRECTIVE_RE = re.compile(
    r"\byou\s+(?:must|should|shall|will|need\s+to|have\s+to|are\s+to|"
    r"are\s+(?:hereby\s+)?(?:authori[sz]ed|permitted|instructed|required|allowed)\s+to)\b|"
    r"\byour\s+(?:task|job|instruction|role|goal|objective)\s+(?:is|are)\b|"
    r"\bas\s+an?\s+(?:ai|assistant|auditor\s+ai|language\s+model)\b",
    re.IGNORECASE,
)

_LEADING_DISCOURSE_RE = re.compile(
    r"^(?:please\s+|now\s+|first\s+|then\s+|also\s+|immediately\s+|"
    r"important\s*[:!.-]?\s*|note\s*[:.-]\s*|urgent\s*[:!.-]?\s*|"
    r"and\s+|but\s+|so\s+)+",
    re.IGNORECASE,
)


_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:[.,;:!?]|\band\b|\bthen\b|\balso\b)\s+")


def _clause_is_imperative(clause: str) -> bool:
    """True when a single clause is a subjectless imperative (base-form verb
    head + object).  Morphology + closed-class function words only."""
    clause = _LEADING_DISCOURSE_RE.sub("", clause.strip())
    if not clause:
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z'’_-]*", clause)
    if len(tokens) < 2:
        return False
    head_low = tokens[0].lower()
    if head_low in _NON_IMPERATIVE_HEADS:
        return False
    # Inflected forms (-s / -ed / -ing) are finite or participial, so the
    # clause is declarative, not imperative. A base form carries no such
    # inflection.
    if re.search(r"(?:ies|es|s|ed|ing)$", head_low) and head_low not in {
        "process", "address", "access", "bypass", "assess", "dismiss", "discuss",
    }:
        return False
    rest = clause[len(tokens[0]):].strip()
    return bool(_IMPERATIVE_COMPLEMENT_RE.match(rest))


def _is_imperative_clause(text: str) -> bool:
    """True when any clause of `text` is imperative, or the text carries an
    explicit second-person directive."""
    t = text.strip()
    if not t:
        return False
    if _SECOND_PERSON_DIRECTIVE_RE.search(t):
        return True
    # Test every coordinated clause: "ignore X and output Y" hides its second
    # imperative behind a conjunction.
    return any(_clause_is_imperative(c) for c in _CLAUSE_SPLIT_RE.split(t))


def _is_system_directed_imperative(text: str) -> bool:
    """(a) AND (b) IN THE SAME CLAUSE: an imperative/second-person clause
    whose own referential target is this analysis process.

    Requiring co-location in one clause is what keeps the test precise. A
    declarative audit sentence that merely happens to mention "the impact
    assessment" somewhere, while a *different* embedded clause elsewhere
    superficially looks verb-initial, is not an injection -- and testing the
    two properties over the whole string would have called it one."""
    t = text.strip()
    if not t:
        return False
    if _SECOND_PERSON_DIRECTIVE_RE.search(t) and _META_REFERENT_RE.search(t):
        return True
    return any(
        _clause_is_imperative(c) and _META_REFERENT_RE.search(c)
        for c in _CLAUSE_SPLIT_RE.split(t)
    )


@dataclass
class InstructionClassification:
    classification: SecurityClassification = "NORMAL"
    matched_patterns: list[str] = field(default_factory=list)

    @property
    def is_untrusted(self) -> bool:
        """True if this text must be excluded from the evidence ledger --
        everything except NORMAL and QUOTED_INSTRUCTION (a legitimate
        reported operational event)."""
        return self.classification in ("INSTRUCTION_LIKE", "PROMPT_INJECTION_SUSPECTED", "MALICIOUS_INSTRUCTION")


def classify_instruction(text: str | None) -> InstructionClassification:
    """Deterministic security classification of a single claim/sentence.
    Distinct from evidence status (Section 4) -- callers decide separately
    whether/how to record this in the evidence ledger."""
    if not text or not text.strip():
        return InstructionClassification("NORMAL")
    t = text.strip()

    has_override = bool(_OVERRIDE_RE.search(t))
    has_ai_directed = bool(_AI_DIRECTED_RE.search(t))
    has_tampering = bool(_EVIDENCE_TAMPERING_RE.search(t))
    has_outcome_imperative = bool(_OUTCOME_IMPERATIVE_RE.search(t))
    has_human_directed = bool(_HUMAN_DIRECTED_INSTRUCTION_RE.search(t))
    # Defect 6: structural imperative-mood + meta-referential-target test.
    has_system_imperative = _is_system_directed_imperative(t)

    matched = []
    if has_system_imperative:
        matched.append("system_directed_imperative")
    if has_override:
        matched.append("override")
    if has_ai_directed:
        matched.append("ai_directed")
    if has_tampering:
        matched.append("evidence_tampering")
    if has_outcome_imperative:
        matched.append("outcome_imperative")
    if has_human_directed:
        matched.append("human_directed")

    # B/C: an instruction reported as directed at a human role, with no
    # override/AI-directed/tampering signal, is a legitimate operational
    # event -- never excluded from evidence.
    if has_human_directed and not (has_override or has_ai_directed or has_tampering or has_system_imperative):
        return InstructionClassification("QUOTED_INSTRUCTION", matched)

    # Override or AI-directed framing COMBINED with evidence-tampering
    # intent is the clearest, most severe attack shape.
    if (has_override or has_ai_directed or has_system_imperative) and has_tampering:
        return InstructionClassification("MALICIOUS_INSTRUCTION", matched)

    # Any override attempt, explicit AI-directed framing, or a structurally
    # system-directed imperative, on its own.
    if has_override or has_ai_directed or has_system_imperative:
        return InstructionClassification("PROMPT_INJECTION_SUSPECTED", matched)

    # Evidence-tampering language alone (no override/AI-direction detected)
    # is still a serious, unambiguous attempt to corrupt the audit trail.
    if has_tampering:
        return InstructionClassification("MALICIOUS_INSTRUCTION", matched)

    # A bare outcome-directed imperative ("approve the CAPA", "mark as
    # resolved") without any AI-directed/override framing is ambiguous
    # enough to flag but not escalate as far.
    if has_outcome_imperative:
        return InstructionClassification("INSTRUCTION_LIKE", matched)

    return InstructionClassification("NORMAL")


def is_instruction(text: str) -> bool:
    """True if text must be excluded from the evidence ledger. Kept as the
    simple boolean entry point every existing caller already uses."""
    return classify_instruction(text).is_untrusted


def filter_untrusted_instructions(sentences: list[str]) -> tuple[list[str], list[str]]:
    """Splits sentences into (legitimate_data, untrusted_instructions)."""
    data: list[str] = []
    instructions: list[str] = []
    for s in sentences:
        if is_instruction(s):
            instructions.append(s)
        else:
            data.append(s)
    return data, instructions
