"""Evidence-ID registry (Phase 9.9) -- provenance bookkeeping only.

BOUNDARY (category C: provenance). Evidence has two namespaces:

  * internal handle  -- `E{index}` into the evidence ledger (what the prompts
    show the model and what validators check), plus the reserved `FINDING`;
  * stated label     -- the label the evidence ITSELF carries in its text
    ("C2: ...", "R1) ...") -- the identity an auditor knows it by.

This module never interprets meaning. It only (1) resolves a model-emitted
reference to the internal handle and (2) maps an internal handle back to the
evidence's own stated label for display. Rules that keep provenance honest:

  1. A DIRECT stated label (on the ledger item's own claim) always wins. A
     positional guess can never claim a label that any item states directly.
  2. A label appearing on two different items is AMBIGUOUS and resolves to
     nothing (fail closed) -- it is never silently pointed at one of them.
  3. Labels found in the finding text align to ledger items by position ONLY
     when there are no direct labels and the counts match one-to-one.
  4. A bare 1-based positional alias (`C<n>` / `<n>`) is a last resort, used only
     when NO item states a label, and only for labels no other rule claims.
  5. Nothing is ever invented: an unresolvable reference resolves to None.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_LABEL_PREFIX_RE = re.compile(r"^\s*([A-Za-z]{1,4}\s?\d{1,3})\s*[:.)\]\-]")
_LABEL_IN_TEXT_RE = re.compile(r"\b([A-Za-z]{1,3}\s?\d{1,3})\s*[:.)\]\-]")
_INTERNAL_RE = re.compile(r"^(?:E\d+|FINDING)$")
_FINDING_ALIASES = ("FINDING", "FINDINGTEXT", "THEFINDING", "F0", "F1", "FIND")


def _norm(label: str) -> str:
    return str(label or "").replace(" ", "").upper()


@dataclass
class EvidenceIdRegistry:
    label_to_internal: dict[str, str] = field(default_factory=dict)
    internal_to_label: dict[str, str] = field(default_factory=dict)
    ambiguous: set[str] = field(default_factory=set)
    internal_ids: set[str] = field(default_factory=set)

    def resolve(self, ref: str | None) -> str | None:
        """Model-emitted reference -> internal handle, or None (never guessed)."""
        r = _norm(ref)
        if not r:
            return None
        if _INTERNAL_RE.match(r):
            return r if (r == "FINDING" or r in self.internal_ids) else None
        if r in self.ambiguous:
            return None
        if r in self.label_to_internal:
            return self.label_to_internal[r]
        if r in _FINDING_ALIASES:
            return "FINDING"
        return None

    def display(self, internal_id: str) -> str:
        """Internal handle -> the evidence's own label (or the handle itself)."""
        return self.internal_to_label.get(internal_id, internal_id)


def build_registry(evidence_ledger: list, finding_text: str = "") -> EvidenceIdRegistry:
    reg = EvidenceIdRegistry()
    n = len(evidence_ledger or [])
    reg.internal_ids = {f"E{i}" for i in range(n)}

    direct: dict[str, list[str]] = {}
    for i, item in enumerate(evidence_ledger or []):
        claim = getattr(item, "claim", None) or getattr(item, "text", "") or ""
        m = _LABEL_PREFIX_RE.match(str(claim))
        if m:
            direct.setdefault(_norm(m.group(1)), []).append(f"E{i}")
    for lab, ids in direct.items():
        if len(ids) == 1:
            reg.label_to_internal[lab] = ids[0]
            reg.internal_to_label[ids[0]] = lab
        else:
            reg.ambiguous.add(lab)

    if not direct:
        text_labels: list[str] = []
        for lab in _LABEL_IN_TEXT_RE.findall(str(finding_text or "")):
            u = _norm(lab)
            if u not in text_labels:
                text_labels.append(u)
        if text_labels and len(text_labels) == n:
            for i, lab in enumerate(text_labels):
                reg.label_to_internal[lab] = f"E{i}"
                reg.internal_to_label[f"E{i}"] = lab

    # last resort positional aliases: only when NO item states a label (so the
    # namespace is not explicit), and only for labels nothing else has claimed.
    for i in range(0 if direct else n):
        eid = f"E{i}"
        for alias in (f"C{i + 1}", str(i), str(i + 1)):
            if alias not in reg.label_to_internal and alias not in reg.ambiguous:
                reg.label_to_internal[alias] = eid
    return reg
