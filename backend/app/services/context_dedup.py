"""Structural de-duplication of the finding text against the evidence ledger.

The evidence ledger is built from the finding's own sentences, so sending both
the full finding AND every ledger item sends the same text twice. This helper
answers one STRUCTURAL question -- "is the finding text exactly the union of the
ledger items?" -- by verbatim containment and character coverage. It reads no
meaning, matches no vocabulary, and defaults to False (keep both) whenever
anything is excluded, reformatted, clipped or uncertain, so the safe behavior
(full finding AND ledger) is the fallback.
"""

from __future__ import annotations

from typing import Any, Iterable


def _norm(s: Any) -> str:
    return " ".join(str(s or "").split())


def ledger_covers_finding(
    finding_text: str, claims: Iterable[Any], max_claim_chars: int | None = None
) -> bool:
    """True only when every ledger claim is a verbatim part of the finding text,
    none would be clipped by the prompt, and together the claims account for
    (virtually) all of the finding's non-space characters -- i.e. nothing in the
    finding (an excluded instruction, a reformatted attribution, ...) is missing
    from the ledger."""
    ft = _norm(finding_text)
    cl = [_norm(c) for c in claims]
    cl = [c for c in cl if c]
    if not ft or not cl:
        return False
    if any(c not in ft for c in cl):
        return False
    if max_claim_chars is not None and any(len(c) > max_claim_chars for c in cl):
        return False
    covered = sum(len(c.replace(" ", "")) for c in dict.fromkeys(cl))
    total = len(ft.replace(" ", ""))
    return total > 0 and covered >= 0.98 * total
