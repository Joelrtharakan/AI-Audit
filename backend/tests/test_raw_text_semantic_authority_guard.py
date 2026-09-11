"""Phase 10 -- CI guard: deterministic code must not grow NEW semantic-authority
classifiers driven by raw finding text.

The architecture rule (memory: llm-primary-semantic-architecture,
semantic-subject-gate): the LLM owns semantic interpretation. Deterministic
code may validate structure/arithmetic/provenance and may keyword-match in the
*fail-closed deterministic fallback* paths (locked by ~400 tests, out of scope
here) -- but a NEW keyword/regex classifier on `finding_text` in production is
an architectural regression.

This guard is a BASELINE SNAPSHOT, not a naive grep: it records every existing
`<literal> in finding_text` / `re.<x>(<literal>, finding_text)` /
`finding_text.lower()` site and fails only when the set GROWS. Reducing the set
is encouraged (update the baseline down).

Sanctioned, never-flagged: prompt construction, logging, serialization, tests,
and the structural subject gate (semantic_subject.py).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_APP = _BACKEND / "app"
_BASELINE_FILE = Path(__file__).with_name("_raw_text_semantic_authority_baseline.txt")

# Files where raw-text inspection is architecturally sanctioned.
_ALLOWLIST = {
    "app/services/semantic_subject.py",       # the structural/grammatical subject gate
    "app/services/prompt_builder.py",          # prompt construction
}

_TEXT_VARS = r"(?:finding_text|raw_finding|request_finding_text)"
# keyword-classifier shapes
_PATTERNS = [
    re.compile(rf'["\'][^"\']+["\']\s+in\s+{_TEXT_VARS}\b'),
    re.compile(rf'{_TEXT_VARS}\.lower\(\)'),
    re.compile(rf're\.(?:search|match|findall|finditer)\s*\([^,]+,\s*{_TEXT_VARS}\b'),
    re.compile(rf'\bin\s+{_TEXT_VARS}\.lower\(\)'),
]


def _scan() -> set[str]:
    hits: set[str] = set()
    for path in sorted(_APP.rglob("*.py")):
        rel = path.relative_to(_BACKEND).as_posix()
        if rel in _ALLOWLIST:
            continue
        for i, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.split("#", 1)[0]
            if any(p.search(line) for p in _PATTERNS):
                # token is file + a normalized snippet (not line number, so
                # unrelated edits don't churn the baseline)
                snippet = re.sub(r"\s+", " ", raw_line.strip())[:120]
                hits.add(f"{rel} :: {snippet}")
    return hits


def _load_baseline() -> set[str]:
    if not _BASELINE_FILE.exists():
        return set()
    return {
        ln.strip() for ln in _BASELINE_FILE.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    }


def test_no_new_raw_text_semantic_authority_classifiers():
    current = _scan()
    baseline = _load_baseline()
    added = current - baseline
    assert not added, (
        "New raw-finding-text keyword/regex classifier(s) introduced in production "
        "code. The LLM owns semantic interpretation. If this is genuinely in a "
        "fail-closed deterministic fallback, add it to "
        f"{_BASELINE_FILE.name} with a justifying comment:\n  "
        + "\n  ".join(sorted(added))
    )


def test_baseline_has_no_stale_entries():
    """Entries removed from the code should be pruned from the baseline."""
    stale = _load_baseline() - _scan()
    assert not stale, (
        "Baseline lists patterns no longer in the code -- prune them:\n  "
        + "\n  ".join(sorted(stale))
    )


# --------------------------------------------------------------------------- #
# detector self-tests
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("line, flagged", [
    ('if "monthly" in finding_text:', True),
    ("if re.search(r'\\bhospital\\b', finding_text):", True),
    ("text_low = finding_text.lower()", True),
    ('if any(w in finding_text.lower() for w in KEYWORDS):', True),
    # legitimate, must NOT flag:
    ('messages = build_messages(finding_text=finding_text)', False),
    ('prompt = template.format(finding_text=finding_text)', False),
    ('logger.info("analyzing %s", finding_text[:80])', False),
    ('return resolve_deviation(finding_text)', False),
    ('state["request"].finding_text', False),
    ('for token in finding_text.split():', False),
])
def test_detector_precision(line, flagged):
    assert any(p.search(line.split("#", 1)[0]) for p in _PATTERNS) is flagged
