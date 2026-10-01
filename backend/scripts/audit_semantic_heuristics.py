"""Phase 9.9 -- inventory of regex / vocabulary heuristics in production code.

Scans `app/` with the AST for:
  * module-level `NAME = re.compile(...)` constants        (kind=regex)
  * module-level set/frozenset/tuple/list constants with >= MIN_VOCAB string
    literals                                               (kind=vocab)
  * inline `re.search/match/findall/sub/finditer(<literal>, ...)` calls inside
    functions                                              (kind=inline-regex)

For each hit it records file, name, line and the number of tests that
reference the constant name. Classification (A-G) is NOT guessed from content:
it comes from the explicit `CLASSIFICATION` table in
`docs/PHASE_9_9_SEMANTIC_HEURISTIC_INVENTORY.md` (module-level), so a human
decision is always recorded. Run with `--json` for machine output.

Usage: python scripts/audit_semantic_heuristics.py [--json]
"""
from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
APP = BACKEND / "app"
TESTS = BACKEND / "tests"
MIN_VOCAB = 4
_RE_FUNCS = {"search", "match", "findall", "finditer", "sub", "fullmatch", "split"}


def _is_re_call(node: ast.AST, names: set[str]) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "re"
            and node.func.attr in names)


def _string_literals(node: ast.AST) -> int:
    return sum(1 for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str))


def scan_file(path: Path) -> list[dict]:
    rel = path.relative_to(BACKEND).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    hits: list[dict] = []
    for node in tree.body:
        targets, value = [], None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if value is None or not targets or not isinstance(targets[0], ast.Name):
            continue
        name = targets[0].id
        if _is_re_call(value, {"compile"}):
            hits.append({"file": rel, "name": name, "line": node.lineno, "kind": "regex"})
        elif isinstance(value, (ast.Set, ast.Tuple, ast.List)) or (
            isinstance(value, ast.Call) and getattr(value.func, "id", "") in ("frozenset", "set", "tuple")
        ):
            if _string_literals(value) >= MIN_VOCAB and not name.startswith("__"):
                hits.append({"file": rel, "name": name, "line": node.lineno, "kind": "vocab"})
    inline = defaultdict(int)
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(fn):
                if _is_re_call(n, _RE_FUNCS) and n.args and isinstance(n.args[0], ast.Constant):
                    inline[fn.name] += 1
    for fname, count in inline.items():
        hits.append({"file": rel, "name": f"{fname}()", "line": 0, "kind": "inline-regex", "count": count})
    return hits


def test_references(names: set[str]) -> dict[str, int]:
    corpus = [p.read_text(encoding="utf-8", errors="ignore") for p in TESTS.rglob("*.py")]
    return {n: sum(1 for c in corpus if n.rstrip("()") in c) for n in names}


def main() -> None:
    hits: list[dict] = []
    for p in sorted(APP.rglob("*.py")):
        hits.extend(scan_file(p))
    refs = test_references({h["name"] for h in hits if h["kind"] != "inline-regex"})
    for h in hits:
        h["test_files_referencing"] = refs.get(h["name"], "n/a")
    if "--json" in sys.argv:
        print(json.dumps(hits, indent=1))
        return
    by_file: dict[str, list[dict]] = defaultdict(list)
    for h in hits:
        by_file[h["file"]].append(h)
    print(f"{len(hits)} heuristic sites in {len(by_file)} files")
    for f, hs in sorted(by_file.items(), key=lambda kv: -len(kv[1])):
        regex = sum(1 for h in hs if h["kind"] == "regex")
        vocab = sum(1 for h in hs if h["kind"] == "vocab")
        inl = sum(h.get("count", 0) for h in hs if h["kind"] == "inline-regex")
        print(f"{len(hs):4d}  regex={regex:3d} vocab={vocab:3d} inline={inl:3d}  {f}")


if __name__ == "__main__":
    main()
