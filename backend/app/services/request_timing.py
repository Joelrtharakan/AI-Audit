"""Lightweight request-scoped timing and LLM-call accounting.

One dict lives in a ContextVar for the duration of an investigation request.
Mutating the dict (never rebinding the variable) keeps it shared across the
tasks LangGraph spawns. Everything here is best-effort and never raises: it is
observability only and has no effect on the analysis.

Records: total request duration, LLM invocation count, LLM generation time,
and per-stage wall time (graph nodes, prompt construction, validation).
"""

from __future__ import annotations

import contextvars
import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

_current: contextvars.ContextVar[dict | None] = contextvars.ContextVar("request_timing", default=None)


def begin() -> None:
    _current.set({"t0": time.monotonic(), "llm_calls": [], "stages": {}})


def record_llm_call(
    node: str | None, elapsed_ms: int, success: bool = True,
    prompt_tokens: int | None = None, output_tokens: int | None = None,
) -> None:
    try:
        st = _current.get()
        if st is not None:
            st["llm_calls"].append({
                "node": node or "?", "ms": int(elapsed_ms), "ok": bool(success),
                "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
            })
    except Exception:  # noqa: BLE001 - observability must never affect the request
        pass


def record_stage(name: str, elapsed_ms: int | None) -> None:
    """Add an already-measured duration to a named stage."""
    try:
        st = _current.get()
        if st is not None and elapsed_ms is not None:
            st["stages"][name] = st["stages"].get(name, 0) + int(elapsed_ms)
    except Exception:  # noqa: BLE001
        pass


@contextmanager
def stage(name: str) -> Iterator[None]:
    t = time.monotonic()
    try:
        yield
    finally:
        try:
            st = _current.get()
            if st is not None:
                st["stages"][name] = st["stages"].get(name, 0) + int((time.monotonic() - t) * 1000)
        except Exception:  # noqa: BLE001
            pass


def summary() -> dict[str, Any]:
    st = _current.get()
    if st is None:
        return {}
    calls = st["llm_calls"]
    return {
        "total_ms": int((time.monotonic() - st["t0"]) * 1000),
        "llm_invocations": len(calls),
        "llm_generation_ms": sum(c["ms"] for c in calls),
        "llm_calls": calls,
        "stages_ms": dict(st["stages"]),
    }


def log_summary(logger: logging.Logger) -> None:
    s = summary()
    if s:
        logger.info(
            "REQUEST TIMING total_ms=%s llm_invocations=%s llm_generation_ms=%s llm_calls(node,ms,ok,prompt_tok,out_tok)=%s stages_ms=%s",
            s["total_ms"], s["llm_invocations"], s["llm_generation_ms"],
            [(c["node"], c["ms"], c["ok"], c.get("prompt_tokens"), c.get("output_tokens")) for c in s["llm_calls"]],
            s["stages_ms"],
        )
