"""Request-scoped 'last LLM call' metadata.

A single ContextVar the LLM boundary writes after every call and that
`core_synthesis` (and anything else) reads to populate the report's
`provider_used` / `fallback_used` / `provider_attempts` / token telemetry.

`app.services.ollama_client.get_last_call_metadata` and
`app.services.llm_router.get_last_call_metadata` now delegate here so existing
call sites keep working unchanged.
"""

from __future__ import annotations

import contextvars
from typing import Any

_last_call_metadata: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "llm_last_call_metadata", default={}
)


def set_last_call_metadata(meta: dict[str, Any]) -> None:
    _last_call_metadata.set(dict(meta))


def get_last_call_metadata() -> dict[str, Any]:
    return _last_call_metadata.get()
