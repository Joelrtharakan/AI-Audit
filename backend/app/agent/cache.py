"""In-memory analysis cache keyed by a deterministic hash of finding context.

The key includes model + prompt_version so that swapping models or editing a
prompt template can never silently serve a stale result generated under a
different configuration. DEGRADED-mode results are never written here at all
(see investigate.py) -- a transient provider failure must not permanently
poison the cache for a finding that a healthy call would analyze normally.
"""

import hashlib
from typing import Any, Optional

_CACHE: dict[str, dict[str, Any]] = {}


def compute_cache_key(
    finding_text: str,
    department: str = "",
    standard: str = "",
    model: str = "",
    prompt_version: str = "",
) -> str:
    payload = (
        f"{finding_text.strip().lower()}|{department.strip().lower()}|{standard.strip().lower()}"
        f"|{model.strip().lower()}|{prompt_version.strip().lower()}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cached_analysis(cache_key: str) -> Optional[dict[str, Any]]:
    return _CACHE.get(cache_key)


def set_cached_analysis(cache_key: str, data: dict[str, Any]) -> None:
    _CACHE[cache_key] = data
