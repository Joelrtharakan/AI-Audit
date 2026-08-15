"""In-memory analysis cache keyed by a deterministic hash of finding context."""

import hashlib
import json
from typing import Any, Optional

_CACHE: dict[str, dict[str, Any]] = {}

def compute_cache_key(finding_text: str, department: str = "", standard: str = "") -> str:
    payload = f"{finding_text.strip().lower()}|{department.strip().lower()}|{standard.strip().lower()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def get_cached_analysis(cache_key: str) -> Optional[dict[str, Any]]:
    return _CACHE.get(cache_key)

def set_cached_analysis(cache_key: str, data: dict[str, Any]) -> None:
    _CACHE[cache_key] = data
