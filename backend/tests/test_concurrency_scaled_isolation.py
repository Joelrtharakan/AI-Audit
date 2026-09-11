"""Spec §15: scaled concurrent state-isolation -- 50 heterogeneous findings
through the remediation engine concurrently, each with a UNIQUE marker in its
structured input; assert no marker from finding A appears in finding B's
result, provenance, or components. Offline (fake LLM per request).
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.models.agent import EvidenceItem, EvidenceStatus
from app.remediation.engine import estimate_remediation_cost


class _FakeLLM:
    def __init__(self, p): self._p = json.dumps(p)
    async def chat_completion(self, m, **k):
        await asyncio.sleep(0)  # yield -> real interleaving
        return self._p


def _payload(marker: str, qty: int, rate: int):
    return {
        "strategy": {"remediation_summary": f"fix {marker}", "remediation_type": "corrective",
                     "interpretation_confidence": "HIGH"},
        "activities": [{"activity_id": "A0", "description": f"Correct {marker}", "derived_from": "FINDING"}],
        "cost_components": [{
            "component_id": "C0", "description": f"{marker} part", "activity_ids": ["A0"],
            "cost_category": "parts", "value_kind": "REMEDIATION_COST",
            "quantity": qty, "quantity_unit": "unit", "quantity_basis": "EVIDENCED",
            "unit_cost": rate, "unit_cost_basis": "VERIFIED", "currency": "INR",
            "amount_type": "PER_UNIT", "recurrence": "ONE_TIME", "source_reference_ids": ["E0"],
        }],
        "calculation_proposals": [], "overall_status": "EVIDENCE_BACKED", "estimability": "ESTIMABLE",
    }


@pytest.mark.asyncio
async def test_50_concurrent_findings_no_cross_contamination():
    N = 50
    markers = [f"MARKER_{i:03d}_XYZ" for i in range(N)]
    qtys = [i % 7 + 1 for i in range(N)]
    rates = [1000 + i * 111 for i in range(N)]

    async def one(i):
        ev = [EvidenceItem(claim=f"{markers[i]} needs correction", status=EvidenceStatus.VERIFIED, source="t")]
        return i, await estimate_remediation_cost(
            finding_text=f"A finding about {markers[i]}.", evidence_ledger=ev,
            client=_FakeLLM(_payload(markers[i], qtys[i], rates[i])))

    results = dict(await asyncio.gather(*(one(i) for i in range(N))))
    for i, rc in results.items():
        want = qtys[i] * rates[i]
        assert rc.one_time_cost == want, (i, rc.one_time_cost, want)
        blob = json.dumps(rc.model_dump(), default=str)
        # this finding's own marker is fine; no OTHER finding's marker may appear
        for j, m in enumerate(markers):
            if j != i:
                assert m not in blob, f"finding {i} leaked marker of finding {j}"
        assert rc.ai_provenance  # provenance present per request
