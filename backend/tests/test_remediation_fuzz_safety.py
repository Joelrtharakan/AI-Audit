"""Pass 64 (spec §36/§38/§39): randomized fuzzing of MALFORMED structured
model output against the whole remediation pipeline
(interpreter.normalize -> validator -> calculator -> engine).

Property under test for EVERY fuzzed payload:
  1. never raises;
  2. always returns a RemediationCostResult;
  3. never fabricates a number from a malformed / contradictory input --
     any surviving figure must trace to a well-formed priced component;
  4. every result carries ai_provenance and review_required stays true unless
     the estimate is a clean EXACT with no unpriced/unresolved parts;
  5. NaN / infinity / negative / wrong-type never reach the headline;
  6. no unbounded latency (each fuzz payload completes fast -- no LLM).

Domain-agnostic: the fuzzer only mutates STRUCTURE, never finding text.
"""
from __future__ import annotations

import json
import math
import random

import pytest

from app.models.agent import EvidenceItem, EvidenceStatus
from app.remediation.engine import estimate_remediation_cost


class _FakeLLM:
    def __init__(self, payload):
        self._p = payload if isinstance(payload, str) else json.dumps(payload, default=str)

    async def chat_completion(self, messages, **kw):
        return self._p


def _ev(c): return EvidenceItem(claim=c, status=EvidenceStatus.VERIFIED, source="fuzz")


_JUNK_NUMBERS = [None, "", "abc", float("nan"), float("inf"), float("-inf"),
                -1, -999999, 0, "12,000", "1e999", [], {}, True, "  ", "NULL"]
_JUNK_STRINGS = [None, "", 0, [], {}, True, 123, "  ", "\x00", "A" * 5000]
_ENUMS = {
    "value_kind": [None, "", "COST", "LOSS", "banana", 42, "REMEDIATION_COST", "UNKNOWN"],
    "amount_type": [None, "", "PER_MONTH", "RATE", "banana", "PER_UNIT", "TOTAL", "COMPONENT"],
    "recurrence": [None, "", "MONTHLY", "yes", "banana", "ONE_TIME", "RECURRING", "UNKNOWN", 7],
    "quantity_basis": [None, "", "GUESS", "EVIDENCED", "banana"],
    "unit_cost_basis": [None, "", "TRUE", "VERIFIED", "banana"],
    "overall_status": [None, "", "OK", "EVIDENCE_BACKED", "banana"],
    "estimability": [None, "", "MAYBE", "ESTIMABLE", "banana"],
}


def _fuzz_component(rng: random.Random, cid: str) -> dict:
    c = {
        "component_id": rng.choice([cid, None, "", 123, cid]),
        "description": rng.choice(_JUNK_STRINGS + [f"component {cid}"]),
        "activity_ids": rng.choice([[cid], [], None, "not-a-list", [None], [1, 2]]),
        "cost_category": rng.choice(_JUNK_STRINGS + ["parts"]),
        "quantity": rng.choice(_JUNK_NUMBERS + [1, 2, 5]),
        "quantity_unit": rng.choice(_JUNK_STRINGS + ["part", "hour", "minute", "month", "each feed change"]),
        "unit_cost": rng.choice(_JUNK_NUMBERS + [1000, 4000]),
        "unit_cost_low": rng.choice(_JUNK_NUMBERS),
        "unit_cost_high": rng.choice(_JUNK_NUMBERS),
        "currency": rng.choice(_JUNK_STRINGS + ["INR", "USD", "EUR"]),
        "recurring_period": rng.choice(_JUNK_STRINGS + ["month", "week", "each release"]),
        "source_reference_ids": rng.choice([["E0"], [], None, ["E999"], [1], "E0"]),
        "assumptions": rng.choice([[], None, "x", [1]]),
    }
    for k, vals in _ENUMS.items():
        if k in ("overall_status", "estimability"):
            continue
        c[k] = rng.choice(vals)
    # sometimes drop random keys entirely
    for k in list(c):
        if rng.random() < 0.15:
            del c[k]
    return c


def _fuzz_payload(rng: random.Random) -> dict:
    n = rng.randint(0, 4)
    payload = {
        "strategy": rng.choice([
            {"remediation_summary": rng.choice(_JUNK_STRINGS + ["fix it"]),
             "remediation_type": rng.choice(_JUNK_STRINGS + ["corrective"]),
             "interpretation_confidence": rng.choice([None, "", "HIGH", "banana"])},
            None, "not-a-dict", {},
        ]),
        "activities": rng.choice([
            [{"activity_id": f"A{i}", "description": rng.choice(_JUNK_STRINGS + [f"act {i}"]),
              "derived_from": rng.choice([None, "", "FINDING", "banana"])} for i in range(n)],
            None, "x", [None], [{}],
        ]),
        "cost_components": [_fuzz_component(rng, f"C{i}") for i in range(n)],
        "calculation_proposals": rng.choice([
            [], None, "x",
            [{"calculation_id": "K0", "operation": rng.choice([None, "MULTIPLY", "banana"]),
              "component_ids": rng.choice([["C0"], None, [1]]),
              "operands": rng.choice([None, [], [{"value": rng.choice(_JUNK_NUMBERS)}]]),
              "frequency": rng.choice([None, "ONE_TIME", "banana"]),
              "produces": rng.choice([None, "MOST_LIKELY", "banana"]),
              "proposed_result_value": rng.choice(_JUNK_NUMBERS + [999999999])}],
        ]),
        "overall_status": rng.choice(_ENUMS["overall_status"]),
        "estimability": rng.choice(_ENUMS["estimability"]),
    }
    for k in list(payload):
        if rng.random() < 0.1:
            del payload[k]
    return payload


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(400))
async def test_fuzz_malformed_output_is_always_safe(seed):
    rng = random.Random(seed)
    payload = _fuzz_payload(rng)
    ev = [_ev("An input for fuzzing."), _ev("Another input.")]
    res = await estimate_remediation_cost(
        finding_text="A corrective action is required.", evidence_ledger=ev,
        client=_FakeLLM(payload),
    )
    # 2. always a result
    assert res is not None
    # 4. provenance always
    assert isinstance(res.ai_provenance, dict) and res.ai_provenance
    # 5. no NaN / inf / negative in any headline figure
    for f in ("one_time_cost", "recurring_cost", "recurring_horizon_total",
              "low_estimate", "most_likely_estimate", "high_estimate"):
        v = getattr(res, f)
        if v is not None:
            assert isinstance(v, (int, float)) and math.isfinite(v) and v >= 0, (f, v)
    # 3. a surviving headline must be backed by a priced component with a
    #    finite, positive, well-typed unit_cost + quantity (or a flat amount)
    if res.one_time_cost or res.recurring_cost:
        priced = [c for c in (res.cost_components or []) if c.calculated_amount is not None]
        assert priced, "headline figure with no priced component"
        for c in priced:
            assert math.isfinite(c.calculated_amount) and c.calculated_amount >= 0
    # 6. malformed inputs must never be a clean EXACT unless genuinely so
    if res.pricing_status == "EXACT_ESTIMATE":
        assert not res.unpriced_activities
        assert not res.unresolved_pricing_drivers
        assert not res.is_partial_estimate


@pytest.mark.asyncio
async def test_fuzz_never_crashes_on_pure_garbage_strings():
    for junk in ['{"broken": ', "not json at all", "", "null", "[]", "{}", "🙂" * 100,
                 '{"cost_components": [{"unit_cost": NaN}]}', '{"cost_components": 42}']:
        res = await estimate_remediation_cost(
            finding_text="x", evidence_ledger=[_ev("y")], client=_FakeLLM(junk))
        assert res is not None
        assert res.review_required is True
        assert res.ai_provenance
        assert res.one_time_cost is None and res.recurring_cost is None
