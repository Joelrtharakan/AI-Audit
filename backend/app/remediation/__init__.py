"""Remediation Cost Estimation module.

Answers "what will it approximately cost to correct/prevent this finding?" --
kept strictly separate from `app.financial` ("what financial impact did the
finding cause?"). Generic and semantic: the LLM infers the remediation model
from the finding context; deterministic code only validates structure and
executes arithmetic.
"""

__all__ = [
    "RemediationCostResult",
    "RemediationCostComponentResult",
    "CostBasis",
    "RemediationEstimateStatus",
    "RemediationConfidence",
    "estimate_remediation_cost",
    "honest_not_assessable",
]

_MODEL_EXPORTS = frozenset(
    {
        "RemediationCostResult",
        "RemediationCostComponentResult",
        "CostBasis",
        "RemediationEstimateStatus",
        "RemediationConfidence",
    }
)


def __getattr__(name: str):
    # Lazy (PEP 562): importing the canonical models must never force the
    # interpreter/engine (and their LLM-client imports) to load.
    if name in _MODEL_EXPORTS:
        from app.remediation import models

        return getattr(models, name)
    if name in ("estimate_remediation_cost", "honest_not_assessable"):
        from app.remediation import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
