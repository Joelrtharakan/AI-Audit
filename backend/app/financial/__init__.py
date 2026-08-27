"""Evidence-Grounded Financial Exposure & Cost-of-Recurrence Analysis module."""

__all__ = [
    "AnnualizedExposure",
    "CapaEconomicAnalysis",
    "ConfirmedFinancialImpact",
    "CostOfQualityBreakdown",
    "FinancialAmountType",
    "FinancialAnalysisResult",
    "FinancialConfidenceLevel",
    "FinancialEpistemicStatus",
    "FinancialObservation",
    "FinancialScenarioAnalysis",
    "FinancialUncertainty",
    "PotentialFinancialExposure",
    "RecurrenceAnalysis",
    "ScenarioEstimate",
    "analyze_financial_exposure",
]

_MODELS_EXPORTS = frozenset(__all__) - {"analyze_financial_exposure"}


def __getattr__(name: str):
    # Lazy (PEP 562) so that `from app.financial.models import X` -- which Python must
    # execute this __init__ for first -- never forces `engine`/`extractor` to load (they
    # import from `app.models.agent`, which imports `FinancialAnalysisResult` from here,
    # a circular import if these were eager module-level imports).
    if name == "analyze_financial_exposure":
        from app.financial.engine import analyze_financial_exposure

        return analyze_financial_exposure
    if name in _MODELS_EXPORTS:
        from app.financial import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
