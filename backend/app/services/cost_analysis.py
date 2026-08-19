"""Deterministic Cost & Financial Impact Analysis Service (Sections 26-42 & Hardening).

Enforces:
  - Strict conditional activation (INV-COST-01): cost_factor_detected is True
    ONLY when verified/reported evidence contains an explicit or semantic cost signal.
  - Zero financial hallucinations (INV-COST-02): never invents amounts or rates.
  - Financial Concept Distinction: Transaction amounts and duplicate payments
    are NOT converted into actual losses without evidence.
  - Evidence provenance (INV-COST-03): traces amounts to source claim IDs.
  - Deterministic arithmetic (INV-COST-04, INV-COST-08): computes formulas in Python, never in LLM prompts.
  - Clear actual vs potential exposure vs estimated distinction (INV-COST-05).
  - Absolute separation between financial exposure and root cause (INV-COST-06).
  - Uncalculable costs gracefully resolved as REQUIRES_ASSESSMENT (INV-COST-07).
  - Zero secondary cost inventions (Rule 8): no fabricated investigation/remediation costs.
"""

from __future__ import annotations

import re
from typing import Any

from app.models.agent import (
    CostComponent,
    CostEvidenceStatus,
    CostFactorType,
    CostImpact,
    EvidenceClaim,
    EvidenceItem,
    EvidenceStatus,
    FinancialAmount,
)


# ---------------------------------------------------------------------------
# 1. Financial Signal Detection Patterns
# ---------------------------------------------------------------------------

_CURRENCY_SYMBOLS_RE = re.compile(
    r"(?:\$|€|£|₹|Rs\.?|INR|USD|EUR|GBP|AUD|CAD)\s*\d+(?:,\d+)*(?:\.\d+)?|\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:INR|USD|EUR|GBP|rupees?|dollars?|euros?)\b",
    re.IGNORECASE,
)

_EXPLICIT_COST_TERMS_RE = re.compile(
    r"\b(?:cost|costs|expense|expenses|financial\s+loss|monetary\s+loss|monetary\s+value|"
    r"budget|rework\s+cost|scrap\s+cost|replacement\s+cost|repair\s+cost|overtime\s+cost|"
    r"downtime\s+cost|production\s+loss|revenue\s+loss|penalty|penalties|fine|fines|"
    r"compensation|refund|refunds|credit|credits|disposal\s+cost|external\s+service\s+cost|"
    r"supplier\s+charge|operational\s+expense|cost\s+of\s+nonconformance|cost\s+impact|"
    r"financial\s+impact|duplicate\s+payments?|overpayment|overpayments?|paid\s+twice|"
    r"double\s+payment|purchase\s+value|payment\s+amount|transaction\s+amount)\b",
    re.IGNORECASE,
)

_SEMANTIC_FINANCIAL_CONSEQUENCE_RE = re.compile(
    r"\b(?:"
    r"rework(?:ed|ing)?|"
    r"scrap(?:ped|ping)?|"
    r"replacement\s+materials?|"
    r"replace\s+the\s+batch|"
    r"additional\s+(?:labor|operators|materials?|shifts?|overtime)|"
    r"wasted\s+materials?|"
    r"unavailable\s+for\s+\d+\s*(?:hours|hrs|days|shifts)|"
    r"downtime\s+of\s+\d+\s*(?:hours|hrs|days|shifts)|"
    r"production\s+was\s+delayed|"
    r"delayed\s+production|"
    r"production\s+halted|"
    r"line\s+stoppage|"
    r"incurred\s+additional|"
    r"batch\s+had\s+to\s+be\s+scrapped|"
    r"batch\s+was\s+scrapped|"
    r"duplicate\s+payment|"
    r"paid\s+twice|"
    r"overpayment"
    r")\b",
    re.IGNORECASE,
)

_ESTIMATE_MARKERS_RE = re.compile(
    r"\b(?:estimate|estimates|estimated|approximate|approximately|potential|projected|expected|around)\b",
    re.IGNORECASE,
)

_LABOR_HOURS_RE = re.compile(
    r"\b(?P<hours>\d+(?:\.\d+)?)\s*(?:rework\s+hours?|labor\s+hours?|hours?\s+of\s+rework|hours?\s+of\s+labor|hours?)\b",
    re.IGNORECASE,
)

_HOURLY_RATE_RE = re.compile(
    r"(?P<curr>₹|\$|€|£|Rs\.?|INR|USD|EUR)?\s*(?P<rate>\d+(?:,\d+)*(?:\.\d+)?)\s*(?:INR|USD|EUR)?\s*/\s*(?:hour|hr)\b",
    re.IGNORECASE,
)

_IRRECOVERABLE_RE = re.compile(
    r"\b(?:irrecoverab(?:ly|le)|irreversib(?:ly|le)|written\s+off|write[- ]off|unrecoverable)\b",
    re.IGNORECASE,
)

_REFUND_RE = re.compile(
    r"\b(?:refund(?:ed|s)?|recover(?:ed|y)?|credit(?:ed)?|revers(?:ed|al)?|returned)\b",
    re.IGNORECASE,
)


def has_cost_signals(text: str) -> bool:
    """True if text contains explicit monetary amounts, cost terms, or semantic consequence markers."""
    if not text:
        return False
    return bool(
        _CURRENCY_SYMBOLS_RE.search(text)
        or _EXPLICIT_COST_TERMS_RE.search(text)
        or _SEMANTIC_FINANCIAL_CONSEQUENCE_RE.search(text)
    )


# ---------------------------------------------------------------------------
# 2. Factor Type & Driver Classification
# ---------------------------------------------------------------------------

def classify_cost_factor_type(text: str) -> str:
    """Classify the primary financial factor category."""
    t = text.lower()
    if "duplicate payment" in t or "paid twice" in t or "double payment" in t:
        return CostFactorType.DUPLICATE_PAYMENT.value
    if "overpayment" in t or "overpaid" in t:
        return CostFactorType.OVERPAYMENT.value
    if "rework" in t:
        return CostFactorType.REWORK.value
    if "scrap" in t or "scrapped" in t:
        return CostFactorType.SCRAP.value
    if "downtime" in t or "unavailable for" in t or "production was delayed" in t or "delayed production" in t or "line stoppage" in t:
        return CostFactorType.DOWNTIME.value
    if "replacement" in t or "replace" in t:
        return CostFactorType.REPLACEMENT.value
    if "penalty" in t or "fine" in t:
        return CostFactorType.PENALTY.value if "penalty" in t else CostFactorType.FINE.value
    if "compensation" in t or "refund" in t:
        return CostFactorType.REFUND.value
    if "overtime" in t:
        return CostFactorType.OVERTIME.value
    if "labor" in t or "operator" in t:
        return CostFactorType.LABOR.value
    if "material" in t:
        return CostFactorType.MATERIAL.value
    if "production loss" in t or "revenue loss" in t:
        return CostFactorType.REVENUE_LOSS.value
    return CostFactorType.OTHER.value


def detect_cost_drivers(text: str) -> list[str]:
    """Identify supported cost drivers from finding/evidence text."""
    drivers = []
    t = text.lower()
    if "duplicate payment" in t or "paid twice" in t:
        return []  # No invented secondary drivers for duplicate payment
    if "overpayment" in t:
        return []
    if "rework" in t:
        drivers.append("Rework labor and reprocessing")
    if "scrap" in t or "scrapped" in t:
        drivers.append("Scrap and batch loss")
    if "replacement" in t or "replace" in t:
        drivers.append("Replacement materials")
    if any(w in t for w in ("downtime", "unavailable for", "delayed production", "production was delayed", "line stoppage")):
        drivers.append("Equipment downtime and delayed production")
    if "additional operator" in t or "additional labor" in t or "overtime" in t:
        drivers.append("Additional personnel / overtime labor")
    if "material" in t and "Replacement materials" not in drivers and "Scrap and batch loss" not in drivers:
        drivers.append("Material consumption / waste")
    if any(w in t for w in ("penalty", "fine", "compensation", "refund")):
        drivers.append("Regulatory penalty / customer compensation")
    if "disposal" in t:
        drivers.append("Disposal / hazardous waste handling")
    return drivers


def get_default_missing_evidence(factor_type: str, text: str) -> tuple[list[str], list[str]]:
    """Return (missing_inputs, evidence_required) tailored to the factor type."""
    t = text.lower()
    missing = []
    evidence_req = []

    if factor_type == CostFactorType.DUPLICATE_PAYMENT.value or "duplicate" in t:
        missing.extend(["Payment reversal/credit note status", "Accounts-payable reconciliation verification"])
        evidence_req.extend([
            "Payment reversal/recovery records",
            "Supplier credit note",
            "Bank and payment transaction records",
            "Accounts-payable reconciliation records",
        ])
    elif "rework" in t:
        missing.extend(["Documented rework labor hours", "Applicable hourly labor rate", "Rework material consumption"])
        evidence_req.extend(["Signed rework batch records", "Shop-floor labor logs", "Material requisition records"])
    elif "scrap" in t or "scrapped" in t:
        missing.extend(["Batch standard manufacturing cost", "Raw material inventory value", "Scrap disposal fees"])
        evidence_req.extend(["Bill of materials costing records", "Inventory write-off authorization", "Disposal manifest"])
    elif "downtime" in t or "unavailable" in t:
        missing.extend(["Verified downtime duration", "Standard hourly production line output value", "Overtime recovery cost"])
        evidence_req.extend(["Equipment maintenance/downtime log", "Production scheduling records", "OEE operational reports"])
    else:
        missing.extend(["Detailed itemized expense breakdown", "Applicable accounting cost standards"])
        evidence_req.extend(["Invoices, timecards, inventory transfer slips, and accounting ledger records"])

    return missing, evidence_req


# ---------------------------------------------------------------------------
# 3. Value & Currency Parsing
# ---------------------------------------------------------------------------

def parse_currency(text: str) -> str:
    """Detect the relevant currency."""
    if "₹" in text or re.search(r"\b(?:INR|rupees?|Rs\.?)\b", text, re.IGNORECASE):
        return "INR"
    if "€" in text or re.search(r"\b(?:EUR|euros?)\b", text, re.IGNORECASE):
        return "EUR"
    if "£" in text or re.search(r"\b(?:GBP|pounds?)\b", text, re.IGNORECASE):
        return "GBP"
    if "$" in text or re.search(r"\b(?:USD|dollars?)\b", text, re.IGNORECASE):
        return "USD"
    return "INR"


def extract_explicit_amounts(text: str) -> list[tuple[float, str]]:
    """Extract (amount, currency) tuples from text, supporting Indian numbering (e.g. 1,25,000)."""
    results = []
    for m in re.finditer(
        r"(?P<prefix>₹|\$|€|£|Rs\.?|INR|USD|EUR)?\s*(?P<num>\d+(?:,\d+)*(?:\.\d+)?)\s*(?P<suffix>INR|USD|EUR|GBP|rupees?|dollars?|euros?)?",
        text,
        re.IGNORECASE,
    ):
        raw_num = m.group("num").replace(",", "")
        prefix = m.group("prefix") or ""
        suffix = m.group("suffix") or ""
        if not prefix and not suffix:
            continue
        try:
            val = float(raw_num)
            curr = parse_currency(f"{prefix} {suffix}")
            results.append((val, curr))
        except ValueError:
            continue
    return results


def format_currency_amount(amount: float, curr: str) -> str:
    """Format currency with standard symbol/code and commas."""
    sym = "₹" if curr == "INR" else ("$" if curr == "USD" else ("€" if curr == "EUR" else ("£" if curr == "GBP" else f"{curr} ")))
    if amount.is_integer():
        return f"{sym}{int(amount):,}"
    return f"{sym}{amount:,.2f}"


# ---------------------------------------------------------------------------
# 4. Deterministic Calculation Engine
# ---------------------------------------------------------------------------

def try_calculate_deterministic_cost(text: str) -> tuple[float | None, list[CostComponent], str | None, list[str]]:
    """Attempt deterministic calculation if discrete formula inputs (e.g. hours * rate + material) exist."""
    components: list[CostComponent] = []
    curr = parse_currency(text)

    # 1. Check for Labor Hours * Rate
    m_hours = _LABOR_HOURS_RE.search(text)
    m_rate = _HOURLY_RATE_RE.search(text)

    labor_total = None
    labor_basis = None
    if m_hours and m_rate:
        try:
            hours = float(m_hours.group("hours"))
            rate = float(m_rate.group("rate").replace(",", ""))
            labor_total = round(hours * rate, 2)
            labor_curr = parse_currency(m_rate.group(0))
            curr = labor_curr
            sym = "₹" if curr == "INR" else ("$" if curr == "USD" else f"{curr} ")
            labor_basis = f"{int(hours) if hours.is_integer() else hours} rework hours × {sym}{int(rate) if rate.is_integer() else rate}/hour"
            components.append(CostComponent(
                name="Labor rework",
                amount=labor_total,
                currency=labor_curr,
                category="LABOR",
                basis=labor_basis,
                provenance="CALCULATED",
            ))
        except (ValueError, IndexError):
            pass

    # 2. Check for discrete material amount mentioned in addition to labor
    m_mat = re.search(r"(?:material\s+(?:cost\s+)?(?:of|=|\:|\s+was)?\s*)?(?P<curr>₹|\$|€|£|Rs\.?|INR|USD|EUR)?\s*(?P<amt>\d+(?:,\d+)*(?:\.\d+)?)\s*(?P<currsuf>INR|USD|EUR)?\s*(?:in\s+)?material", text, re.IGNORECASE)
    mat_total = None
    if m_mat:
        try:
            amt = float(m_mat.group("amt").replace(",", ""))
            mat_curr = parse_currency(m_mat.group(0))
            mat_total = amt
            sym = "₹" if mat_curr == "INR" else ("$" if mat_curr == "USD" else f"{mat_curr} ")
            components.append(CostComponent(
                name="Material",
                amount=mat_total,
                currency=mat_curr,
                category="MATERIAL",
                basis=f"{sym}{int(amt) if amt.is_integer() else amt} material cost",
                provenance="REPORTED",
            ))
        except (ValueError, IndexError):
            pass

    if labor_total is not None:
        total = labor_total + (mat_total or 0.0)
        sym = "₹" if curr == "INR" else ("$" if curr == "USD" else f"{curr} ")
        basis_parts = [labor_basis]
        if mat_total is not None:
            basis_parts.append(f"{sym}{int(mat_total) if mat_total.is_integer() else mat_total} material")
        full_basis = " + ".join(b for b in basis_parts if b)
        assumptions = [f"Labor rate of {sym}{int(m_rate.group('rate').replace(',', '')) if m_rate else 'standard'}/hour is applicable."]
        return total, components, full_basis, assumptions

    return None, components, None, []


# ---------------------------------------------------------------------------
# 5. Core Detection & Analysis Function
# ---------------------------------------------------------------------------

def analyze_cost_and_financial_impact(
    finding_text: str,
    evidence_ledger: list[Any] | None = None,
    evidence_claims: list[EvidenceClaim] | None = None,
) -> CostImpact | None:
    """Analyze finding and evidence for financial impact.
    
    If no cost factor is detected -> returns None (omits section entirely).
    Distinguishes transaction amounts (duplicate payments) from actual losses.
    """
    if not finding_text and not evidence_ledger and not evidence_claims:
        return None

    # Combine all finding and evidence text
    all_texts = [finding_text]
    claim_id_map = {}
    verified_claim_texts = []
    reported_claim_texts = []

    if evidence_claims:
        for c in evidence_claims:
            all_texts.append(c.text)
            claim_id_map[c.text] = c.claim_id
            if c.status == EvidenceStatus.VERIFIED:
                verified_claim_texts.append(c.text)
            else:
                reported_claim_texts.append(c.text)
    elif evidence_ledger:
        for idx, e in enumerate(evidence_ledger, start=1):
            claim_text = getattr(e, "claim", str(e))
            all_texts.append(claim_text)
            cid = f"C{idx}"
            claim_id_map[claim_text] = cid
            if getattr(e, "status", None) == EvidenceStatus.VERIFIED:
                verified_claim_texts.append(claim_text)
            else:
                reported_claim_texts.append(claim_text)

    combined_text = " ".join(all_texts)

    # 1. Deterministic Signal Check
    if not has_cost_signals(combined_text):
        return None

    factor_type = classify_cost_factor_type(combined_text)
    drivers = detect_cost_drivers(combined_text)
    curr = parse_currency(combined_text)

    # Evidence IDs citing cost
    matched_cids = [
        cid for txt, cid in claim_id_map.items() if has_cost_signals(txt)
    ]
    if not matched_cids:
        matched_cids = ["C1"]

    # 2. Check for Deterministic Formula Calculation (e.g. Labor Hours * Rate + Material)
    calc_total, calc_comps, calc_basis, calc_assumptions = try_calculate_deterministic_cost(combined_text)
    if calc_total is not None:
        missing_inputs, evidence_req = get_default_missing_evidence(factor_type, combined_text)
        exposure_str = format_currency_amount(calc_total, curr)
        fin_amt = FinancialAmount(
            amount=calc_total,
            formatted=exposure_str,
            currency=curr or "INR",
            factor=factor_type,
            source_claim_ids=matched_cids,
            support_status="ESTIMATED",
            confidence="MEDIUM",
        )
        return CostImpact(
            cost_factor_detected=True,
            cost_factor_type=factor_type,
            financial_factor=factor_type,
            financial_status=CostEvidenceStatus.ESTIMATED.value,
            currency=curr,
            financial_amount=fin_amt,
            verified_cost=None,
            reported_cost=None,
            estimated_cost=calc_total,
            potential_exposure=calc_total,
            potential_cost_exposure=exposure_str,
            actual_loss=None,
            actual_loss_status="NOT_ESTABLISHED",
            cost_components=calc_comps,
            cost_drivers=drivers,
            calculation_basis=calc_basis,
            assumptions=calc_assumptions,
            missing_cost_inputs=missing_inputs,
            evidence_required=evidence_req,
            evidence_ids=matched_cids,
            confidence="MEDIUM",
            narrative=f"The finding is associated with an estimated financial exposure of approximately {exposure_str}, calculated from available labor and material inputs.",
        )

    # 3. Check for Explicit Stated Amounts
    explicit_amounts = extract_explicit_amounts(combined_text)
    if explicit_amounts:
        amount, parsed_curr = explicit_amounts[0]
        curr = parsed_curr
        formatted_amount_str = format_currency_amount(amount, curr)

        # -------------------------------------------------------------------
        # 3a. Specialized Duplicate Payment / Transaction Logic (Hardening)
        # -------------------------------------------------------------------
        if factor_type in (CostFactorType.DUPLICATE_PAYMENT.value, CostFactorType.OVERPAYMENT.value) or re.search(r"\b(?:duplicate\s+(?:supplier\s+|vendor\s+|invoice\s+)?payments?|paid\s+twice|double\s+payments?)\b", combined_text, re.IGNORECASE):
            factor_type = CostFactorType.DUPLICATE_PAYMENT.value
            missing_inputs, evidence_req = get_default_missing_evidence(factor_type, combined_text)
            
            # Check for refund / recovery in text
            refund_matches = []
            for m in re.finditer(r"(?:refund(?:ed|s)?|recover(?:ed|y)?|credit(?:ed)?|revers(?:ed|al)?|returned)\s*(?:of\s*)?(?P<prefix>₹|\$|€|£|Rs\.?|INR|USD|EUR)?\s*(?P<amt>\d+(?:,\d+)*(?:\.\d+)?)", combined_text, re.IGNORECASE):
                try:
                    refund_matches.append(float(m.group("amt").replace(",", "")))
                except ValueError:
                    pass

            is_irrecoverable = bool(_IRRECOVERABLE_RE.search(combined_text))
            fin_amt = FinancialAmount(
                amount=amount,
                formatted=formatted_amount_str,
                currency=curr or "INR",
                factor=factor_type,
                source_claim_ids=matched_cids,
                support_status="VERIFIED",
                confidence="HIGH",
            )

            if is_irrecoverable:
                # Confirmed actual loss
                return CostImpact(
                    cost_factor_detected=True,
                    cost_factor_type=factor_type,
                    financial_factor=factor_type,
                    financial_status="VERIFIED_LOSS",
                    currency=curr,
                    financial_amount=fin_amt,
                    transaction_amount=amount,
                    gross_exposure=amount,
                    outstanding_amount=amount,
                    net_exposure=amount,
                    potential_exposure=amount,
                    potential_cost_exposure=formatted_amount_str,
                    actual_loss=amount,
                    actual_loss_status="VERIFIED",
                    recoverable_amount=0.0,
                    recovered_amount=0.0,
                    unrecovered_amount=amount,
                    recoverability="IRRECOVERABLE",
                    recoverability_status="IRRECOVERABLE",
                    amount_confidence="HIGH",
                    classification_confidence="HIGH",
                    recovery_confidence="HIGH",
                    actual_loss_confidence="HIGH",
                    cost_components=[],
                    cost_drivers=[],
                    calculation_basis=f"Confirmed irrecoverable loss: {formatted_amount_str}",
                    assumptions=[],
                    missing_cost_inputs=[],
                    evidence_required=["Dispute resolution log", "Write-off authorization documentation"],
                    evidence_ids=matched_cids,
                    confidence="HIGH",
                    narrative=f"Duplicate payment of {formatted_amount_str} was identified and confirmed as irrecoverably lost.",
                )
            elif refund_matches:
                recovered_amt = refund_matches[0]
                rec_formatted = format_currency_amount(recovered_amt, curr)
                if recovered_amt >= amount:
                    # Full recovery
                    return CostImpact(
                        cost_factor_detected=True,
                        cost_factor_type=factor_type,
                        financial_factor=factor_type,
                        financial_status="RECOVERED",
                        currency=curr,
                        financial_amount=fin_amt,
                        transaction_amount=amount,
                        gross_exposure=amount,
                        outstanding_amount=0.0,
                        net_exposure=0.0,
                        potential_exposure=amount,
                        potential_cost_exposure=formatted_amount_str,
                        actual_loss=0.0,
                        actual_loss_status="ESTABLISHED",
                        recoverable_amount=amount,
                        recovered_amount=recovered_amt,
                        unrecovered_amount=0.0,
                        recoverability="RECOVERED",
                        recoverability_status="RECOVERED",
                        amount_confidence="HIGH",
                        classification_confidence="HIGH",
                        recovery_confidence="HIGH",
                        actual_loss_confidence="HIGH",
                        cost_components=[],
                        cost_drivers=[],
                        calculation_basis=f"Duplicate payment: {formatted_amount_str} - Confirmed refund: {rec_formatted} = Actual loss: {format_currency_amount(0.0, curr)}",
                        assumptions=[],
                        missing_cost_inputs=[],
                        evidence_required=["Bank reconciliation statement confirming credit receipt"],
                        evidence_ids=matched_cids,
                        confidence="HIGH",
                        narrative=f"Duplicate payment of {formatted_amount_str} was identified; full recovery of {rec_formatted} has been confirmed (actual financial loss: {format_currency_amount(0.0, curr)}).",
                    )
                else:
                    # Partial recovery
                    unrecovered = amount - recovered_amt
                    unrec_formatted = format_currency_amount(unrecovered, curr)
                    return CostImpact(
                        cost_factor_detected=True,
                        cost_factor_type=factor_type,
                        financial_factor=factor_type,
                        financial_status="POTENTIAL_EXPOSURE",
                        currency=curr,
                        financial_amount=fin_amt,
                        transaction_amount=amount,
                        gross_exposure=amount,
                        outstanding_amount=unrecovered,
                        net_exposure=unrecovered,
                        potential_exposure=amount,
                        potential_cost_exposure=formatted_amount_str,
                        actual_loss=unrecovered,
                        actual_loss_status="POTENTIAL_UNRECOVERED",
                        recoverable_amount=amount,
                        recovered_amount=recovered_amt,
                        unrecovered_amount=unrecovered,
                        recoverability="PARTIALLY_RECOVERED",
                        recoverability_status="PARTIALLY_RECOVERED",
                        amount_confidence="HIGH",
                        classification_confidence="HIGH",
                        recovery_confidence="HIGH",
                        actual_loss_confidence="MEDIUM",
                        cost_components=[],
                        cost_drivers=[],
                        calculation_basis=f"Duplicate payment: {formatted_amount_str} - Confirmed refund: {rec_formatted} = Unrecovered exposure: {unrec_formatted}",
                        assumptions=["Remaining balance requires recovery tracking."],
                        missing_cost_inputs=["Remaining balance recovery timeline"],
                        evidence_required=["Supplier credit note for remaining balance", "Collection follow-up log"],
                        evidence_ids=matched_cids,
                        confidence="HIGH",
                        narrative=f"Duplicate payment of {formatted_amount_str} was identified; partial refund of {rec_formatted} confirmed, leaving unrecovered financial exposure of {unrec_formatted}.",
                    )
            else:
                # Standard unconfirmed duplicate payment
                return CostImpact(
                    cost_factor_detected=True,
                    cost_factor_type=factor_type,
                    financial_factor=factor_type,
                    financial_status="VERIFIED",
                    currency=curr,
                    financial_amount=fin_amt,
                    transaction_amount=amount,
                    gross_exposure=amount,
                    outstanding_amount=amount,
                    net_exposure=amount,
                    potential_exposure=amount,
                    potential_cost_exposure=formatted_amount_str,
                    actual_loss=None,
                    actual_loss_status="NOT_ESTABLISHED",
                    recoverability="UNKNOWN",
                    recoverability_status="REQUIRES_VERIFICATION",
                    amount_confidence="HIGH",
                    classification_confidence="HIGH",
                    recovery_confidence="UNKNOWN",
                    actual_loss_confidence="UNKNOWN",
                    cost_components=[],
                    cost_drivers=[],
                    calculation_basis=f"Verified duplicate payment transaction: {formatted_amount_str}",
                    assumptions=["Final loss depends on whether supplier credit or reversal is obtained."],
                    missing_cost_inputs=missing_inputs,
                    evidence_required=evidence_req,
                    evidence_ids=matched_cids,
                    confidence="HIGH",
                    narrative=f"Duplicate payment of {formatted_amount_str} to a supplier was identified (verified transaction). Potential financial exposure is {formatted_amount_str}; actual financial loss is not established pending verification of recovery or supplier credit.",
                )

        # -------------------------------------------------------------------
        # 3b. Standard Direct Cost (Rework, Scrap, Penalty, Labor, etc.)
        # -------------------------------------------------------------------
        is_estimated = bool(_ESTIMATE_MARKERS_RE.search(combined_text))
        is_verified = (
            not is_estimated
            and any(_CURRENCY_SYMBOLS_RE.search(vt) for vt in verified_claim_texts)
        )

        missing_inputs, evidence_req = get_default_missing_evidence(factor_type, combined_text)
        fin_amt = FinancialAmount(
            amount=amount,
            formatted=formatted_amount_str,
            currency=curr or "INR",
            factor=factor_type,
            source_claim_ids=matched_cids,
            support_status="VERIFIED" if is_verified else "REPORTED",
            confidence="HIGH" if is_verified else "MEDIUM",
        )

        if is_verified:
            return CostImpact(
                cost_factor_detected=True,
                cost_factor_type=factor_type,
                financial_factor=factor_type,
                financial_status=CostEvidenceStatus.VERIFIED.value,
                currency=curr,
                financial_amount=fin_amt,
                verified_cost=amount,
                reported_cost=amount,
                estimated_cost=None,
                gross_exposure=amount,
                outstanding_amount=amount,
                net_exposure=amount,
                potential_exposure=amount,
                potential_cost_exposure=formatted_amount_str,
                actual_loss=amount,
                actual_loss_status="VERIFIED",
                amount_confidence="HIGH",
                classification_confidence="HIGH",
                recovery_confidence="UNKNOWN",
                actual_loss_confidence="HIGH",
                cost_components=[
                    CostComponent(
                        name=f"Incurred {factor_type.lower()}",
                        amount=amount,
                        currency=curr,
                        category=factor_type,
                        basis="Directly stated in verified audit evidence",
                        provenance="VERIFIED",
                    )
                ],
                cost_drivers=drivers,
                calculation_basis=f"Verified finding observation: {formatted_amount_str}",
                assumptions=[],
                missing_cost_inputs=[],
                evidence_required=["Financial ledger verification and accounting cost center confirmation"],
                evidence_ids=matched_cids,
                confidence="HIGH",
                narrative=f"The organization incurred a verified {formatted_amount_str} in {factor_type.lower()} costs.",
            )
        else:
            # REPORTED or ESTIMATED
            status_val = CostEvidenceStatus.ESTIMATED.value if is_estimated else CostEvidenceStatus.REPORTED.value
            return CostImpact(
                cost_factor_detected=True,
                cost_factor_type=factor_type,
                financial_factor=factor_type,
                financial_status=status_val,
                currency=curr,
                financial_amount=fin_amt,
                verified_cost=None,
                reported_cost=amount if not is_estimated else None,
                estimated_cost=amount if is_estimated else None,
                gross_exposure=amount,
                outstanding_amount=amount,
                net_exposure=amount,
                potential_exposure=amount,
                potential_cost_exposure=formatted_amount_str,
                actual_loss=None,
                actual_loss_status="NOT_ESTABLISHED",
                amount_confidence="MEDIUM",
                classification_confidence="HIGH",
                recovery_confidence="UNKNOWN",
                actual_loss_confidence="UNKNOWN",
                cost_components=[
                    CostComponent(
                        name=f"Reported/Estimated {factor_type.lower()}",
                        amount=amount,
                        currency=curr,
                        category=factor_type,
                        basis="Attributed estimate in finding text",
                        provenance="REPORTED",
                    )
                ],
                cost_drivers=drivers,
                calculation_basis=f"Reported/estimated statement in finding: {formatted_amount_str}",
                assumptions=["Cost amount is subject to independent financial verification."],
                missing_cost_inputs=missing_inputs,
                evidence_required=evidence_req + ["Financial records and vendor/payroll receipts to confirm actual amount"],
                evidence_ids=matched_cids,
                confidence="MEDIUM",
                narrative=f"The finding is associated with a reported/estimated financial exposure of approximately {formatted_amount_str} (requires verification).",
            )

    # 4. Financial Signal Present but Amounts NOT CALCULABLE
    missing_inputs, evidence_req = get_default_missing_evidence(factor_type, combined_text)
    return CostImpact(
        cost_factor_detected=True,
        cost_factor_type=factor_type,
        financial_factor=factor_type,
        financial_status=CostEvidenceStatus.REQUIRES_ASSESSMENT.value,
        currency=curr,
        verified_cost=None,
        reported_cost=None,
        estimated_cost=None,
        potential_exposure=None,
        potential_cost_exposure="NOT CALCULABLE FROM AVAILABLE EVIDENCE",
        actual_loss=None,
        actual_loss_status="NOT_ESTABLISHED",
        cost_components=[],
        cost_drivers=drivers,
        calculation_basis=None,
        assumptions=[],
        missing_cost_inputs=missing_inputs,
        evidence_required=evidence_req,
        evidence_ids=matched_cids,
        confidence="LOW",
        narrative="Potential financial exposure requires assessment — specific monetary value is not calculable from available evidence without additional accounting/production records.",
    )
