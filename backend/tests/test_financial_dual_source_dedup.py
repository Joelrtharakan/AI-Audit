"""Regression for a real production defect: `extract_financial_observations`
received BOTH `evidence_ledger` (the raw evidence items) AND `evidence_claims`
(app.models.agent.EvidenceClaim, a claim-level restructuring of the exact
same sentences -- see `extract_claims()` in understanding.py, called on the
SAME evidence_ledger) from `report_generator.py`, and concatenated both into
one `sources` list without deduplication.

For any finding where a quantity and a rate live in separate evidence
statements -- which is most real, unstructured audit findings -- this made
the extractor see the quantity TWICE and the rate TWICE, which the ambiguity
guard in the cross-evidence linking logic correctly refuses to resolve
(never guesses which of two candidate quantities/rates belongs together),
silently losing the calculation entirely rather than computing the correct
QUANTITY x RATE result. This is what actually produced "EQ-207 -> INR
12,000" instead of "INR 120,000" via the deterministic-regex path -- the
path actually active in production, since financial_semantic_reasoning_enabled
defaults to False (see app/config.py).

This is a general extractor bug (duplicate source ingestion), not a rule for
"EQ-207" or "10 hours" -- the fix (only fall back to evidence_claims when
evidence_ledger is empty, since it's always derived FROM evidence_ledger and
never carries independent facts) is in app/financial/extractor.py.
"""

from __future__ import annotations

from app.financial.engine import analyze_financial_exposure
from app.models.agent import EvidenceClaim, EvidenceItem, EvidenceStatus


def _claim_for(item: EvidenceItem, claim_id: str) -> EvidenceClaim:
    """Mirrors how understanding.py's extract_claims() derives an
    EvidenceClaim 1:1 from an evidence_ledger item -- the actual shape that
    reaches report_generator.py in production, not a synthetic shortcut."""
    return EvidenceClaim(
        claim_id=claim_id,
        text=item.claim,
        predicate=item.claim,
        source="finding_text",
        status=item.status,
        confidence="HIGH",
        attribution="AUDITOR_OBSERVED",
    )


class TestDualSourceNeverDuplicatesEvidence:
    """Same fact set, four unrelated domains -- proving the fix is general,
    not tuned to the equipment-downtime wording."""

    def test_quantity_and_rate_pass_both_ledger_and_claims(self):
        e1 = EvidenceItem(claim="Equipment EQ-207 was unavailable for 10 hours.", status=EvidenceStatus.VERIFIED, source="AUDITOR_FINDING")
        e2 = EvidenceItem(claim="Production records verify an average production disruption cost of INR 12,000 per hour for this equipment.", status=EvidenceStatus.VERIFIED, source="AUDITOR_FINDING")
        ledger = [e1, e2]
        claims = [_claim_for(e1, "C1"), _claim_for(e2, "C2")]
        text = e1.claim + " " + e2.claim

        result = analyze_financial_exposure(finding_text=text, evidence_ledger=ledger, evidence_claims=claims)
        assert result.confirmed_impact.verified_gross_exposure == 120000.0

    def test_scrap_material_quantity_rate_dual_source(self):
        e1 = EvidenceItem(claim="2,400 units were discarded as scrap.", status=EvidenceStatus.VERIFIED, source="AUDITOR_FINDING")
        e2 = EvidenceItem(claim="Manufacturing records indicate an average material value of EUR 18.50 per unit.", status=EvidenceStatus.VERIFIED, source="AUDITOR_FINDING")
        ledger = [e1, e2]
        claims = [_claim_for(e1, "C1"), _claim_for(e2, "C2")]
        text = e1.claim + " " + e2.claim

        result = analyze_financial_exposure(finding_text=text, evidence_ledger=ledger, evidence_claims=claims)
        assert result.confirmed_impact.verified_gross_exposure == 2400 * 18.50

    def test_support_hours_rate_dual_source(self):
        e1 = EvidenceItem(claim="850 support hours were logged against the incident.", status=EvidenceStatus.VERIFIED, source="AUDITOR_FINDING")
        e2 = EvidenceItem(claim="The verified blended labor rate was USD 42 per hour.", status=EvidenceStatus.VERIFIED, source="AUDITOR_FINDING")
        ledger = [e1, e2]
        claims = [_claim_for(e1, "C1"), _claim_for(e2, "C2")]
        text = e1.claim + " " + e2.claim

        result = analyze_financial_exposure(finding_text=text, evidence_ledger=ledger, evidence_claims=claims)
        assert result.confirmed_impact.verified_gross_exposure == 850 * 42

    def test_ledger_only_and_claims_only_agree_with_dual_source(self):
        e1 = EvidenceItem(claim="1,250 transactions required manual reprocessing.", status=EvidenceStatus.VERIFIED, source="AUDITOR_FINDING")
        e2 = EvidenceItem(claim="Logistics confirmed a verified handling charge of AED 32 per transaction.", status=EvidenceStatus.VERIFIED, source="AUDITOR_FINDING")
        ledger = [e1, e2]
        claims = [_claim_for(e1, "C1"), _claim_for(e2, "C2")]
        text = e1.claim + " " + e2.claim

        ledger_only = analyze_financial_exposure(finding_text=text, evidence_ledger=ledger)
        claims_only = analyze_financial_exposure(finding_text=text, evidence_claims=claims)
        dual_source = analyze_financial_exposure(finding_text=text, evidence_ledger=ledger, evidence_claims=claims)

        expected = 1250 * 32
        assert ledger_only.confirmed_impact.verified_gross_exposure == expected
        assert claims_only.confirmed_impact.verified_gross_exposure == expected
        assert dual_source.confirmed_impact.verified_gross_exposure == expected
