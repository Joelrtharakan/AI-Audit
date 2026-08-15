# LQMS AI Agent Evaluation & Regression Report

**Overall Evaluation Score:** `84.5 / 100`

**Production Readiness Gate Status:** `FAILED`

### Production Gate Reasons:
- :warning: Overall score 84.5 below threshold 90.0

## 1. Dimension Scores Summary

| Dimension | Average Score (%) | Status |
|---|---|---|
| Fact Preservation | 100.0% | PASSED |
| Observation Quality | 4.0% | NEEDS IMPROVEMENT |
| Root Cause Discipline | 100.0% | PASSED |
| Evidence Boundary | 100.0% | PASSED |
| Hypothesis Quality | 70.0% | NEEDS IMPROVEMENT |
| Evidence Recommendations | 71.0% | NEEDS IMPROVEMENT |
| Capa Discipline | 100.0% | PASSED |
| Impact Assessment | 100.0% | PASSED |
| Consistency | 100.0% | PASSED |

## 2. Failure Pattern Analysis

| Failure Code | Severity | Occurrences | Example |
|---|---|---|---|
| `OBSERVATION_MISCLASSIFICATION` | `MEDIUM` | 24 | Observation quality evaluated as ObservationQualityStatus.SUFFICIENT, expected SUFFICIENT. |
| `IRRELEVANT_HYPOTHESIS` | `MEDIUM` | 15 | Candidate hypothesis invokes unanchored 2nd-order domain: 'The active procedure version for the an operator may not reflect the required operational steps.' |
| `MISSING_EVIDENCE_REQUIREMENT` | `MEDIUM` | 29 | Recommended evidence missed key item: 'SOP-CLN-004 copy' |

## 3. Finding-by-Finding Breakdown

| Finding ID | Score (0-100) | Failures Count |
|---|---|---|
| `F001` | 77.5 | 4 |
| `F002` | 92.5 | 3 |
| `F003` | 80.0 | 3 |
| `F004` | 87.5 | 2 |
| `F005` | 75.0 | 5 |
| `F006` | 90.0 | 1 |
| `F007` | 90.0 | 1 |
| `F008` | 90.0 | 1 |
| `F009` | 85.0 | 2 |
| `F010` | 85.0 | 3 |
| `F011` | 82.5 | 4 |
| `F012` | 75.0 | 5 |
| `F013` | 75.0 | 5 |
| `F014` | 85.0 | 3 |
| `F015` | 85.0 | 3 |
| `F016` | 85.0 | 2 |
| `F017` | 85.0 | 3 |
| `F018` | 87.5 | 2 |
| `F019` | 85.0 | 3 |
| `F020` | 87.5 | 2 |
| `F021` | 87.5 | 2 |
| `F022` | 90.0 | 1 |
| `F023` | 90.0 | 1 |
| `F024` | 75.0 | 5 |
| `F025` | 85.0 | 2 |