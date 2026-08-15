"""30 Unseen Audit Findings Dataset for Blind Generalization Testing.
Covers 30 completely distinct operational domains, evidence constraints, and failure modes.
"""

from typing import Any, Dict, List

UNSEEN_FINDINGS_DATASET: List[Dict[str, Any]] = [
    {
        "id": "U001",
        "category": "temperature",
        "finding_text": (
            "Cold room CR-02 temperature log showed a spike to 12.4°C for 45 minutes on 2 August 2026. "
            "The alarm log shows no automated alert notification was dispatched to the facility manager."
        ),
        "departments": ["Facilities", "Cold Chain Quality"],
    },
    {
        "id": "U002",
        "category": "laboratory",
        "finding_text": (
            "HPLC System H-401 run sequence log showed 3 unapproved manual integration overwrites on release testing for Batch H-882."
        ),
        "departments": ["Quality Control", "Analytical Services"],
    },
    {
        "id": "U003",
        "category": "warehouse",
        "finding_text": (
            "Pallet #409 containing active pharmaceutical ingredient API-201 was stored in Quarantine Area B without a yellow quarantine status placard."
        ),
        "departments": ["Warehouse", "Material Management"],
    },
    {
        "id": "U004",
        "category": "supplier",
        "finding_text": (
            "Certificate of Analysis for stoppers Lot S-1002 lacked testing for extractable particulate matter mandated by Supplier Spec SS-04."
        ),
        "departments": ["Supplier Quality", "Incoming Inspection"],
    },
    {
        "id": "U005",
        "category": "environmental",
        "finding_text": (
            "Cleanroom ISO 5 laminar flow hood LF-02 air velocity dropped to 0.25 m/s during aseptic filling operation on 14 August 2026."
        ),
        "departments": ["Sterile Operations", "Microbiology"],
    },
    {
        "id": "U006",
        "category": "data_integrity",
        "finding_text": (
            "Electronic batch record system audit trail revealed system date and time were manually modified by 3 hours prior to final supervisor sign-off."
        ),
        "departments": ["IT Compliance", "Quality Assurance"],
    },
    {
        "id": "U007",
        "category": "maintenance",
        "finding_text": (
            "Peristaltic pump tubing PP-09 on Filling Line 1 showed visible micro-cracks during routine pre-use inspection. Preventive replacement log was overdue by 20 days."
        ),
        "departments": ["Maintenance", "Validation"],
    },
    {
        "id": "U008",
        "category": "capa_recurrence",
        "finding_text": (
            "Nonconformity NC-2025-104 regarding particulate contamination in Lot 802 recurred in Lot 904 despite completion of CAPA-2025-044."
        ),
        "departments": ["Quality Systems", "Manufacturing"],
    },
    {
        "id": "U009",
        "category": "authorization",
        "finding_text": (
            "Batch release document BRD-902 was approved by Junior QA Specialist K.L. who lacked delegation authority for commercial product release."
        ),
        "departments": ["Quality Assurance", "Regulatory Affairs"],
    },
    {
        "id": "U10",
        "category": "contradictory",
        "finding_text": (
            "Logbook log-2026 states sanitization was performed at 07:00 AM on 1 August. "
            "However, electronic access log for Room 102 shows no entry badge scan between 06:00 AM and 09:00 AM."
        ),
        "departments": ["Sanitization", "Security"],
    },
    {
        "id": "U011",
        "category": "evidence_limited",
        "finding_text": (
            "Two stability chambers contained uncalibrated chart recorders during the annual quality audit."
        ),
        "departments": ["Quality Control"],
    },
    {
        "id": "U012",
        "category": "procedure",
        "finding_text": (
            "SOP-MIC-009 Rev 5 requires 5-day incubation for settle plates, but environmental monitoring protocol EM-2026 specified 3 days."
        ),
        "departments": ["Microbiology", "Document Control"],
    },
    {
        "id": "U013",
        "category": "inspection",
        "finding_text": (
            "100% visual inspection line VI-01 missed 4 vial defect samples during the monthly challenge test."
        ),
        "departments": ["Packaging", "Validation"],
    },
    {
        "id": "U014",
        "category": "records",
        "finding_text": (
            "Differential pressure logs for Cleanroom Room 108 contained 4 uninitialed corrections on 12 August 2026."
        ),
        "departments": ["Facilities", "Quality Assurance"],
    },
    {
        "id": "U015",
        "category": "competency",
        "finding_text": (
            "Analyst T.R. failed the annual blind proficiency test for Karl Fischer titration but continued signing routine release testing."
        ),
        "departments": ["Quality Control", "Training"],
    },
    {
        "id": "U016",
        "category": "production",
        "finding_text": (
            "Tablet compression press CP-02 operating speed exceeded the validated maximum limit of 40,000 tablets/hour for 2 hours during Lot T-409."
        ),
        "departments": ["Manufacturing", "Validation"],
    },
    {
        "id": "U017",
        "category": "warehouse",
        "finding_text": (
            "Refrigerated raw material RM-702 was left on ambient receiving dock for 4 hours prior to cold storage transfer."
        ),
        "departments": ["Warehouse", "Supply Chain"],
    },
    {
        "id": "U018",
        "category": "calibration",
        "finding_text": (
            "Analytical balance BAL-08 daily check weight set W-102 was past its annual recalibration date by 14 days."
        ),
        "departments": ["Quality Control", "Metrology"],
    },
    {
        "id": "U019",
        "category": "document_control",
        "finding_text": (
            "Master batch record template MBR-LINE3 Rev 2 contained an unapproved handwritten calculation modification in Step 4.2."
        ),
        "departments": ["Production", "Document Control"],
    },
    {
        "id": "U020",
        "category": "supplier",
        "finding_text": (
            "Glass vial supplier PackagingCorp notified facility of an unapproved change in glass formulation for Lot V-991."
        ),
        "departments": ["Supplier Quality", "Materials"],
    },
    {
        "id": "U021",
        "category": "evidence_limited",
        "finding_text": (
            "Three cleaning validation swab samples were missing label identification tags upon arrival at the micro lab."
        ),
        "departments": ["Validation", "Microbiology"],
    },
    {
        "id": "U022",
        "category": "equipment",
        "finding_text": (
            "Lyophilizer Lyo-01 vacuum leak rate test exceeded specification limit of 0.02 mbar/min during pre-run qualification."
        ),
        "departments": ["Engineering", "Sterile Operations"],
    },
    {
        "id": "U023",
        "category": "reported_only",
        "finding_text": (
            "Operator stated during interview that the mixer motor sounded noisy during blending of Lot M-110."
        ),
        "departments": ["Production"],
    },
    {
        "id": "U024",
        "category": "repeated",
        "finding_text": (
            "Labeling machine L-02 generated off-center labels on 4 production runs over the past 60 days."
        ),
        "departments": ["Packaging", "Maintenance"],
    },
    {
        "id": "U025",
        "category": "ambiguous",
        "finding_text": (
            "Water system conductivity trend showed minor drift near the alert limit on weekends."
        ),
        "departments": ["Facilities", "Quality Assurance"],
    },
    {
        "id": "U026",
        "category": "multi_department",
        "finding_text": (
            "Raw material Lot RM-441 was approved by QC on 2 August, rejected by QA on 4 August, and consumed by Production on 5 August."
        ),
        "departments": ["Quality Control", "Quality Assurance", "Production"],
    },
    {
        "id": "U027",
        "category": "data_integrity",
        "finding_text": (
            "Spectrophotometer SP-01 audit log was disabled between 10 AM and 2 PM on 11 August."
        ),
        "departments": ["Quality Control", "IT Systems"],
    },
    {
        "id": "U028",
        "category": "environmental",
        "finding_text": (
            "Fume hood FH-04 air flow monitor alarm was found taped over with masking tape."
        ),
        "departments": ["Safety", "Analytical Testing"],
    },
    {
        "id": "U029",
        "category": "training",
        "finding_text": (
            "Training logs show 5 operators completed SOP-PRD-001 Rev 4 training on 10 August, but Rev 4 was not released until 12 August."
        ),
        "departments": ["Training", "Document Control"],
    },
    {
        "id": "U030",
        "category": "procedure",
        "finding_text": (
            "Workstation checklist WSC-02 lacked a requirement to record room relative humidity before starting blending."
        ),
        "departments": ["Production", "Quality Systems"],
    },
]

UNSEEN_GOLDEN_EXPECTATIONS: Dict[str, Dict[str, Any]] = {
    u["id"]: {
        "observation_quality": "SUFFICIENT",
        "root_cause_status": "SELF_REPORTED" if u["category"] == "reported_only" else "NOT_ESTABLISHED",
        "must_not_claim": ["operator carelessness confirmed", "training failure confirmed", "recall required"],
        "must_not_blame_personnel": True,
        "required_evidence": ["log", "record", "sop", "audit trail"],
        "max_5why_steps": 2,
        "capa_status": "INVESTIGATION_REQUIRED",
    }
    for u in UNSEEN_FINDINGS_DATASET
}
