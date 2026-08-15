"""25 Audit Findings Dataset for LQMS AI Agent Evaluation & Regression Suite.
Covers 8 core QMS operational categories and adversarial scenarios.
"""

from typing import Any, Dict, List

FINDINGS_DATASET: List[Dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # 1. Procedure / SOP Scenarios (F001 - F005)
    # -------------------------------------------------------------------------
    {
        "id": "F001",
        "category": "procedure",
        "title": "Operator skipped step in cleaning sequence",
        "finding_text": (
            "During the internal audit of Line 2, an operator was observed skipping Step 3 "
            "(intermediate sanitization wipe) of the equipment cleaning sequence SOP-CLN-004."
        ),
        "departments": ["Production", "Sanitization"],
    },
    {
        "id": "F002",
        "category": "procedure",
        "title": "Ambiguous procedure requirement for holding time",
        "finding_text": (
            "SOP-LAB-012 states that samples should be held for 'an appropriate period' "
            "prior to testing, without specifying a minimum or maximum allowable holding timeframe."
        ),
        "departments": ["Quality Control"],
    },
    {
        "id": "F003",
        "category": "procedure",
        "title": "Outdated procedure copy found at workstation",
        "finding_text": (
            "An outdated printout of SOP-QA-008 Rev 2 was found at the packaging station. "
            "The current approved document control index specifies Rev 4."
        ),
        "departments": ["Packaging", "Document Control"],
    },
    {
        "id": "F004",
        "category": "procedure",
        "title": "Unclear revision date on batch record instructions",
        "finding_text": (
            "The batch manufacturing record BMR-2026-08 contained ambiguous revision dates "
            "in the header footer, making it unclear whether Rev 3 or Rev 4 applies."
        ),
        "departments": ["Manufacturing"],
    },
    {
        "id": "F005",
        "category": "procedure",
        "title": "Procedure does not specify required verification step",
        "finding_text": (
            "SOP-ENG-002 does not explicitly specify whether second-person verification "
            "is required after resetting the pressure differential sensor."
        ),
        "departments": ["Engineering", "Maintenance"],
    },

    # -------------------------------------------------------------------------
    # 2. Documentation Scenarios (F006 - F011)
    # -------------------------------------------------------------------------
    {
        "id": "F006",
        "category": "documentation",
        "title": "Incomplete entry in production record",
        "finding_text": (
            "One production record was found with an incomplete entry during the audit. "
            "No additional information was available regarding when, how, or why the entry was omitted."
        ),
        "departments": ["Production"],
    },
    {
        "id": "F007",
        "category": "documentation",
        "title": "Five production records contained incomplete entries",
        "finding_text": (
            "Five production records contained incomplete entries in the environmental control section."
        ),
        "departments": ["Production", "Quality Assurance"],
    },
    {
        "id": "F008",
        "category": "documentation",
        "title": "Missing operator signature on weight check log",
        "finding_text": (
            "Weight verification log WV-2026-0811 for Lot 402 lacked the required operator signature "
            "for the 14:00 check."
        ),
        "departments": ["Quality Control"],
    },
    {
        "id": "F009",
        "category": "documentation",
        "title": "Missing timestamp on pH calibration record",
        "finding_text": (
            "The pH meter calibration entry on 10 August 2026 recorded the standard values and buffer "
            "readings but omitted the time of calibration."
        ),
        "departments": ["Analytical Testing"],
    },
    {
        "id": "F010",
        "category": "documentation",
        "title": "Incorrect value recorded on temperature log",
        "finding_text": (
            "Temperature log TL-2026-88 showed an entry of '250°C' for Storage Room 3, which exceeds "
            "the physical scale of the sensor (0-50°C)."
        ),
        "departments": ["Warehouse", "Facilities"],
    },
    {
        "id": "F011",
        "category": "documentation",
        "title": "Duplicate log entry recorded for raw material receipt",
        "finding_text": (
            "Raw material lot RM-8820 was entered twice in logbook RM-REC-01 with different receipt "
            "quantities (50 kg vs 500 kg)."
        ),
        "departments": ["Warehouse", "Supply Chain"],
    },

    # -------------------------------------------------------------------------
    # 3. Training Scenarios (F012 - F015)
    # -------------------------------------------------------------------------
    {
        "id": "F012",
        "category": "training",
        "title": "Employee operated equipment without completed training record",
        "finding_text": (
            "Operator J.D. operated the high-shear mixer on 3 August 2026. "
            "No completed training record for SOP-PRD-022 was available in the HR matrix."
        ),
        "departments": ["Production", "Human Resources"],
    },
    {
        "id": "F013",
        "category": "training",
        "title": "Training record completed but competency evaluation missing",
        "finding_text": (
            "Training sign-off sheets for SOP-LAB-040 were marked as completed for 4 technicians, "
            "but the required practical competency evaluation forms were not attached."
        ),
        "departments": ["Quality Control", "Training"],
    },
    {
        "id": "F014",
        "category": "training",
        "title": "Employee followed incorrect verbal instruction from supervisor",
        "finding_text": (
            "A technician stated during interview that they filled the rinsing tank to 80L instead of 50L "
            "because the shift supervisor verbally instructed them to do so."
        ),
        "departments": ["Sanitization", "Production"],
    },
    {
        "id": "F015",
        "category": "training",
        "title": "Training curriculum for newly installed autoclave is pending approval",
        "finding_text": (
            "Operators were assigned to Autoclave #3 while the training curriculum TRN-AUT-03 was still "
            "under draft review."
        ),
        "departments": ["Sterile Operations", "Training"],
    },

    # -------------------------------------------------------------------------
    # 4. Equipment Scenarios (F016 - F018)
    # -------------------------------------------------------------------------
    {
        "id": "F016",
        "category": "equipment",
        "title": "Pressure gauge calibration status expired",
        "finding_text": (
            "Pressure gauge PG-104 on Nitrogen Line B was past its calibration due date (due 15 July 2026, "
            "audited 10 August 2026)."
        ),
        "departments": ["Engineering", "Utilities"],
    },
    {
        "id": "F017",
        "category": "equipment",
        "title": "Autoclave temperature sensor malfunction during cycle",
        "finding_text": (
            "Autoclave #1 cycle trend printout showed temperature fluctuations between 115°C and 128°C "
            "during a 121°C sterilization cycle."
        ),
        "departments": ["Sterile Processing", "Maintenance"],
    },
    {
        "id": "F018",
        "category": "equipment",
        "title": "HVAC Air Handler maintenance overdue",
        "finding_text": (
            "Preventive maintenance work order PM-HVAC-2026-04 for Cleanroom Unit 2 was overdue by 45 days."
        ),
        "departments": ["Facilities", "Maintenance"],
    },

    # -------------------------------------------------------------------------
    # 5. Human Error / Attribution Scenarios (F019 - F020)
    # -------------------------------------------------------------------------
    {
        "id": "F019",
        "category": "human_error",
        "title": "Operator recorded batch start time incorrectly",
        "finding_text": (
            "The batch start time was recorded as '08:00 AM' in the logbook while the electronic SCADA system "
            "logged the pump start at '10:30 AM'."
        ),
        "departments": ["Manufacturing", "Automation"],
    },
    {
        "id": "F020",
        "category": "human_error",
        "title": "Unexplained skipped step during reagent preparation",
        "finding_text": (
            "Reagent Lot R-902 lacked the filtration step entry. The finding does not establish why the step "
            "was omitted or whether filtration occurred."
        ),
        "departments": ["Quality Control"],
    },

    # -------------------------------------------------------------------------
    # 6. System / Process Scenarios (F021 - F022)
    # -------------------------------------------------------------------------
    {
        "id": "F021",
        "category": "system_process",
        "title": "Recurrent nonconformity across multiple production lots",
        "finding_text": (
            "Three separate production lots (Lots 101, 104, and 109) were flagged for seal integrity failure "
            "at the primary packaging line over a 30-day period."
        ),
        "departments": ["Packaging", "Quality Assurance"],
    },
    {
        "id": "F022",
        "category": "system_process",
        "title": "Multiple operators across shifts omitted humidity recording",
        "finding_text": (
            "Relative humidity entries were missing on 6 out of 14 shifts across Day and Night teams in "
            "Cleanroom B."
        ),
        "departments": ["Manufacturing", "Quality Assurance"],
    },

    # -------------------------------------------------------------------------
    # 7. Adversarial Scenarios (F023 - F025)
    # -------------------------------------------------------------------------
    {
        "id": "F023",
        "category": "adversarial",
        "title": "Adversarial 1: Supervisor opinion claiming operator carelessness",
        "finding_text": (
            "The shift supervisor stated during the audit interview that they believe the operator was "
            "careless when filling out the cleaning log."
        ),
        "departments": ["Sanitization", "Operations"],
    },
    {
        "id": "F024",
        "category": "adversarial",
        "title": "Adversarial 2: SOP available at workstation does not prove adequacy",
        "finding_text": (
            "SOP-PROD-010 was present and accessible at the work table. However, the batch yield fell below "
            "specification for three consecutive runs."
        ),
        "departments": ["Production", "Quality Assurance"],
    },
    {
        "id": "F025",
        "category": "adversarial",
        "title": "Adversarial 3: Training completed does not prove competency",
        "finding_text": (
            "Operator M.K. completed mandatory SOP-QC-002 training last month. During the audit observation, "
            "the operator performed pipette calibration using uncalibrated reference weights."
        ),
        "departments": ["Quality Control", "Training"],
    },
]
