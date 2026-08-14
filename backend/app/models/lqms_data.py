"""Typed data models for LQMS records returned by the tool layer.

These are the data contracts between the ASP.NET LQMS and the agent.
When LQMS_ASPNET_BASE_URL is empty or the server is unreachable, the
ASP.NET client returns empty-but-valid instances and logs a warning.
The agent records the gap in its evidence ledger and continues.

Nothing here is ever written back to the LQMS -- these are all read models.
"""

from __future__ import annotations

from pydantic import BaseModel


class FindingRecord(BaseModel):
    finding_id: str = ""
    ca_number: str = ""
    audit_number: str = ""
    audit_date: str = ""
    clause_number: str = ""
    audit_question: str = ""
    departments: list[str] = []
    nature_of_nc: str = ""
    auditors: list[str] = []
    auditees: list[str] = []
    audit_criteria: str = ""
    area_audited: str = ""
    finding_type: str = ""
    severity: str = ""
    likelihood: str = ""
    risk_result: str = ""
    observation: str = ""
    has_attachments: bool = False
    is_available: bool = False  # False when LQMS not configured


class AuditRecord(BaseModel):
    audit_id: str = ""
    audit_number: str = ""
    audit_date: str = ""
    audit_type: str = ""
    scope: str = ""
    lead_auditor: str = ""
    auditees: list[str] = []
    departments: list[str] = []
    standard: str = ""
    status: str = ""
    is_available: bool = False


class CAPARecord(BaseModel):
    capa_id: str = ""
    ca_number: str = ""
    finding_id: str = ""
    immediate_action: str = ""
    root_cause: str = ""
    root_cause_category: str = ""
    preventive_action: str = ""
    impact_analysis: str = ""
    status: str = ""
    opened_date: str = ""
    closed_date: str = ""
    department: str = ""
    is_available: bool = False


class TrainingRecord(BaseModel):
    record_id: str = ""
    staff_role: str = ""
    department: str = ""
    training_title: str = ""
    training_date: str = ""
    document_reference: str = ""
    completed: bool = False
    is_available: bool = False


class EquipmentRecord(BaseModel):
    equipment_id: str = ""
    name: str = ""
    department: str = ""
    last_calibration_date: str = ""
    next_calibration_due: str = ""
    calibration_status: str = ""
    maintenance_notes: str = ""
    is_available: bool = False


class DepartmentInfo(BaseModel):
    department_id: str = ""
    name: str = ""
    head: str = ""
    staff_count: int = 0
    primary_activities: list[str] = []
    applicable_standards: list[str] = []
    is_available: bool = False


class PreviousFinding(BaseModel):
    finding_id: str = ""
    audit_number: str = ""
    audit_date: str = ""
    observation: str = ""
    root_cause_category: str = ""
    ca_status: str = ""
    recurrence: bool = False
    department: str = ""
    clause: str = ""
    is_available: bool = False
