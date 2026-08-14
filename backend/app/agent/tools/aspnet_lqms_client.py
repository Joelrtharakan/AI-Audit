"""Real HTTP client for the ASP.NET LQMS API.

Design:
  - When LQMS_ASPNET_BASE_URL is empty or the server is unreachable, every method
    returns an empty-but-valid typed result (is_available=False) and logs a warning.
  - The agent receives this empty result, adds a gap to the evidence ledger, and
    continues. No crash, no fake data.
  - All methods are GET-only. Write operations are structurally impossible here.
  - Bearer token auth via LQMS_ASPNET_API_KEY (optional; omitted if empty).

To connect to a real ASP.NET LQMS:
  1. Set LQMS_ASPNET_BASE_URL=https://your-lqms-host/api in .env
  2. Set LQMS_ASPNET_API_KEY=<token> if the API requires auth
  3. Adjust the URL path patterns in each method to match your ASP.NET routes
"""

from __future__ import annotations

import logging

import httpx

from app.config import get_settings
from app.models.lqms_data import (
    AuditRecord,
    CAPARecord,
    DepartmentInfo,
    EquipmentRecord,
    FindingRecord,
    PreviousFinding,
    TrainingRecord,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 10.0  # per-request; overridden by agent_tool_timeout_seconds in config


def _empty_finding() -> FindingRecord:
    return FindingRecord(is_available=False)


def _empty_audit() -> AuditRecord:
    return AuditRecord(is_available=False)


def _empty_capas() -> list[CAPARecord]:
    return []


def _empty_trainings() -> list[TrainingRecord]:
    return []


def _empty_findings() -> list[PreviousFinding]:
    return []


def _empty_equipment() -> EquipmentRecord:
    return EquipmentRecord(is_available=False)


def _empty_dept() -> DepartmentInfo:
    return DepartmentInfo(is_available=False)


class AspNetLQMSClient:
    """Thin HTTP client wrapping the ASP.NET LQMS REST API.

    Every method follows the same pattern:
      1. Check LQMS_ASPNET_BASE_URL — if empty, log and return empty result.
      2. Build the GET request.
      3. On any HTTP/network error, log a warning and return empty result.
      4. Parse the response JSON into a typed model.
      5. Return the typed model with is_available=True.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        key = self._settings.lqms_aspnet_api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _base(self) -> str | None:
        url = self._settings.lqms_aspnet_base_url.rstrip("/")
        return url if url else None

    async def _get(self, path: str) -> dict | list | None:
        mode = getattr(self._settings, "lqms_mode", "production")
        if mode.lower() == "mock":
            logger.info("LQMS_MODE=mock active — tool call using safe mock records (DEMO DATA — NOT ORGANIZATIONAL DATA): %s", path)
            return self._get_mock_data(path)

        base = self._base()
        if not base:
            logger.warning("LQMS_ASPNET_BASE_URL not configured — tool call skipped: %s", path)
            return None
        url = f"{base}{path}"
        try:
            async with httpx.AsyncClient(timeout=self._settings.agent_tool_timeout_seconds) as client:
                resp = await client.get(url, headers=self._headers())
                if resp.status_code == 200:
                    return resp.json()
                logger.warning("LQMS HTTP %s on %s", resp.status_code, path)
                return None
        except Exception as exc:
            logger.warning("LQMS tool network error on %s: %s", path, exc)
            return None

    def _get_mock_data(self, path: str) -> dict | list | None:
        """Safe mock data fallback clearly tagged as DEMO DATA — NOT ORGANIZATIONAL DATA."""
        if "training" in path:
            return [{
                "record_id": "TR-DEMO-01",
                "staff_role": "Operator (DEMO DATA)",
                "department": "Laboratory",
                "training_title": "SOP-OPS-014 Retraining (DEMO)",
                "training_date": "2026-01-15",
                "document_reference": "SOP-OPS-014",
                "completed": False,
                "is_available": True
            }]
        if "equipment" in path:
            return {
                "equipment_id": "EQ-DEMO-01",
                "name": "Analytical Balance (DEMO DATA)",
                "department": "Laboratory",
                "last_calibration_date": "2025-10-01",
                "next_calibration_due": "2026-04-01",
                "calibration_status": "Overdue (DEMO)",
                "is_available": True
            }
        return None

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def get_finding(self, finding_id: str) -> FindingRecord:
        """Fetch a single finding/CA record by ID."""
        data = await self._get(f"/findings/{finding_id}")
        if not data or not isinstance(data, dict):
            return _empty_finding()
        try:
            return FindingRecord(**data, is_available=True)
        except Exception as exc:
            logger.warning("Failed to parse FindingRecord: %s", exc)
            return _empty_finding()

    async def get_audit(self, audit_id: str) -> AuditRecord:
        """Fetch audit metadata by audit ID."""
        data = await self._get(f"/audits/{audit_id}")
        if not data or not isinstance(data, dict):
            return _empty_audit()
        try:
            return AuditRecord(**data, is_available=True)
        except Exception as exc:
            logger.warning("Failed to parse AuditRecord: %s", exc)
            return _empty_audit()

    async def get_related_capa(self, finding_id: str) -> list[CAPARecord]:
        """Fetch all CAPAs linked to a finding."""
        data = await self._get(f"/findings/{finding_id}/capas")
        if not data or not isinstance(data, list):
            return _empty_capas()
        results = []
        for item in data:
            try:
                results.append(CAPARecord(**item, is_available=True))
            except Exception as exc:
                logger.warning("Failed to parse CAPARecord item: %s", exc)
        return results

    async def get_capa_history(self, department: str) -> list[CAPARecord]:
        """Fetch CAPA history for a department."""
        data = await self._get(f"/capas?department={department}")
        if not data or not isinstance(data, list):
            return _empty_capas()
        results = []
        for item in data:
            try:
                results.append(CAPARecord(**item, is_available=True))
            except Exception as exc:
                logger.warning("Failed to parse CAPARecord item: %s", exc)
        return results

    async def get_previous_findings(
        self, department: str, clause: str | None = None
    ) -> list[PreviousFinding]:
        """Fetch previous findings for a department, optionally filtered by clause."""
        path = f"/findings?department={department}"
        if clause:
            path += f"&clause={clause}"
        data = await self._get(path)
        if not data or not isinstance(data, list):
            return _empty_findings()
        results = []
        for item in data:
            try:
                results.append(PreviousFinding(**item, is_available=True))
            except Exception as exc:
                logger.warning("Failed to parse PreviousFinding item: %s", exc)
        return results

    async def get_audit_history(self, audit_id: str) -> list[AuditRecord]:
        """Fetch all audits related to an audit (e.g. previous cycles)."""
        data = await self._get(f"/audits/{audit_id}/history")
        if not data or not isinstance(data, list):
            return []
        results = []
        for item in data:
            try:
                results.append(AuditRecord(**item, is_available=True))
            except Exception as exc:
                logger.warning("Failed to parse AuditRecord item: %s", exc)
        return results

    async def get_equipment(self, equipment_id: str) -> EquipmentRecord:
        """Fetch equipment/calibration record by equipment ID."""
        data = await self._get(f"/equipment/{equipment_id}")
        if not data or not isinstance(data, dict):
            return _empty_equipment()
        try:
            return EquipmentRecord(**data, is_available=True)
        except Exception as exc:
            logger.warning("Failed to parse EquipmentRecord: %s", exc)
            return _empty_equipment()

    async def get_training_record(
        self, department: str, staff_role: str | None = None
    ) -> list[TrainingRecord]:
        """Fetch training records for a department, optionally filtered by staff role."""
        path = f"/training?department={department}"
        if staff_role:
            path += f"&role={staff_role}"
        data = await self._get(path)
        if not data or not isinstance(data, list):
            return _empty_trainings()
        results = []
        for item in data:
            try:
                results.append(TrainingRecord(**item, is_available=True))
            except Exception as exc:
                logger.warning("Failed to parse TrainingRecord item: %s", exc)
        return results

    async def get_department_information(self, department: str) -> DepartmentInfo:
        """Fetch department profile and metadata."""
        data = await self._get(f"/departments/{department}")
        if not data or not isinstance(data, dict):
            return _empty_dept()
        try:
            return DepartmentInfo(**data, is_available=True)
        except Exception as exc:
            logger.warning("Failed to parse DepartmentInfo: %s", exc)
            return _empty_dept()
