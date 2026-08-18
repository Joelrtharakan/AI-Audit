from pydantic import BaseModel, Field, field_validator


class AnalyzeFindingRequest(BaseModel):
    finding_text: str = Field(..., min_length=1)
    department: str = ""
    branch: str = ""
    standard: str = ""
    clause: str = ""
    finding_type: str = ""
    nature_of_nc: str = ""
    risk_severity: str = ""
    risk_likelihood: str = ""
    risk_result: str = ""
    llm_provider: str = ""

    @field_validator("finding_text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("finding_text must not be blank")
        return v.strip()
