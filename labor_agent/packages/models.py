from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class LaborPosting(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    date: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    employment_type: Optional[str] = None
    seniority: Optional[str] = None
    skills: List[str] = Field(default_factory=list)


class LaborData(BaseModel):
    job_postings: List[LaborPosting] = Field(default_factory=list)


class LaborAnalysisOutput(BaseModel):
    market_summary: str = ""
    role_demand: Dict[str, int] = Field(default_factory=dict)
    top_skills_by_role: Dict[str, List[str]] = Field(default_factory=dict)
    overall_top_skills: List[str] = Field(default_factory=list)
    hiring_trends: List[str] = Field(default_factory=list)
    recommended_focus: List[str] = Field(default_factory=list)


class MissingInfo(BaseModel):
    missing_fields: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)


class MarketPlan(BaseModel):
    short_term: List[str] = Field(default_factory=list)
    mid_term: List[str] = Field(default_factory=list)
    long_term: List[str] = Field(default_factory=list)
    weekly_schedule: List[str] = Field(default_factory=list)
    resources: List[str] = Field(default_factory=list)


class InferenceOutput(BaseModel):
    extracted: LaborData
    analysis: LaborAnalysisOutput
    missing_info: MissingInfo
    learning_plan: MarketPlan
