from pydantic import BaseModel, Field
from typing import List, Optional

class LearnerData(BaseModel):
    courses: List[str] = Field(default_factory=list)
    grades: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    research_goals: List[str] = Field(default_factory=list)
    education_level: Optional[str] = None
    certifications: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    preferred_domains: List[str] = Field(default_factory=list)
    availability: Optional[str] = None
    learning_preferences: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)


class LearnerAnalysisOutput(BaseModel):
    summarized_report: str = ""
    skill_profile: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class MissingInfo(BaseModel):
    missing_fields: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)


class LearningPlan(BaseModel):
    short_term: List[str] = Field(default_factory=list)
    mid_term: List[str] = Field(default_factory=list)
    long_term: List[str] = Field(default_factory=list)
    weekly_schedule: List[str] = Field(default_factory=list)
    resources: List[str] = Field(default_factory=list)


class InferenceOutput(BaseModel):
    extracted: LearnerData
    analysis: LearnerAnalysisOutput
    missing_info: MissingInfo
    learning_plan: LearningPlan