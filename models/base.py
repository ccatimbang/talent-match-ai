from typing import List, Optional
from pydantic import BaseModel

class Skill(BaseModel):
    name: str
    level: Optional[str] = None
    years: Optional[float] = None

class CandidateProfile(BaseModel):
    name: str
    title: str
    skills: List[Skill]
    experience_years: Optional[float] = None
    education: Optional[List[str]] = None
    summary: Optional[str] = None 