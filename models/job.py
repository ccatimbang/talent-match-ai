from typing import List, Optional
from pydantic import BaseModel
from .base import Skill

class JobPosting(BaseModel):
    id: str
    title: str
    required_skills: List[Skill]
    preferred_skills: Optional[List[Skill]] = None
    min_experience_years: Optional[float] = None
    description: str 