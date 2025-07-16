from typing import Optional, List, Union
from pydantic import BaseModel
from .base import CandidateProfile
from .matching import MatchResult

class GraphState(BaseModel):
    """State object passed between LangGraph nodes"""
    resume_text: Optional[Union[str, bytes]] = None
    candidate_profile: Optional[CandidateProfile] = None
    job_matches: Optional[List[MatchResult]] = None
    current_step: str = "start"
    error: Optional[str] = None 