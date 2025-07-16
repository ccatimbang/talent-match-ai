from enum import Enum
from pydantic import BaseModel, Field
from .base import CandidateProfile
from .job import JobPosting

class MatchStatus(str, Enum):
    AUTO_MATCHED = "auto_matched"
    RECRUITER_REVIEW = "recruiter_review"
    REJECTED = "rejected"

class MatchResult(BaseModel):
    candidate_profile: CandidateProfile
    matched_job: JobPosting
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    status: MatchStatus 