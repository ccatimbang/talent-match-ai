from typing import Annotated
import json
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from models import GraphState, MatchStatus
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

QA_PROMPT = """
Review the following candidate-job match and validate the matching decision.
Provide a detailed analysis considering:
1. Skills alignment (required vs. nice-to-have)
2. Experience level match
3. Career trajectory alignment
4. Potential red flags or gaps

Candidate Profile:
{candidate_profile}

Job Match:
{job_match}

Initial Match Score: {confidence_score}
Initial Reasoning: {reasoning}

Return a JSON object with:
{
    "validated_score": float,  # Your assessed match score (0.0-1.0)
    "detailed_analysis": str,  # Detailed explanation of your assessment
    "recommendation": str,     # "proceed" or "review" or "reject"
    "key_strengths": list,    # Top 3 strengths for this match
    "key_gaps": list         # Top 3 gaps or concerns
}
"""


class QAAgent:
    def __init__(self, model: Annotated[ChatOpenAI, "OpenAI model for QA"] = None):
        self.model = model or ChatOpenAI(model="gpt-4-turbo-preview")

    def format_candidate_profile(self, state: GraphState) -> str:
        """Format candidate profile for QA review"""
        profile = state.candidate_profile
        return f"""
        Name: {profile.name}
        Title: {profile.title}
        Experience: {profile.experience_years} years
        Skills: {', '.join(f'{s.name} ({s.level})' for s in profile.skills)}
        Summary: {profile.summary or 'Not provided'}
        Education: {', '.join(profile.education or ['Not provided'])}
        """

    def format_job_match(self, match) -> str:
        """Format job match for QA review"""
        job = match.matched_job
        return f"""
        Title: {job.title}
        Required Skills: {', '.join(f'{s.name} ({s.level})' for s in job.required_skills)}
        Preferred Skills: {', '.join(f'{s.name} ({s.level})' for s in (job.preferred_skills or []))}
        Min Experience: {job.min_experience_years} years
        Description: {job.description}
        """

    async def __call__(self, state: GraphState) -> GraphState:
        """LangGraph node implementation"""
        if state.error or not state.job_matches:
            return state

        try:
            # Review each match
            for match in state.job_matches:
                messages = [
                    HumanMessage(content=QA_PROMPT.format(
                        candidate_profile=self.format_candidate_profile(state),
                        job_match=self.format_job_match(match),
                        confidence_score=match.confidence_score,
                        reasoning=match.reasoning
                    ))
                ]
                
                response = await self.model.ainvoke(messages)
                analysis = json.loads(response.content)
                
                # Update match based on QA review
                match.confidence_score = float(analysis["validated_score"])
                match.reasoning = analysis["detailed_analysis"]
                
                # Update status based on recommendation
                if analysis["recommendation"] == "proceed":
                    match.status = MatchStatus.AUTO_MATCHED
                elif analysis["recommendation"] == "review":
                    match.status = MatchStatus.RECRUITER_REVIEW
                else:
                    match.status = MatchStatus.REJECTED
            
            state.current_step = "complete"
            return state
            
        except Exception as e:
            state.error = f"Failed to perform QA review: {str(e)}"
            return state 