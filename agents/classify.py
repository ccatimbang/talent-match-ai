from typing import Annotated, List, Dict
import json
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from models import GraphState, Skill
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CLASSIFICATION_PROMPT = """
Analyze and enrich the following list of technical skills. For each skill:
1. Determine the skill level based on context (beginner/intermediate/expert)
2. Estimate years of experience if not provided
3. Categorize into domains (e.g., Frontend, Backend, DevOps, AI/ML, etc.)

Skills to analyze:
{skills}

Return a JSON object with the enriched skills data. Example format:
{
    "skills": [
        {
            "name": "Python",
            "level": "expert",
            "years": 5,
            "domain": "Backend"
        }
    ]
}

Base your analysis on the candidate's experience, projects, and overall profile:
{profile_summary}
"""


class ClassifyAgent:
    def __init__(self, model: Annotated[ChatOpenAI, "OpenAI model for classification"] = None):
        self.model = model or ChatOpenAI(model="gpt-4-turbo-preview")

    def format_skills_for_prompt(self, skills: List[Skill]) -> str:
        """Format skills list for the prompt"""
        return "\n".join([
            f"- {skill.name}" + 
            (f" (Level: {skill.level})" if skill.level else "") +
            (f" ({skill.years} years)" if skill.years else "")
            for skill in skills
        ])

    def create_profile_summary(self, state: GraphState) -> str:
        """Create a summary of the candidate's profile for context"""
        profile = state.candidate_profile
        return f"""
        Name: {profile.name}
        Title: {profile.title}
        Total Experience: {profile.experience_years or 'Unknown'} years
        Summary: {profile.summary or 'Not provided'}
        """

    async def __call__(self, state: GraphState) -> GraphState:
        """LangGraph node implementation"""
        if state.error or not state.candidate_profile:
            return state

        try:
            # Prepare the classification prompt
            skills_text = self.format_skills_for_prompt(state.candidate_profile.skills)
            profile_summary = self.create_profile_summary(state)
            
            messages = [
                HumanMessage(content=CLASSIFICATION_PROMPT.format(
                    skills=skills_text,
                    profile_summary=profile_summary
                ))
            ]
            
            # Get enriched skills data
            response = await self.model.ainvoke(messages)
            data = json.loads(response.content)
            
            # Update the candidate profile with enriched skills
            state.candidate_profile.skills = [Skill(**skill) for skill in data["skills"]]
            state.current_step = "match"
            return state
            
        except Exception as e:
            state.error = f"Failed to classify skills: {str(e)}"
            return state 