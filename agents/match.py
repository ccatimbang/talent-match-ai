from typing import Annotated, List
import json
import numpy as np
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import faiss
from models import GraphState, JobPosting, MatchResult, MatchStatus
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Sample job catalog - in production this would come from a database
SAMPLE_JOBS = [
    JobPosting(
        id="job1",
        title="Senior Backend Engineer",
        required_skills=[{"name": "Python", "level": "expert"}, {"name": "PostgreSQL", "level": "intermediate"}],
        preferred_skills=[{"name": "AWS", "level": "intermediate"}],
        min_experience_years=5,
        description="Build scalable backend services..."
    ),
    JobPosting(
        id="job2",
        title="ML Engineer",
        required_skills=[{"name": "Python", "level": "expert"}, {"name": "PyTorch", "level": "intermediate"}],
        preferred_skills=[{"name": "AWS", "level": "intermediate"}],
        min_experience_years=3,
        description="Develop ML models..."
    )
]

class MatchAgent:
    def __init__(
        self,
        model: Annotated[ChatOpenAI, "OpenAI model for matching"] = None,
        embeddings: Annotated[OpenAIEmbeddings, "OpenAI embeddings model"] = None
    ):
        self.model = model or ChatOpenAI(model="gpt-4-turbo-preview")
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.jobs = SAMPLE_JOBS
        self._init_index()

    def _init_index(self):
        """Initialize FAISS index with job embeddings"""
        self.job_texts = [
            f"{job.title}\n{job.description}\nRequired: {', '.join(s.name for s in job.required_skills)}"
            for job in self.jobs
        ]
        
        embeddings = self.embeddings.embed_documents(self.job_texts)
        dimension = len(embeddings[0])
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype=np.float32))

    async def get_matches(self, state: GraphState, top_k: int = 3) -> List[MatchResult]:
        """Find top job matches for a candidate"""
        profile = state.candidate_profile
        
        # Create candidate embedding
        candidate_text = f"{profile.title}\n{profile.summary or ''}\nSkills: {', '.join(s.name for s in profile.skills)}"
        candidate_embedding = self.embeddings.embed_query(candidate_text)
        
        # Search similar jobs
        D, I = self.index.search(np.array([candidate_embedding], dtype=np.float32), top_k)
        
        matches = []
        for idx in I[0]:
            job = self.jobs[idx]
            
            # Get detailed match analysis from LLM
            messages = [
                HumanMessage(content=f"""
                Analyze the match between this candidate and job. Return a JSON with:
                - confidence_score: 0.0-1.0 based on skill and experience match
                - reasoning: Brief explanation of the match quality
                
                Candidate:
                - Title: {profile.title}
                - Experience: {profile.experience_years} years
                - Skills: {', '.join(f'{s.name} ({s.level})' for s in profile.skills)}
                
                Job:
                - Title: {job.title}
                - Required Skills: {', '.join(f'{s.name} ({s.level})' for s in job.required_skills)}
                - Preferred Skills: {', '.join(f'{s.name} ({s.level})' for s in (job.preferred_skills or []))}
                - Min Experience: {job.min_experience_years} years
                """)
            ]
            
            response = await self.model.ainvoke(messages)
            analysis = json.loads(response.content)
            
            match = MatchResult(
                candidate_profile=profile,
                matched_job=job,
                confidence_score=float(analysis["confidence_score"]),
                reasoning=analysis["reasoning"],
                status=MatchStatus.AUTO_MATCHED if analysis["confidence_score"] >= 0.8 else MatchStatus.RECRUITER_REVIEW
            )
            matches.append(match)
            
        return matches

    async def __call__(self, state: GraphState) -> GraphState:
        """LangGraph node implementation"""
        if state.error or not state.candidate_profile:
            return state

        try:
            matches = await self.get_matches(state)
            state.job_matches = matches
            state.current_step = "qa"
            return state
            
        except Exception as e:
            state.error = f"Failed to find job matches: {str(e)}"
            return state 