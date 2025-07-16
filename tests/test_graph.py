import pytest
from models import GraphState, CandidateProfile, Skill
from graph import create_talent_match_graph

@pytest.fixture
def sample_resume_text():
    with open("data/resume_sample.txt", "r") as f:
        return f.read()

@pytest.fixture
def sample_state(sample_resume_text):
    return GraphState(
        resume_text=sample_resume_text
    )

@pytest.mark.asyncio
async def test_workflow_success(sample_state):
    """Test successful workflow execution"""
    graph = create_talent_match_graph()
    final_state = await graph.ainvoke(sample_state)
    
    assert not final_state.error
    assert final_state.current_step == "complete"
    assert final_state.candidate_profile is not None
    assert len(final_state.job_matches) > 0

@pytest.mark.asyncio
async def test_workflow_invalid_resume():
    """Test workflow with invalid resume"""
    graph = create_talent_match_graph()
    state = GraphState(resume_text="Not a valid resume")
    
    final_state = await graph.ainvoke(state)
    assert final_state.error is not None

@pytest.mark.asyncio
async def test_match_scoring():
    """Test match scoring and status assignment"""
    graph = create_talent_match_graph()
    
    # Create a profile that should match ML Engineer role
    state = GraphState(
        candidate_profile=CandidateProfile(
            name="Test Engineer",
            title="ML Engineer",
            skills=[
                Skill(name="Python", level="expert"),
                Skill(name="PyTorch", level="expert"),
                Skill(name="Machine Learning", level="expert")
            ],
            experience_years=6
        )
    )
    
    final_state = await graph.ainvoke(state)
    
    assert not final_state.error
    assert len(final_state.job_matches) > 0
    
    # Check if at least one match has high confidence
    high_confidence_matches = [
        m for m in final_state.job_matches 
        if m.confidence_score >= 0.8
    ]
    assert len(high_confidence_matches) > 0 