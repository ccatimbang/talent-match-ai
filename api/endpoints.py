from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException
from io import BytesIO

from graph import create_talent_match_graph
from models import GraphState, MatchResult

# Create router
router = APIRouter()

# Create workflow graph
graph = create_talent_match_graph()

@router.post("/match/resume", response_model=List[MatchResult])
async def match_resume(resume: UploadFile = File(...)):
    """
    Upload a resume PDF and get matching job recommendations
    """
    try:
        # Validate file type
        if not resume.content_type or 'pdf' not in resume.content_type.lower():
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload a PDF file."
            )
        
        # Read resume file
        content = await resume.read()
        if not content:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Initialize graph state with raw PDF bytes
        state = GraphState(
            resume_text=content
        )
        
        # Run workflow
        final_state = await graph.ainvoke(state)
        
        if final_state.error:
            raise HTTPException(status_code=400, detail=final_state.error)
            
        if not final_state.job_matches:
            raise HTTPException(status_code=404, detail="No matching jobs found")
            
        return final_state.job_matches
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process resume: {str(e)}"
        ) 