from typing import TypeVar, Dict, Any
from langgraph.graph import StateGraph, END
from agents.ingest import IngestAgent
from agents.extract import ExtractAgent
from agents.classify import ClassifyAgent
from agents.match import MatchAgent
from agents.qa import QAAgent
from models import GraphState

# Type variable for the graph state
S = TypeVar("S", bound=GraphState)

def create_talent_match_graph() -> StateGraph:
    """Create the talent matching workflow graph"""
    
    # Initialize workflow graph
    workflow = StateGraph(state_schema=GraphState)
    
    # Add nodes
    workflow.add_node("ingest", IngestAgent())
    workflow.add_node("extract", ExtractAgent())
    workflow.add_node("classify", ClassifyAgent())
    workflow.add_node("match", MatchAgent())
    workflow.add_node("qa", QAAgent())
    
    # Define edges
    workflow.add_edge("ingest", "extract")
    workflow.add_edge("extract", "classify")
    workflow.add_edge("classify", "match")
    workflow.add_edge("match", "qa")
    
    # Set entry point
    workflow.set_entry_point("ingest")
    
    # Define conditional routing function
    def should_continue(x: GraphState) -> str:
        if x.error or x.current_step == "complete":
            return "error"
        return "complete"
    
    # Add conditional end
    workflow.add_conditional_edges(
        "qa",
        should_continue,
        {
            "complete": END,
            "error": END
        }
    )
    
    # Compile graph
    return workflow.compile() 