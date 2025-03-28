from typing import TypedDict, List, Dict, Any, Optional, Union
from pydantic import BaseModel

class AgentAction(BaseModel):
    """Model for agent actions in the research graph."""
    tool: str
    tool_input: Dict[str, Any]
    log: str

class ResearchRequest(BaseModel):
    """Request model for research queries"""
    query: str
    year: int
    quarter: str
    agent_type: str = "combined"
    metadata_filters: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What were Nvidia's breakthroughs in Q3 2022?",
                "year": 2022,
                "quarter": "Q3",
                "agent_type": "combined",
                "metadata_filters": None
            }
        }

class ResearchState(BaseModel):
    """State model for the research graph."""
    input: str
    chat_history: List[Dict[str, str]] = []
    intermediate_steps: List[AgentAction] = []
    metadata_filters: Dict[str, Any] = {}
    mode: str = "combined"
    output: Optional[str] = None
    query: str
    year: str
    quarter: str
    agent_type: str
    snowflake_results: Dict[str, Any] = {}
    rag_results: Dict[str, Any] = {}
    web_results: Dict[str, Any] = {}
    final_report: str = ""
    next_node: Optional[str] = None
    agent_sequence: List[str] = []

def create_initial_state(request: ResearchRequest) -> ResearchState:
    """Create initial state from a ResearchRequest"""
    return ResearchState(
        input=request.query,
        query=request.query,
        year=str(request.year),
        quarter=request.quarter,
        agent_type=request.agent_type,
        metadata_filters=request.metadata_filters or {},
        snowflake_results={},
        rag_results={},
        web_results={},
        final_report="",
        agent_sequence=[],
        chat_history=[],
        intermediate_steps=[],
        mode=request.agent_type,
        output=None,
        next_node=None
    ) 