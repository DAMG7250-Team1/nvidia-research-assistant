# backend/state.py
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
class AgentAction:
    """Action returned by agent."""
    def __init__(self, tool: str, tool_input: Dict, log: str = ""):
        self.tool = tool
        self.tool_input = tool_input
        self.log = log

class ResearchState(TypedDict):
    """State for the Research Agent."""
    input: str  # User's query
    chat_history: List  # Conversation history
    intermediate_steps: List[AgentAction]  # Results from agent actions
    metadata_filters: Optional[Dict]  # Optional year/quarter filters
    mode: str  # "pinecone", "web_search", "snowflake", or "combined"