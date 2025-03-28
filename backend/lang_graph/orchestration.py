from langgraph.graph import StateGraph, END
from state import ResearchState
from graph_functions import run_oracle, router, rag_search, web_search, generate_final_answer, snowflake_search

# Global variable to store the compiled graph
_GLOBAL_GRAPH = None

def initialize_research_graph():
    """
    Initialize the research graph once and store it in memory.
    Returns the compiled graph instance.
    """
    global _GLOBAL_GRAPH
    
    if _GLOBAL_GRAPH is None:
        print("Initializing new research graph...")
        # Create a new graph
        graph = StateGraph(ResearchState)
        
        # Add nodes with lambda wrappers
        graph.add_node("oracle", lambda x: run_oracle(x))
        graph.add_node("rag_search", lambda x: rag_search(x))
        graph.add_node("web_search", lambda x: web_search(x))
        graph.add_node("snowflake_search", lambda x: snowflake_search(x))
        graph.add_node("final_answer", lambda x: generate_final_answer(x))
        
        # Set the entry point
        graph.set_entry_point("oracle")
        
        # Add conditional edges
        graph.add_conditional_edges(
            "oracle",
            router,
            {
                "rag_search": "rag_search",
                "web_search": "web_search",
                "snowflake_search": "snowflake_search",
                "final_answer": "final_answer"
            }
        )
        
        # Add edges from search nodes back to oracle
        graph.add_edge("rag_search", "oracle")
        graph.add_edge("web_search", "oracle")
        graph.add_edge("snowflake_search", "oracle")
        graph.add_edge("final_answer", END)
        
        # Compile and store the graph
        _GLOBAL_GRAPH = graph.compile()
        print("Research graph initialized and compiled successfully")
    
    return _GLOBAL_GRAPH

def run_research_graph(query, year_quarter_dict=None, mode="combined"):
    """
    Run the research workflow using the existing graph instance.
    """
    print("\n" + "#"*100)
    print(f"📊 STARTING RESEARCH GRAPH EXECUTION 📊")
    print("#"*100)
    print(f"Query: \"{query}\"")
    print(f"Mode: {mode}")
    print(f"Year/quarter filters: {year_quarter_dict}")
    print("#"*100 + "\n")
    
    # Initialize the state for this query
    state = {
        "input": query,
        "chat_history": [],
        "intermediate_steps": [],
        "metadata_filters": year_quarter_dict or {},
        "mode": mode
    }
    
    # Get the existing graph instance
    graph = initialize_research_graph()
    print(f"Using existing graph with nodes: {list(graph.nodes.keys())}")
    result = graph.invoke(state)
    
    print("\n" + "#"*100)
    print("📋 GRAPH EXECUTION COMPLETED")
    print("#"*100 + "\n")
    
    # Extract result
    if "output" in result:
        print("Using output field from result")
        return result["output"]
    
    # Fallback to intermediate steps
    try:
        if "intermediate_steps" in result and result["intermediate_steps"]:
            for step in result["intermediate_steps"]:
                if hasattr(step, 'tool') and step.tool == "final_answer_result":
                    print("Using final_answer_result from intermediate_steps")
                    return step.log
    except Exception as e:
        print(f"Error extracting from intermediate_steps: {e}")
    
    print("No result found, returning fallback message")
    return "No comprehensive results available. Please try again with a different query."