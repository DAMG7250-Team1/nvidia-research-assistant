import os
import sys
from pathlib import Path
from typing import Dict, Any, TypedDict, Annotated, List, Optional, Union

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from .state import ResearchState, AgentAction, ResearchRequest, create_initial_state
import os
from dotenv import load_dotenv
from agents.rag_agent import AgenticResearchAssistant
from agents.web_agent import NVIDIAWebSearchAgent
from agents.snowflake_agent import generate_snowflake_insights, get_nvidia_historical
import re
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
import logging
from langchain.tools import tool

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the LLM
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    max_tokens=2000
)

# Initialize agents
@tool
def snowflake_tool(query: str, metadata_filters: dict) -> dict:
    """
    Tool for querying NVIDIA's financial data from Snowflake.
    
    Args:
        query (str): The user's query about NVIDIA's financial data
        metadata_filters (dict): Dictionary containing filters like year and quarter
        
    Returns:
        dict: Dictionary containing the query results and any generated visualizations
    """
    return generate_snowflake_insights(query, metadata_filters)

snowflake_agent = snowflake_tool
rag_agent = AgenticResearchAssistant()
web_agent = NVIDIAWebSearchAgent()

# Global variable to store the compiled graph
_GLOBAL_GRAPH = None

def initialize_research_graph():
    """
    Initialize the research graph once and store it in memory.
    Returns the compiled graph instance.
    """
    global _GLOBAL_GRAPH
    
    if _GLOBAL_GRAPH is None:
        logger.info("Initializing new research graph...")
        # Create a new graph
        graph = StateGraph(ResearchState)
        
        # Add nodes
        graph.add_node("route_based_on_agent", route_based_on_agent)
        graph.add_node("snowflake_node", run_snowflake_agent)
        graph.add_node("rag_node", run_rag_agent)
        graph.add_node("web_node", run_web_agent)
        graph.add_node("generate_final_report", generate_final_report)
        
        # Set the entry point
        graph.set_entry_point("route_based_on_agent")
        
        # Add conditional edges from routing node
        graph.add_conditional_edges(
            "route_based_on_agent",
            lambda x: x.next_node or "generate_final_report",
            {
                "snowflake_node": "snowflake_node",
                "rag_node": "rag_node",
                "web_node": "web_node",
                "generate_final_report": "generate_final_report"
            }
        )
        
        # Add conditional edges for sequence-based routing
        graph.add_conditional_edges(
            "snowflake_node",
            should_continue,
            {
                "rag_node": "rag_node",
                "web_node": "web_node",
                "generate_final_report": "generate_final_report",
                END: END
            }
        )
        
        graph.add_conditional_edges(
            "rag_node",
            should_continue,
            {
                "web_node": "web_node",
                "generate_final_report": "generate_final_report",
                END: END
            }
        )
        
        graph.add_conditional_edges(
            "web_node",
            should_continue,
            {
                "generate_final_report": "generate_final_report",
                END: END
            }
        )
        
        # Add edge from final report to END
        graph.add_edge("generate_final_report", END)
        
        # Compile and store the graph
        _GLOBAL_GRAPH = graph.compile()
        logger.info("Research graph initialized and compiled successfully")
    
    return _GLOBAL_GRAPH

async def run_research_graph(
    query: str,
    year: int,
    quarter: str,
    agent_type: str = "combined",
    metadata_filters: Optional[Dict[str, Any]] = None
) -> Dict:
    """Run the complete research workflow"""
    try:
        global _GLOBAL_GRAPH
        
        # Initialize research graph if not already initialized
        if _GLOBAL_GRAPH is None:
            logger.info("Initializing research graph...")
            _GLOBAL_GRAPH = initialize_research_graph()
        
        # Create initial state
        request = ResearchRequest(
            query=query,
            year=year,
            quarter=quarter,
            agent_type=agent_type,
            metadata_filters=metadata_filters
        )
        state = create_initial_state(request)
        
        # Run the graph
        logger.info(f"Running research graph for query: {query}")
        result = _GLOBAL_GRAPH.invoke(state)
        
        # Extract and format results
        rag_results = result.get("rag_results", {})
        if isinstance(rag_results, str):
            rag_results = {"answer": rag_results}
            
        snowflake_results = result.get("snowflake_results", {})
        if isinstance(snowflake_results, str):
            snowflake_results = {"summary": snowflake_results}
            
        web_results = result.get("web_results", {})
        if isinstance(web_results, str):
            web_results = {"answer": web_results}
        
        # Return formatted response
        return {
            "status": "success",
            "data": {
                "final_report": result.get("final_report", ""),
                "rag_results": rag_results,
                "snowflake_results": snowflake_results,
                "web_results": web_results
            }
        }
        
    except Exception as e:
        logger.error(f"Error in research workflow: {str(e)}")
        logger.error("Full error details:", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "data": {
                "final_report": "",
                "rag_results": {"error": str(e)},
                "snowflake_results": {},
                "web_results": {}
            }
        }

def route_based_on_agent(state: ResearchState) -> ResearchState:
    """
    Route to the appropriate node based on the agent type.
    """
    logger.info(f"Routing based on agent type: {state.agent_type}")
    
    if state.agent_type == "snowflake":
        state.next_node = "snowflake_node"
    elif state.agent_type == "rag":
        state.next_node = "rag_node"
    elif state.agent_type == "web":
        state.next_node = "web_node"
    elif state.agent_type == "combined":
        # For combined analysis, start with snowflake
        state.next_node = "snowflake_node"
        # Define the sequence for combined mode
        state.agent_sequence = ["snowflake_node", "rag_node", "web_node", "generate_final_report"]
    else:
        state.next_node = "generate_final_report"
    
    logger.info(f"Routed to node: {state.next_node}")
    return state

def run_snowflake_agent(state: ResearchState) -> ResearchState:
    """Run Snowflake agent to get historical data"""
    try:
        logger.info("Running Snowflake agent...")
        result = get_nvidia_historical(
            year=state.year,
            quarter=state.quarter
        )
        state.snowflake_results = {
            "summary": result["summary"],
            "chart_paths": result["chart_paths"]
        }
        logger.info("Snowflake agent completed successfully")
    except Exception as e:
        logger.error(f"Error in Snowflake agent: {str(e)}")
        state.snowflake_results = {"error": f"Snowflake agent error: {str(e)}"}
    return state

def run_rag_agent(state: ResearchState) -> ResearchState:
    """Run RAG agent to search Pinecone database"""
    try:
        logger.info("Running RAG agent...")
        result = rag_agent.search_pinecone_db(
            query=state.query,
            year_quarter_dict={
                "year": state.year,
                "quarter": state.quarter
            },
            chunk_strategy="markdown"
        )
        
        # Update the state with the RAG results
        if result and isinstance(result, dict):
            if result.get("status") == "success":
                state.rag_results = {
                    "answer": result.get("answer", ""),
                    "status": "success",
                    "chunks_used": result.get("chunks_used", 0),
                    "strategy": result.get("strategy", "unknown")
                }
                logger.info("RAG agent completed successfully")
            else:
                state.rag_results = {
                    "error": result.get("error", "Unknown error"),
                    "status": "error",
                    "answer": ""
                }
                logger.error(f"RAG agent error: {result.get('error', 'Unknown error')}")
        else:
            state.rag_results = {
                "error": "No results found",
                "status": "error",
                "answer": ""
            }
            logger.warning("No results found in RAG search")
            
    except Exception as e:
        logger.error(f"Error in RAG agent: {str(e)}")
        state.rag_results = {
            "error": str(e),
            "status": "error",
            "answer": ""
        }
    return state

def run_web_agent(state: ResearchState) -> ResearchState:
    """Run Web agent to get real-time data"""
    try:
        logger.info("Running Web agent...")
        
        # Initialize web agent
        web_agent = NVIDIAWebSearchAgent()
        
        # Generate comprehensive report
        result = web_agent.generate_comprehensive_research_report(state.query)
        
        if result["status"] == "success":
            logger.info("Web agent completed successfully")
            # Format web results for the state
            web_results = {
                "answer": result.get("analysis", ""),
                "results": result.get("results", []),
                "status": "success"
            }
            if "visualizations" in result:
                web_results["visualizations"] = result["visualizations"]
        else:
            logger.error(f"Web agent error: {result.get('message', 'Unknown error')}")
            web_results = {
                "error": result.get("message", "Unknown error in web search"),
                "status": "error"
            }
        
        state.web_results = web_results
        
    except Exception as e:
        logger.error(f"Error in Web agent: {str(e)}")
        state.web_results = {
            "error": f"Web agent error: {str(e)}",
            "status": "error"
        }
    return state

def generate_final_report(state: ResearchState) -> ResearchState:
    """Generate final report combining results from all agents"""
    try:
        logger.info("Generating final report...")
        
        # Prepare context from all agents
        context = []
        
        # Add RAG results if available
        if state.rag_results and "answer" in state.rag_results:
            context.append(f"RAG Results:\n{state.rag_results['answer']}")
            
        # Add Snowflake results if available
        if state.snowflake_results:
            if "summary" in state.snowflake_results:
                context.append(f"Snowflake Results:\n{state.snowflake_results['summary']}")
            # Add chart download links if available
            if "chart_paths" in state.snowflake_results:
                chart_section = "\n## Charts and Visualizations\n"
                chart_section += "The following charts are available for download:\n\n"
                for metric, path in state.snowflake_results["chart_paths"].items():
                    if path:  # Only add if path exists
                        chart_section += f"- [{metric} Trend Chart]({path})\n"
                context.append(chart_section)
            
        # Add Web results if available
        if state.web_results:
            if "answer" in state.web_results:
                context.append(f"Web Results:\n{state.web_results['answer']}")
            if "results" in state.web_results:
                web_section = "\n## Web Search Results\n"
                for result in state.web_results["results"]:
                    web_section += f"### {result.get('title', 'Result')}\n"
                    web_section += f"Link: {result.get('link', 'No link available')}\n"
                    web_section += f"Snippet: {result.get('snippet', 'No snippet available')}\n\n"
                context.append(web_section)
        
        # Generate final report using LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analyst. Analyze the following information and provide a comprehensive report.
            
            Format your response with the following structure:
            1. Executive Summary
            2. Key Findings
            3. Detailed Analysis
            4. Recommendations
            
            For any numerical data or comparisons, present them in well-formatted markdown tables.
            Use clear section headers with ## for main sections and ### for subsections.
            Include bullet points for key points and numbered lists for sequential information.
            Ensure all tables have clear headers and proper alignment.
            
            Make the report easy to read and understand, with proper spacing and formatting."""),
            ("user", f"""Based on the following query and context, generate a comprehensive research report:

            Query: {state.query}

            Context:
            {chr(10).join(context)}

            Please provide a comprehensive analysis with:
            1. Clear section headers
            2. Well-formatted tables for any numerical data
            3. Bullet points for key findings
            4. Proper markdown formatting
            5. Executive summary at the beginning
            6. Clear recommendations at the end""")
        ])
        
        chain = prompt | llm
        response = chain.invoke({})
        
        # Update the state with the final report
        state.final_report = response.content
        logger.info("Final report generated successfully")
        
    except Exception as e:
        logger.error(f"Error generating final report: {str(e)}")
        state.final_report = f"Error generating report: {str(e)}"
    
    return state

def should_continue(state: ResearchState) -> str:
    """
    Determine if the workflow should continue based on the current state.
    Returns the next node to execute or END if the workflow should terminate.
    """
    try:
        # Check for errors in the current node's results
        current_node = state.next_node
        if not current_node:
            logger.warning("No current node found in state")
            return "generate_final_report"
            
        # Get results for the current node
        results = getattr(state, f"{current_node}_results", {})
        if not results:
            logger.warning(f"No results found for node {current_node}")
            return "generate_final_report"
            
        # Check for errors in the results
        if isinstance(results, dict) and "error" in results:
            logger.error(f"Error found in {current_node} results: {results['error']}")
            return "generate_final_report"
                
        # Handle sequence-based routing for combined mode
        if state.mode == "combined":
            sequence = ["snowflake_node", "rag_node", "web_node", "generate_final_report"]
            current_index = sequence.index(current_node)
            
            if current_index < len(sequence) - 1:
                next_node = sequence[current_index + 1]
                logger.info(f"Moving to next node in sequence: {next_node}")
                return next_node
            else:
                logger.info("Sequence complete, generating final report")
                return "generate_final_report"
                
        # For single mode, go directly to final report
        logger.info("Single mode detected, generating final report")
        return "generate_final_report"
        
    except Exception as e:
        logger.error(f"Error in should_continue: {str(e)}")
        return "generate_final_report"

def run_oracle(state: ResearchState):
    """
    Decides which tool to use based on the query and selected mode.
    """
    print("\n" + "="*80)
    print("🔮 ORACLE NODE: Deciding which tool to use")
    print("="*80)
    print(f"Query: \"{state.input}\"")
    print(f"Mode: {state.mode}")
    
    # Get list of tools we've already used
    used_tools = [step.tool for step in state.intermediate_steps]
    print(f"Tools used so far: {used_tools}")
    
    # For individual modes, use the selected tool once and then move to final answer
    if state.mode in ["pinecone", "web_search", "snowflake"]:
        if not used_tools:
            chosen_tool = state.mode
        else:
            chosen_tool = "final_answer"
    
    # For combined mode
    elif state.mode == "combined":
        all_tools = {"web_search", "pinecone", "snowflake"}
        used_real_tools = {tool for tool in used_tools if tool in all_tools}
        
        if not used_tools:
            # First tool selection via LLM
            oracle_prompt = ChatPromptTemplate.from_template(
                """You are an intelligent assistant that decides which tool to use first to answer a user's query about NVIDIA.
                
                Available tools:
                - pinecone: Search NVIDIA quarterly reports and financial data (use for historical financial analysis, earnings, revenue, etc.)
                - web_search: Search recent news and web content (use for current events, market updates, new products, etc.)
                - snowflake: Query financial metrics and stock performance data (use for stock performance, technical indicators, price trends)
                
                The user's query is: {input}
                
                Consider:
                1. Which data source would be most relevant to start with?
                2. For financial history and earnings, use pinecone
                3. For recent events and news, use web_search
                4. For stock performance and metrics, use snowflake
                
                Reply with just one tool name: "pinecone", "web_search", or "snowflake"."""
            )
            
            chosen_tool = llm.invoke(
                oracle_prompt.format(input=state.input)
            ).content.strip().lower()
            print(f"LLM chose initial tool: {chosen_tool}")
            
        elif len(used_real_tools) < 3:
            # Use remaining tools in a strategic order
            unused_tools = all_tools - used_real_tools
            print(f"Unused tools: {unused_tools}")
            
            # Let LLM choose from remaining tools
            next_tool_prompt = ChatPromptTemplate.from_template(
                """Based on the query: "{input}"
                And having already used these tools: {used_tools}
                Which of these remaining tools would be most valuable next: {unused_tools}?
                
                Consider:
                1. What information gaps still exist
                2. Which tool would best complement our current data
                3. How to build a comprehensive answer
                
                Reply with just one tool name from the unused tools."""
            )
            
            chosen_tool = llm.invoke(
                next_tool_prompt.format(
                    input=state.input,
                    used_tools=list(used_real_tools),
                    unused_tools=list(unused_tools)
                )
            ).content.strip().lower()
            
            # Validate chosen tool is unused
            if chosen_tool not in unused_tools:
                chosen_tool = next(iter(unused_tools))
            
            print(f"Selected next tool: {chosen_tool}")
        else:
            print("All tools used - generating final answer")
            chosen_tool = "final_answer"
    
    # Normalize the response
    if "pinecone" in chosen_tool:
        chosen_tool = "pinecone"
    elif "web" in chosen_tool:
        chosen_tool = "web_search"
    elif "snow" in chosen_tool:
        chosen_tool = "snowflake"
    else:
        chosen_tool = "final_answer"
    
    print(f"🎯 ORACLE DECISION: {chosen_tool}")
    print("="*80 + "\n")
    
    # Create the agent action
    action = AgentAction(
        tool=chosen_tool,
        tool_input={
            "query": state.input,
            "metadata_filters": state.metadata_filters
        },
        log=f"Selected {chosen_tool} based on mode: {state.mode}"
    )
    
    # Create new state with updated steps
    return ResearchState(
        input=state.input,
        chat_history=state.chat_history,
        intermediate_steps=state.intermediate_steps + [action],
        metadata_filters=state.metadata_filters,
        mode=state.mode,
        output=state.output,
        query=state.query,
        year=state.year,
        quarter=state.quarter,
        agent_type=state.agent_type,
        snowflake_results=state.snowflake_results,
        rag_results=state.rag_results,
        web_results=state.web_results,
        final_report=state.final_report,
        next_node=state.next_node,
        agent_sequence=state.agent_sequence
    )

def router(state: ResearchState):
    """
    Routes to the next node based on the chosen tool.
    """
    print("\n" + "-"*80)
    print("🧭 ROUTER: Determining next node")
    print("-"*80)
    
    # Get the most recent action's tool
    if state.intermediate_steps:
        latest_tool = state.intermediate_steps[-1].tool
        print(f"Latest tool selected: {latest_tool}")
        
        # Map tool to node name
        if latest_tool == "pinecone":
            return "rag_node"
        elif latest_tool == "web_search":
            return "web_node"
        elif latest_tool == "snowflake":
            return "snowflake_node"
        elif latest_tool == "final_answer":
            return "generate_final_report"
    
    print("No tool selected - defaulting to final answer")
    return "generate_final_report"

def rag_search(state: ResearchState):
    """
    Perform RAG search using Pinecone.
    """
    print("\n" + "="*80)
    print("📚 RAG SEARCH: Searching NVIDIA reports")
    print("="*80)
    
    try:
        # Initialize RAG assistant
        assistant = AgenticResearchAssistant()
        
        # Get the latest action
        action = state.intermediate_steps[-1]
        
        # Perform search
        result = assistant.search_pinecone_db(
            query=action.tool_input["query"],
            year_quarter_dict=action.tool_input["metadata_filters"],
            chunk_strategy="markdown"
        )
        
        print(f"✅ RAG search completed")
        
        # Update chat history
        chat_entry = {
            "role": "assistant",
            "content": f"RAG Search Result: {result}"
        }
        
        # Return updated state
        return ResearchState(
            input=state.input,
            chat_history=state.chat_history + [chat_entry],
            intermediate_steps=state.intermediate_steps,
            metadata_filters=state.metadata_filters,
            mode=state.mode,
            output=state.output
        )
        
    except Exception as e:
        print(f"❌ Error in RAG search: {str(e)}")
        error_msg = f"Error performing RAG search: {str(e)}"
        
        return ResearchState(
            input=state.input,
            chat_history=state.chat_history + [{"role": "error", "content": error_msg}],
            intermediate_steps=state.intermediate_steps,
            metadata_filters=state.metadata_filters,
            mode=state.mode,
            output=error_msg
        )

def web_search(state: ResearchState):
    """
    Perform web search for NVIDIA information.
    """
    print("\n" + "="*80)
    print("🌐 WEB SEARCH: Searching for NVIDIA information")
    print("="*80)
    
    try:
        # Initialize web agent
        agent = NVIDIAWebSearchAgent()
        
        # Get the latest action
        action = state.intermediate_steps[-1]
        query = action.tool_input["query"]
        print(f"Searching for query: {query}")
        
        # Perform search and generate report
        result = agent.generate_comprehensive_research_report(query)
        
        print(f"✅ Web search completed")
        print(f"Status: {result.get('status')}")
        print(f"Number of results: {len(result.get('results', []))}")
        
        if result.get('status') == 'error':
            error_msg = result.get('message', 'Unknown error occurred during web search')
            print(f"❌ Web search error: {error_msg}")
            return ResearchState(
                input=state.input,
                chat_history=state.chat_history + [{"role": "error", "content": error_msg}],
                intermediate_steps=state.intermediate_steps,
                metadata_filters=state.metadata_filters,
                mode=state.mode,
                output=error_msg
            )
        
        # Update chat history with the analysis
        chat_entry = {
            "role": "assistant",
            "content": result.get("analysis", "No analysis available")
        }
        
        # Add search results to intermediate steps for reference
        updated_steps = state.intermediate_steps + [
            AgentAction(
                tool="web_search",
                tool_input={"query": query, "results": result.get("results", [])},
                log=f"Web search completed with {len(result.get('results', []))} results"
            )
        ]
        
        # Return updated state
        return ResearchState(
            input=state.input,
            chat_history=state.chat_history + [chat_entry],
            intermediate_steps=updated_steps,
            metadata_filters=state.metadata_filters,
            mode=state.mode,
            output=state.output
        )
        
    except Exception as e:
        print(f"❌ Error in web search: {str(e)}")
        import traceback
        traceback.print_exc()
        error_msg = f"Error performing web search: {str(e)}"
        
        return ResearchState(
            input=state.input,
            chat_history=state.chat_history + [{"role": "error", "content": error_msg}],
            intermediate_steps=state.intermediate_steps,
            metadata_filters=state.metadata_filters,
            mode=state.mode,
            output=error_msg
        )

def snowflake_search(state: ResearchState):
    """
    Query Snowflake for NVIDIA financial data.
    """
    print("\n" + "="*80)
    print("❄️ SNOWFLAKE: Querying financial data")
    print("="*80)
    
    try:
        # Get the latest action
        action = state.intermediate_steps[-1]
        
        # Generate insights
        result = generate_snowflake_insights(
            query=action.tool_input["query"],
            metadata_filters=action.tool_input["metadata_filters"]
        )
        
        print(f"✅ Snowflake query completed")
        
        # Update chat history
        chat_entry = {
            "role": "assistant",
            "content": f"Snowflake Query Result: {result['summary']}"
        }
        
        # Return updated state
        return ResearchState(
            input=state.input,
            chat_history=state.chat_history + [chat_entry],
            intermediate_steps=state.intermediate_steps,
            metadata_filters=state.metadata_filters,
            mode=state.mode,
            output=state.output
        )
        
    except Exception as e:
        print(f"❌ Error in Snowflake query: {str(e)}")
        error_msg = f"Error querying Snowflake: {str(e)}"
        
        return ResearchState(
            input=state.input,
            chat_history=state.chat_history + [{"role": "error", "content": error_msg}],
            intermediate_steps=state.intermediate_steps,
            metadata_filters=state.metadata_filters,
            mode=state.mode,
            output=error_msg
        )

def generate_final_answer(state: ResearchState):
    """
    Generate a final comprehensive answer based on all gathered information.
    """
    print("\n" + "="*80)
    print("📝 GENERATING FINAL ANSWER")
    print("="*80)
    
    # Extract chat entries
    chat_entries = [entry for entry in state.chat_history if isinstance(entry, dict)]
    
    if not chat_entries:
        print("❌ No chat entries found")
        return {"output": "No information available to generate a response."}
    
    # Combine all gathered information
    gathered_info = "\n".join([
        f"Source {i+1}:\n{entry.get('content', '')}"
        for i, entry in enumerate(chat_entries)
    ])
    
    # Generate final answer with improved formatting
    final_prompt = f"""
    Based on the following information about NVIDIA, generate a comprehensive research report in markdown format:
    
    Query: {state.input}
    
    Gathered Information:
    {gathered_info}
    
    Please provide a well-structured report with the following sections:
    
    # Executive Summary
    [Provide a brief overview of the key findings]
    
    # Key Metrics and Data
    [Present important numbers and statistics in a clear format using tables]
    
    # Market Analysis
    [Discuss market implications and trends]
    
    # Future Outlook
    [Analyze potential future developments and opportunities]
    
    # Sources and References
    [List the sources used in the analysis]
    
    Use markdown formatting for:
    - Headers (## for subsections)
    - Tables (| Header | Value |)
    - Lists (both bullet points and numbered lists)
    - Bold text for important numbers and key points
    - Code blocks for any technical data
    - Emojis for visual appeal (e.g., 📈 for growth, 📉 for decline)
    """
    
    try:
        final_answer = llm.invoke(final_prompt)
        print("✅ Final answer generated successfully")
        return {"output": final_answer.content}
    except Exception as e:
        print(f"❌ Error generating final answer: {str(e)}")
        return {"output": f"Error generating final answer: {str(e)}"} 