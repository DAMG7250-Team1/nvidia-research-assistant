import os
import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from state import AgentAction
import os
from dotenv import load_dotenv
from agents.rag_agent import AgenticResearchAssistant
from agents.web_agent import WebSearchAgent
from agents.snowflake_agent import generate_snowflake_insights
import re
load_dotenv()
# Initialize the LLM
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.2, openai_api_key=api_key)

def run_oracle(state):
    """
    Decides which tool to use based on the query and selected mode.
    """
    print("\n" + "="*80)
    print("🔮 ORACLE NODE: Deciding which tool to use")
    print("="*80)
    print(f"Query: \"{state['input']}\"")
    print(f"Mode: {state.get('mode')}")
    
    # Get list of tools we've already used
    used_tools = [step.tool for step in state.get("intermediate_steps", [])]
    print(f"Tools used so far: {used_tools}")
    
    # For individual modes, use the selected tool once and then move to final answer
    if state.get("mode") in ["pinecone", "web_search", "snowflake"]:
        if not used_tools:
            chosen_tool = state.get("mode")
        else:
            chosen_tool = "final_answer"
    
    # For combined mode
    elif state.get("mode") == "combined":
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
                oracle_prompt.format(input=state["input"])
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
                    input=state["input"],
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
            "query": state["input"],
            "metadata_filters": state.get("metadata_filters", {})
        },
        log=f"Selected {chosen_tool} based on mode: {state.get('mode')}"
    )
    
    return {"intermediate_steps": state["intermediate_steps"] + [action]}

def router(state):
    """
    Routes to the next node based on the chosen tool.
    """
    print("\n" + "-"*80)
    print("🧭 ROUTER: Determining next node")
    print("-"*80)
    
    # Get the most recent action's tool
    if isinstance(state["intermediate_steps"], list) and state["intermediate_steps"]:
        latest_tool = state["intermediate_steps"][-1].tool
        print(f"Latest tool selected: {latest_tool}")
        
        # Map tools to nodes
        tool_to_node = {
            "pinecone": "rag_search",
            "web_search": "web_search",
            "snowflake": "snowflake_search",
            "final_answer": "final_answer"
        }
        
        # Get the next node
        next_node = tool_to_node.get(latest_tool)
        print(f"🚀 ROUTING TO: {next_node}")
        print("-"*80 + "\n")
        return next_node
    else:
        # Default to final_answer if no steps or invalid format
        print("No valid steps found - defaulting to final_answer")
        print("-"*80 + "\n")
        return "final_answer"

def rag_search(state):
    """
    Execute the Pinecone RAG search.
    """
    print("\n" + "="*80)
    print("📚 PINECONE RAG SEARCH NODE: Searching financial reports")
    print("="*80)
    
    # Get the tool input from the last intermediate step
    tool_input = state["intermediate_steps"][-1].tool_input
    
    # Extract the query and metadata filters from the tool input
    query = tool_input.get("query", "")
    metadata_filters = tool_input.get("metadata_filters", {})
    
    print(f"Query: \"{query}\"")
    print(f"Filters: {metadata_filters}")
    
    # Call the search function directly (no tool wrapper)
    research_assistant = AgenticResearchAssistant()
    try:
        result = research_assistant.search_pinecone_db(
            query=query,
            year_quarter_dict=metadata_filters,
            top_k=20
        )
        print(f"✅ Pinecone search completed successfully")
        print(f"Result preview (first 200 chars): {str(result)[:200]}...")
    except Exception as e:
        print(f"❌ Error in rag_search: {e}")
        result = f"Error searching Pinecone: {str(e)}"
    
    print("="*80 + "\n")
    
    # Create a new action with the result
    new_action = AgentAction(
        tool="rag_search_result",
        tool_input=tool_input,
        log=str(result)
    )
    
    # Update the intermediate steps
    return {"intermediate_steps": state["intermediate_steps"] + [new_action]}

def web_search(state):
    """
    Execute the web search.
    """
    print("\n" + "="*80)
    print("🌐 WEB SEARCH NODE: Searching for recent information")
    print("="*80)
    
    tool_input = state["intermediate_steps"][-1].tool_input
    query = tool_input.get("query", "")
    print(f"Query: \"{query}\"")
    
    web_search_agent = WebSearchAgent()
    try:
        # Use run instead of search_news to get both news and trends with links
        search_results = web_search_agent.run(query)
        
        if search_results["status"] == "error":
            result = f"Error in web search: {search_results['error']}"
            print("❌ Web search error:", result)
        else:
            # Format results into readable text with clickable links
            formatted_result = "### Recent News and Analysis\n\n"
            
            # Add news section with clickable links
            if search_results["raw_results"]["news"]:
                formatted_result += "#### 📰 Latest News\n\n"
                for i, item in enumerate(search_results["raw_results"]["news"], 1):
                    formatted_result += f"**{i}. [{item['title']}]({item['link']})**\n"
                    formatted_result += f"📅 {item['date']} | 🔗 [{item['source']}]({item['link']})\n"
                    formatted_result += f"Summary: {item['snippet']}\n\n"
            
            # Add trends section with clickable links
            if search_results["raw_results"]["trends"]:
                formatted_result += "#### 📈 Market Trends & Analysis\n\n"
                for i, item in enumerate(search_results["raw_results"]["trends"], 1):
                    formatted_result += f"**{i}. [{item['title']}]({item['link']})**\n"
                    formatted_result += f"Source: [{item['source']}]({item['link']})\n"
                    formatted_result += f"Key Points: {item['snippet']}\n\n"
            
            # Add insights section
            if search_results.get("insights"):
                formatted_result += "#### 🔍 Analysis\n\n"
                formatted_result += f"{search_results['insights']}\n\n"
            
            result = formatted_result
            print(f"✅ Web search completed successfully")
            print(f"Result preview (first 200 chars): {result[:200]}...")
            
    except Exception as e:
        print(f"❌ Error in web_search: {e}")
        result = f"Error searching web: {str(e)}"
    
    print("="*80 + "\n")
    
    return {"intermediate_steps": state["intermediate_steps"] + [
        AgentAction(
            tool="web_search_result",
            tool_input=tool_input,
            log=str(result)
        )
    ]}

def generate_final_answer(state):
    """
    Generate a comprehensive final answer based on collected information.
    """
    print("\n" + "="*80)
    print("🏁 FINAL ANSWER NODE: Generating comprehensive response")
    print("="*80)
    
    # Extract results from intermediate steps
    rag_result = ""
    web_result = ""
    snowflake_result = ""
    visualization_urls = []  # Store visualization URLs
    web_links = []  # Store web search links
    
    for step in state["intermediate_steps"]:
        if step.tool == "rag_search_result":
            rag_result = step.log
            print("✓ Found Pinecone RAG search results")
        elif step.tool == "web_search_result":
            web_result = step.log
            print("✓ Found Web search results")
            
            # Extract web links before sending to LLM
            news_links = re.findall(r'\[(.*?)\]\((https?://[^\s\)]+)\)', web_result)
            web_links.extend([{
                "title": title,
                "url": url
            } for title, url in news_links])
            print(f"Extracted {len(news_links)} web links")
            
        elif step.tool == "snowflake_search_result":
            snowflake_result = step.log
            print("✓ Found Snowflake search results")
            
            # Extract visualization URLs from Snowflake markdown
            if "## Visualizations" in snowflake_result:
                viz_section = snowflake_result.split("## Visualizations")[1]
                viz_blocks = re.findall(r'!\[(.*?)\]\((.*?)\)\n\n\*(.*?)\*', viz_section)
                visualization_urls.extend([{
                    "title": title,
                    "url": url,
                    "caption": caption
                } for title, url, caption in viz_blocks])
                snowflake_result = snowflake_result.split("## Visualizations")[0]
    
    print(f"Found {len(visualization_urls)} visualizations and {len(web_links)} web links")
    
    # Generate prompt based on available results
    mode = state.get("mode", "combined")
    query = state["input"]
    
    if mode == "snowflake":
        prompt = f"""You are an AI assistant specializing in NVIDIA financial metrics analysis.
        
        Based on the following financial data, provide a comprehensive answer to:
        
        QUERY: {query}
        
        FINANCIAL METRICS ANALYSIS:
        {snowflake_result}
        
        Format your response with clear sections and bullet points where appropriate.
        Focus on the financial metrics and trends.
        DO NOT try to describe any visualizations - they will be added separately.
        """
    
    elif mode == "web_search":
        prompt = f"""You are an AI assistant specializing in NVIDIA market analysis.
        
        Based on the following recent news and market information about NVIDIA, provide a comprehensive answer to:
        
        QUERY: {query}
        
        RECENT NEWS AND MARKET INFORMATION:
        {web_result if web_result else "No recent news data available."}
        
        Please structure your response with the following sections:
        1. Key Findings - Main insights from the news
        2. Market Impact - How this affects NVIDIA's market position
        3. Future Implications - What this means for NVIDIA going forward
        4. Sources - Referenced news articles and analysis
        
        Format your response with clear sections and bullet points where appropriate.
        Preserve any links from the original sources in your response.
        """
    
    elif mode == "combined":
        prompt = f"""You are an AI assistant specializing in comprehensive NVIDIA analysis.
        
        Based on the following information sources, provide a detailed research report answering:
        
        QUERY: {query}
        
        FINANCIAL METRICS ANALYSIS:
        {snowflake_result if snowflake_result else "No financial metrics data available."}
        
        QUARTERLY REPORT DATA:
        {rag_result if rag_result else "No historical financial data available."}
        
        RECENT NEWS:
        {web_result if web_result else "No recent news data available."}
        
        Please structure your report with the following sections:
        1. Executive Summary - Brief overview of findings
        2. Financial Metrics Analysis - Key metrics and trends
        3. Historical Financial Analysis - Insights from quarterly reports
        4. Recent Developments - News and market updates
        5. Conclusion - Key takeaways and implications
        6. Sources - Data sources used
        
        IMPORTANT: DO NOT try to describe any visualizations - they will be added separately.
        Focus on analyzing the data and insights.
        """
    
    elif mode == "pinecone":
        prompt = f"""You are an AI assistant specializing in NVIDIA financial report analysis.
        
        Based on the following information from NVIDIA's quarterly reports, provide a comprehensive answer to:
        
        QUERY: {query}
        
        QUARTERLY REPORT DATA:
        {rag_result if rag_result else "No historical financial data available."}
        
        Please structure your response with:
        1. Key Financial Insights
        2. Historical Trends
        3. Important Developments
        4. Sources
        
        Format your response with clear sections and bullet points where appropriate.
        """
    
    else:
        # Fallback prompt for any other mode
        prompt = f"""You are an AI assistant specializing in comprehensive NVIDIA analysis.
        
        Based on the following information sources, provide a detailed research report answering:
        
        QUERY: {query}
        
        FINANCIAL METRICS ANALYSIS:
        {snowflake_result if snowflake_result else "No financial metrics data available."}
        
        QUARTERLY REPORT DATA:
        {rag_result if rag_result else "No historical financial data available."}
        
        RECENT NEWS:
        {web_result if web_result else "No recent news data available."}
        
        Please structure your report with the following sections:
        1. Executive Summary - Brief overview of findings
        2. Financial Metrics Analysis - Key metrics and trends
        3. Historical Financial Analysis - Insights from quarterly reports
        4. Recent Developments - News and market updates
           - IMPORTANT: Include the original news article links in this section
           - Format as: [Article Title](URL)
        5. Conclusion - Key takeaways and implications
        6. Sources - List all referenced articles with their links
        
        IMPORTANT FORMATTING INSTRUCTIONS:
        - DO NOT try to describe any visualizations - they will be added separately
        - PRESERVE all markdown links from the news section in their original format: [Title](URL)
        - When citing news articles, always maintain the clickable links
        - Include the complete source URLs in the Sources section
        
        Focus on analyzing the data and insights while maintaining all source links.
        """
    
    # Generate the final answer
    print("Generating final response with LLM...")
    response = llm.invoke(prompt)
    
    # Construct the final response with preserved visualizations and web links
    final_response = response.content
    
    # Add web links section if we have any
    if web_links:
        final_response += "\n\n## Sources and References\n\n"
        for link in web_links:
            final_response += f"- [{link['title']}]({link['url']})\n"
    
    # Add visualizations section if we have any
    if visualization_urls:
        final_response += "\n\n## Visualizations\n\n"
        for viz in visualization_urls:
            final_response += f"![{viz['title']}]({viz['url']})\n\n"
            final_response += f"*{viz['caption']}*\n\n"
    
    # Create a new action with the result
    new_action = AgentAction(
        tool="final_answer_result",
        tool_input={"query": query},
        log=final_response
    )
    
    print("✅ Final answer generated successfully")
    print(f"Number of visualizations preserved: {len(visualization_urls)}")
    print(f"Number of web links preserved: {len(web_links)}")
    print(f"Result preview (first 200 chars): {final_response[:200]}...")
    print("="*80 + "\n")
    
    return {
        "output": final_response,
        "intermediate_steps": state["intermediate_steps"] + [new_action]
    }

def snowflake_search(state):
    """
    Execute the Snowflake search for financial data.
    """
    # Get the tool input from the last intermediate step
    tool_input = state["intermediate_steps"][-1].tool_input
    
    print(f"Running snowflake_search with input: {tool_input}")
    
    # Extract the query and metadata filters from the tool input
    query = tool_input.get("query", "")
    metadata_filters = tool_input.get("metadata_filters", {})
    
    print(f"Searching Snowflake with query={query}, filters={metadata_filters}")
    
    try:
        # Call the Snowflake insights function
        result = generate_snowflake_insights(query, metadata_filters)
        print(f"Result: {result}")
        # Format the response for LangGraph with proper markdown
        formatted_result = {
            "text": f"## Financial Data Analysis\n\n{result['summary']}\n\n",
            "visualizations": result['visualizations']
        }
        
        print("\n" + "="*80)
        print("🔄 LANGGRAPH VISUALIZATION FORMATTING")
        print("="*80)
        print("Converting to markdown format...")
        
        # Convert to markdown string with proper image syntax
        markdown_result = formatted_result["text"]
        if result['visualizations']:
            markdown_result += "\n## Visualizations\n\n"
            for viz in result['visualizations']:
                image_markdown = f"![{viz['title']}]({viz['url']})\n\n"
                caption_markdown = f"*{viz['title']} - {', '.join(viz['columns'])}*\n\n"
                markdown_result += image_markdown + caption_markdown
                
                print("\nAdded visualization to markdown:")
                print("Image markdown:", image_markdown.strip())
                print("Caption markdown:", caption_markdown.strip())
        
        print("\nFinal markdown result preview:")
        print("-"*40)
        print(markdown_result[-5000:], "...")
        print("="*80 + "\n")
        
        new_action = AgentAction(
            tool="snowflake_search_result",
            tool_input=tool_input,
            log=markdown_result
        )
        
    except Exception as e:
        print(f"Error in snowflake_search: {e}")
        import traceback
        traceback.print_exc()
        new_action = AgentAction(
            tool="snowflake_search_result",
            tool_input=tool_input,
            log=f"Error searching Snowflake: {str(e)}"
        )
    
    return {"intermediate_steps": state["intermediate_steps"] + [new_action]}