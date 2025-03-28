import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from tavily import TavilyClient
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
# Load environment variables from .env file
load_dotenv()
 
class WebSearchResult(BaseModel):
    """Structured representation of web search results."""
    title: str = Field(description="Title of the search result")
    link: str = Field(description="URL of the search result")
    snippet: str = Field(description="Brief description or summary of the result")
    relevance_score: float = Field(description="Subjective relevance score from 0-1", default=0.5)
    category: Optional[str] = Field(description="Category of the search result", default=None)
 
class NVIDIAWebSearchAgent:
    def __init__(
        self, 
        max_results: int = 5
    ):
        """
        Initialize the Web Search Agent for NVIDIA research.
        Args:
            max_results (int, optional): Maximum number of search results. Defaults to 5.
        """
        # Retrieve API keys from environment variables
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Log API key status (safely)
        logger.info(f"Tavily API key present: {bool(self.tavily_api_key)}")
        logger.info(f"OpenAI API key present: {bool(self.openai_api_key)}")
        
        # Provide more informative error message if API keys are missing
        missing_keys = []
        if not self.tavily_api_key:
            missing_keys.append("TAVILY_API_KEY")
        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")
        if missing_keys:
            error_msg = f"Missing environment variables: {', '.join(missing_keys)}. Please check your .env file or set these environment variables."
            logger.error(error_msg)
            raise ValueError(error_msg)
 
        self.max_results = max_results
        
        try:
            # Initialize Tavily client
            logger.info("Initializing Tavily client...")
            self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
            logger.info("Tavily client initialized successfully")
            
            # Initialize OpenAI client and language model
            logger.info("Initializing OpenAI client...")
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            self.llm = ChatOpenAI(
                temperature=0.2, 
                openai_api_key=self.openai_api_key
            )
            logger.info("OpenAI client initialized successfully")
            
            # Initialize output parser
            self.parser = PydanticOutputParser(pydantic_object=WebSearchResult)
            logger.info("Output parser initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing clients: {str(e)}")
            raise

    def search_web(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform web search using Tavily API.
        Args:
            query (str): The search query
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        try:
            logger.info(f"Performing Tavily search for query: {query}")
            
            # Enhance query with NVIDIA context
            enhanced_query = f"NVIDIA {query}"
            logger.info(f"Enhanced query: {enhanced_query}")
            
            # Perform search using Tavily API
            search_results = self.tavily_client.search(
                query=enhanced_query,
                search_depth="advanced",  # Use advanced search for better results
                max_results=self.max_results
            )
            
            # Extract and format results
            formatted_results = []
            for result in search_results.get('results', []):
                formatted_result = {
                    'title': result.get('title', ''),
                    'link': result.get('url', ''),
                    'snippet': result.get('content', ''),
                    'relevance_score': result.get('score', 0.5),
                    'category': None  # Tavily doesn't provide categories
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error performing web search: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def generate_comprehensive_research_report(self, query: str) -> Dict[str, Any]:
        """
        Generate a comprehensive research report based on web search results.
        Args:
            query (str): The research query
        Returns:
            Dict[str, Any]: Research report containing search results and analysis
        """
        try:
            print(f"\n🔍 Performing web search for query: {query}")
            
            # Perform web search
            search_results = self.search_web(query)
            
            if not search_results:
                print("❌ No search results found")
                return {
                    "status": "error",
                    "message": "No search results found. Please try a different query.",
                    "results": []
                }
            
            print(f"✅ Found {len(search_results)} search results")
            
            # Format search results
            formatted_results = []
            for result in search_results:
                try:
                    formatted_result = WebSearchResult(
                        title=result.get("title", ""),
                        link=result.get("link", ""),
                        snippet=result.get("snippet", ""),
                        relevance_score=result.get("relevance_score", 0.5),
                        category=None
                    )
                    formatted_results.append(formatted_result.dict())
                    print(f"✅ Formatted result: {formatted_result.title}")
                except ValidationError as e:
                    print(f"❌ Error formatting result: {str(e)}")
                    continue
            
            if not formatted_results:
                print("❌ No valid results after formatting")
                return {
                    "status": "error",
                    "message": "No valid results found after processing.",
                    "results": []
                }
            
            print("🤖 Generating analysis using OpenAI...")
            
            # Generate analysis using OpenAI with improved formatting
            analysis_prompt = f"""
            Based on the following search results about NVIDIA, provide a comprehensive analysis in markdown format:
            
            Query: {query}
            
            Search Results:
            {chr(10).join([f"- {r['title']}: {r['snippet']}" for r in formatted_results])}
            
            Please provide a well-structured analysis with the following sections:
            
            # Executive Summary
            [Provide a brief overview of the key findings]
            
            # Key Metrics and Data
            [Present important numbers and statistics in a clear format]
            
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
            """
            
            analysis = self.llm.invoke(analysis_prompt)
            print("✅ Analysis generated successfully")
            
            # Generate visualizations using Snowflake data
            try:
                from backend.agents.snowflake_agent import get_nvidia_historical
                import os
                
                # Get current year and quarter
                from datetime import datetime
                current_year = str(datetime.now().year)
                current_quarter = f"Q{(datetime.now().month - 1) // 3 + 1}"
                
                # Get historical data and generate chart
                summary, chart_path = get_nvidia_historical(
                    year=current_year,
                    quarter=current_quarter,
                    metric="VOLUME",
                    chart_type="line"
                )
                
                # Add chart to the response
                return {
                    "status": "success",
                    "query": query,
                    "results": formatted_results,
                    "analysis": analysis.content,
                    "visualizations": [{
                        "title": "NVIDIA Trading Volume Trend",
                        "url": chart_path,
                        "type": "chart"
                    }]
                }
                
            except Exception as e:
                print(f"❌ Error generating visualizations: {str(e)}")
                # Return response without visualizations if there's an error
                return {
                    "status": "success",
                    "query": query,
                    "results": formatted_results,
                    "analysis": analysis.content
                }
            
        except Exception as e:
            print(f"❌ Error generating research report: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Error generating research report: {str(e)}",
                "results": []
            }
 
# Example usage with improved error handling
def example_usage():
    try:
        # Initialize agent
        web_agent = NVIDIAWebSearchAgent()
        # Generate comprehensive research report
        nvidia_research = web_agent.generate_comprehensive_research_report("AI chip innovations 2024")
        # Pretty print the research report
        import json
        print(json.dumps(nvidia_research, indent=2))
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please create a .env file in your project root with the following:")
        print("TAVILY_API_KEY=your_tavily_key")
        print("OPENAI_API_KEY=your_openai_key")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
 
if __name__ == "__main__":
    example_usage()
 
"""
Setup Instructions:
 
1. Install required libraries:
   pip install python-dotenv tavily-python openai langchain-openai pydantic
 
2. Create a .env file in your project root:
   TAVILY_API_KEY=your_tavily_key
   OPENAI_API_KEY=your_openai_key
 
3. Recommended .env file location:
   Place .env in the root of your project directory
   Add .env to your .gitignore file to prevent accidentally sharing sensitive keys
 
Troubleshooting:
- Ensure python-dotenv is installed
- Verify .env file is in the correct directory
- Check that API keys are correctly set
- Confirm internet connection
"""