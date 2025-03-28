import os
import sys
from pathlib import Path
import boto3
from datetime import datetime
import logging

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any

from backend.features.chunking_stratergy import markdown_chunking, semantic_chunking, sliding_window_chunking
from backend.features.mistral_parser import pdf_mistralocr_converter
from backend.core.s3_client import S3FileManager
from backend.agents.rag_agent import (
    AgenticResearchAssistant,
    create_pinecone_vector_store
)
from backend.agents.snowflake_agent import query_snowflake, generate_chart, get_nvidia_historical
from langgraph.graph import StateGraph, END
from agents.web_agent import NVIDIAWebSearchAgent
from backend.langgraph.research_graph import run_research_graph, initialize_research_graph
from backend.langgraph.state import ResearchRequest

from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_FILE_INDEX = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Validate required environment variables
required_vars = {
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "PINECONE_INDEX_NAME": PINECONE_FILE_INDEX,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "AWS_BUCKET_NAME": AWS_BUCKET_NAME,
    "MISTRAL_API_KEY": MISTRAL_API_KEY,
    "SERPAPI_API_KEY": SERPAPI_API_KEY,
    "TAVILY_API_KEY": TAVILY_API_KEY
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

client = OpenAI()

# Initialize global graph
_GLOBAL_GRAPH = None

class NVDIARequest(BaseModel):
    year: str
    quarter: list
    parser: str = "mistral"  # Default to mistral parser
    chunk_strategy: str
    vector_store: str = "pinecone"  # Default to pinecone
    query: str

class SnowflakeRequest(BaseModel):
    year: str
    quarter: str
    metric: str = "MARKETCAP"  # Default to MARKETCAP
    chart_type: str = "line"  # Chart type: line or bar

    class Config:
        json_schema_extra = {
            "example": {
                "year": "2023",
                "quarter": "Q1",
                "metric": "MARKETCAP",
                "chart_type": "line"
            }
        }

class DocumentQueryRequest(BaseModel):
    parser: str = "mistral"  # Default to mistral parser
    chunk_strategy: str
    vector_store: str = "pinecone"  # Default to pinecone
    file_name: str
    markdown_content: str
    query: str
    
class ResearchRequest(BaseModel):
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

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount the charts directory for static file serving
charts_dir = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(charts_dir, exist_ok=True)
app.mount("/charts", StaticFiles(directory=charts_dir), name="charts")

@app.get("/")
async def root():
    return {"message": "NVIDIA Research Assistant API"}

@app.post("/query_document")
async def query_document(request: DocumentQueryRequest):
    try:
        file_name = request.file_name
        markdown_content = request.markdown_content
        query = request.query
        parser = request.parser
        chunk_strategy = request.chunk_strategy
        vector_store = request.vector_store
        top_k = 10
        
        print(f"Processing query for file: {file_name}")
        print(f"Using parser: {parser}, chunk_strategy: {chunk_strategy}")
        
        # Generate chunks using the specified strategy
        chunks = generate_chunks(markdown_content, chunk_strategy)
        print(f"Generated {len(chunks)} chunks using {chunk_strategy} strategy")
        
        if vector_store == "pinecone":
            # Create vector store and query
            print("Creating Pinecone vector store...")
            records = create_pinecone_vector_store(file_name, chunks, chunk_strategy)
            print(f"Created vector store with {records} records")
            
            print("Querying Pinecone...")
            result_chunks = query_pinecone(
                file=file_name,
                parser=parser,
                chunking_strategy=chunk_strategy,
                query=query,
                top_k=top_k
            )
            
            print(f"Query returned {len(result_chunks)} chunks")
            
            if len(result_chunks) == 0:
                raise HTTPException(status_code=500, detail="No relevant data found in the document")
            
            # Generate response using OpenAI
            print("Generating OpenAI response...")
            message = generate_openai_message_document(query, result_chunks)
            answer = generate_model_response(message)
            
        else:
            raise HTTPException(status_code=400, detail="Only Pinecone vector store is currently supported")

        return {"answer": answer}
    
    except Exception as e:
        print(f"Error in query_document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@app.post("/research")
async def research(request: ResearchRequest):
    """Execute research workflow"""
    try:
        global _GLOBAL_GRAPH
        
        # Initialize research graph if not already initialized
        if _GLOBAL_GRAPH is None:
            logger.info("Initializing research graph...")
            _GLOBAL_GRAPH = initialize_research_graph()
        
        # Run research workflow
        result = await run_research_graph(
            query=request.query,
            year=request.year,
            quarter=request.quarter,
            agent_type=request.agent_type,
            metadata_filters=request.metadata_filters
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result["data"]
        
    except Exception as e:
        logger.error(f"Error in research endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_snowflake")
async def query_snowflake_data(request: SnowflakeRequest):
    """
    Endpoint for direct Snowflake queries (for backward compatibility).
    """
    try:
        summary, chart_path = get_nvidia_historical(
            year=request.year,
            quarter=request.quarter,
            metric=request.metric,
            chart_type=request.chart_type
        )
        
        return {
            "answer": f"{summary}\n\nChart saved as {chart_path}"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching Snowflake data: {str(e)}"
        )

@app.post("/query_nvdia_documents")
async def query_nvdia_documents(request: NVDIARequest):
    """
    Endpoint for direct Pinecone queries (for backward compatibility).
    """
    try:
        research_assistant = AgenticResearchAssistant()
        result = research_assistant.search_pinecone_db(
            query=request.query,
            year_quarter_dict={
                "year": request.year,
                "quarter": request.quarter[0]  # Take first quarter if list
            }
        )
        
        return {"answer": result}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}"
        )

def generate_chunks(markdown_content, chunk_strategy):
    """Generate chunks from markdown content using the specified strategy."""
    if chunk_strategy == "markdown":
        return markdown_chunking(markdown_content, heading_level=2)
    elif chunk_strategy == "semantic":
        return semantic_chunking(markdown_content, max_sentences=10)
    elif chunk_strategy == "sliding_window":
        return sliding_window_chunking(markdown_content, chunk_size=1000, overlap=150)
    else:
        raise HTTPException(
            status_code=400, 
            detail="Invalid chunk strategy. Supported strategies: markdown, semantic, sliding_window"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


