import os
import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Optional

from backend.features.chunking_stratergy import markdown_chunking, semantic_chunking, sliding_window_chunking
from backend.features.mistral_parser import pdf_mistralocr_converter
from backend.core.s3_client import S3FileManager
from backend.agents.rag_agent import (
    create_pinecone_vector_store, 
    query_pinecone, 
    generate_openai_message, 
    generate_model_response,
    generate_openai_message_document
)

from openai import OpenAI

# Load environment variables
load_dotenv()

class NVDIARequest(BaseModel):
    year: str
    quarter: list
    parser: str = "mistral"  # Default to mistral parser
    chunk_strategy: str
    vector_store: str = "pinecone"  # Default to pinecone
    query: str

class DocumentQueryRequest(BaseModel):
    parser: str = "mistral"  # Default to mistral parser
    chunk_strategy: str
    vector_store: str = "pinecone"  # Default to pinecone
    file_name: str
    markdown_content: str
    query: str
    
app = FastAPI()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_FILE_INDEX = os.getenv("PINECONE_FILE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

client = OpenAI()

@app.get("/")
def read_root():
    return {"message": "NVDIA Financial Reports Analysis: FastAPI Backend with OpenAI Integration available for user queries..."}

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

@app.post("/query_nvdia_documents")
async def query_nvdia_documents(request: NVDIARequest):
    try:
        year = request.year
        quarter = request.quarter
        parser = request.parser
        chunk_strategy = request.chunk_strategy
        query = request.query
        vector_store = request.vector_store
        top_k = 10

        # Construct the S3 path for the NVIDIA document
        base_path = "nvidia-reports"  # This is the correct path in S3
        print(f"Using base path: {base_path}")
        s3_obj = S3FileManager(AWS_BUCKET_NAME, base_path)
        
        # Get the PDF content from S3 and process it with Mistral
        pdf_filename = f"nvidia_raw_pdf_{year}_{quarter[0]}.pdf"  # Keep the Q prefix
        print(f"Attempting to load PDF from: {base_path}/{pdf_filename}")
        pdf_content = s3_obj.load_s3_pdf(pdf_filename)
        if not pdf_content:
            raise HTTPException(status_code=404, detail=f"NVIDIA report for {year} {quarter[0]} not found at {base_path}/{pdf_filename}")
        print(f"Successfully loaded PDF, size: {len(pdf_content)} bytes")

        # Process the PDF with Mistral OCR
        print("Starting Mistral OCR processing...")
        try:
            md_file_name, markdown_content = pdf_mistralocr_converter(pdf_content, base_path, s3_obj)
            print(f"Successfully processed PDF with Mistral OCR, markdown size: {len(markdown_content)} bytes")
        except Exception as e:
            print(f"Error during Mistral OCR processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing PDF with Mistral OCR: {str(e)}")
        
        # Generate chunks using the specified strategy
        print(f"Generating chunks using {chunk_strategy} strategy...")
        chunks = generate_chunks(markdown_content, chunk_strategy)
        print(f"Generated {len(chunks)} chunks using {chunk_strategy} strategy")

        if vector_store == "pinecone":
            # Create vector store and query
            file_name = f"{year}_{quarter[0]}"  # Keep the Q prefix here too
            print(f"Creating Pinecone vector store for file: {file_name}")
            create_pinecone_vector_store(file_name, chunks, chunk_strategy)
            
            # Query using the same namespace format
            print(f"Querying Pinecone with namespace format: {parser}_{chunk_strategy}")
            result_chunks = query_pinecone(
                file=file_name,
                parser=parser,
                chunking_strategy=chunk_strategy,
                query=query,
                top_k=top_k
            )
            
            if len(result_chunks) == 0:
                raise HTTPException(status_code=500, detail="No relevant data found in the document")
            print(f"Found {len(result_chunks)} relevant chunks")
            
            # Generate response using OpenAI
            print("Generating OpenAI response...")
            message = generate_openai_message(result_chunks, year, quarter, query)
            answer = generate_model_response(message)
            print("Successfully generated response")
            
        else:
            raise HTTPException(status_code=400, detail="Only Pinecone vector store is currently supported for NVIDIA documents")

        return {"answer": answer}
    
    except Exception as e:
        print(f"Error in query_nvdia_documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

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

