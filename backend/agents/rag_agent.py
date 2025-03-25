 
import os
import logging
import sys
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from core.s3_client import S3FileManager
from dotenv import load_dotenv
from features.chunking_stratergy import markdown_chunking, semantic_chunking, sliding_window_chunking
from typing import List, Optional
import requests
import time
from fastapi import HTTPException

# Load the .env file from project root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(dotenv_path)

# Debug print to verify variables are loaded
print(f"PINECONE_API_KEY exists: {bool(os.getenv('PINECONE_API_KEY'))}")
print(f"PINECONE_ENVIRONMENT exists: {bool(os.getenv('PINECONE_ENVIRONMENT'))}")
print(f"PINECONE_INDEX_NAME exists: {bool(os.getenv('PINECONE_INDEX_NAME'))}")
print(f"OPENAI_API_KEY exists: {bool(os.getenv('OPENAI_API_KEY'))}")

# List of required environment variables
required_vars = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "OPENAI_API_KEY"]
# Adjust this list based on what your code actually requires

# Check for missing variables
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)  # Exit with error

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "nvidia-rag-pipeline")
AWS_S3_BUCKET = os.getenv("AWS_BUCKET_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_ENVIRONMENT=os.getenv("PINECONE_ENVIRONMENT")

# Validate environment variables
# if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME, AWS_S3_BUCKET, OPENAI_API_KEY]):
#     logger.error("Missing required environment variables.")
#     raise ValueError("Missing required environment variables.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def connect_to_pinecone_index():
    """Connect to the Pinecone index and return it."""
    logger.info(f"Connecting to Pinecone with index name: {PINECONE_INDEX_NAME}")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Create index only if it doesn't exist
        if not pc.has_index(PINECONE_INDEX_NAME):
            logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1536,  # OpenAI text-embedding-3-small dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
                tags={
                    "environment": "development",
                    "model": "text-embedding-3-small"
                }
            )
            logger.info("Waiting for index to be ready...")
            time.sleep(10)  # Wait for index to be ready
            
        index = pc.Index(PINECONE_INDEX_NAME)
        return index
    except Exception as e:
        logger.error(f"Error connecting to Pinecone: {str(e)}")
        raise

def read_markdown_file(file, s3_obj):
    """Read the content of a markdown file from S3."""
    try:
        content = s3_obj.load_s3_file_content(file)
        return content
    except Exception as e:
        logger.error(f"Error reading markdown file {file}: {str(e)}")
        raise

def get_embedding(text: str) -> List[float]:
    """Generate an embedding for the given text using OpenAI text-embedding-3-small."""
    try:
        logger.info(f"Generating embedding for text (length: {len(text)})")
        
        response = client.embeddings.create(
            model="text-embedding-3-small",  # Using OpenAI's latest embedding model
            input=text,
            encoding_format="float"  # Explicitly request float format
        )
        embedding = response.data[0].embedding
        logger.info(f"Successfully generated embedding of size {len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        logger.error(f"OpenAI API error details: {str(e)}")
        raise

def create_pinecone_vector_store(file, chunks, chunk_strategy):
    """Create a vector store in Pinecone for the given chunks."""
    index = connect_to_pinecone_index()
    if not index:
        logger.error("Failed to connect to Pinecone index")
        return 0

    vectors = []
    file_parts = file.split('/')
    
    # Handle both test documents and actual NVIDIA reports
    if len(file_parts) > 2:
        # For actual NVIDIA reports
        parser = file_parts[1]
        identifier = file_parts[2]
        year = identifier[15:19] if len(identifier) > 19 else "unknown"
        quarter = identifier[20:22] if len(identifier) > 22 else "unknown"
    else:
        # For test documents
        parser = "mistral"  # Use mistral as default parser for test documents
        year = "test"
        quarter = "test"
    
    records = 0
    # Ensure consistent namespace naming
    namespace = f"{parser}_{chunk_strategy.lower()}"  # Convert to lowercase for consistency

    logger.info(f"Processing {len(chunks)} chunks for {file}")
    logger.info(f"Using namespace: {namespace}")
    logger.info(f"Year: {year}, Quarter: {quarter}")
    
    for i, chunk in enumerate(chunks):
        if not chunk.strip():  # Skip empty chunks
            continue
            
        try:
            embedding = get_embedding(chunk)
            if not any(embedding):  # Skip if embedding is all zeros
                logger.warning(f"Skipping chunk {i}: Generated zero embedding")
                continue
                
            vectors.append((
                f"{file}_chunk_{i}",  # Unique ID
                embedding,  # Embedding vector
                {
                    "year": year,
                    "quarter": quarter,
                    "text": chunk,
                    "parser": parser,
                    "strategy": chunk_strategy.lower()  # Store lowercase strategy name
                }  # Metadata
            ))
            
            # Batch upload every 20 vectors
            if len(vectors) >= 20:
                try:
                    index.upsert(vectors=vectors, namespace=namespace)
                    records += len(vectors)
                    logger.info(f"Inserted batch of {len(vectors)} vectors. Total: {records}")
                    vectors.clear()
                except Exception as e:
                    logger.error(f"Error uploading batch: {str(e)}")
                    vectors.clear()  # Clear failed batch and continue
                    
        except Exception as e:
            logger.error(f"Error processing chunk {i}: {str(e)}")
            continue

    # Upload any remaining vectors
    if vectors:
        try:
            index.upsert(vectors=vectors, namespace=namespace)
            records += len(vectors)
            logger.info(f"Inserted final batch of {len(vectors)} vectors. Total: {records}")
        except Exception as e:
            logger.error(f"Error uploading final batch: {str(e)}")

    logger.info(f"Successfully processed {records} chunks for {file}")
    return records

def query_pinecone(
    file: str,
    parser: str,
    chunking_strategy: str,
    query: str,
    top_k: int = 5  # Default to 5 chunks
) -> List[str]:
    """Query Pinecone index with filters"""
    try:
        # Get query embedding
        logger.info(f"Generating embedding for query: {query}")
        query_embedding = get_embedding(query)
        logger.info(f"Generated query embedding of size {len(query_embedding)}")
        
        # Connect to index
        index = connect_to_pinecone_index()
        if not index:
            logger.error("Failed to connect to Pinecone index")
            return []
            
        # Use the same namespace format as in create_pinecone_vector_store
        namespace = f"{parser}_{chunking_strategy.lower()}"  # Convert to lowercase for consistency
        logger.info(f"Querying namespace: {namespace}")
        
        # Query index with lower similarity threshold
        try:
            logger.info("Executing Pinecone query...")
            results = index.query(
                vector=query_embedding,
                namespace=namespace,
                top_k=top_k,  # Limit to top_k most relevant chunks
                include_metadata=True,
                min_score=0.0  # Lower the similarity threshold
            )
            
            # Log results for debugging
            logger.info(f"Query returned {len(results.matches)} matches")
            for match in results.matches:
                logger.info(f"Match score: {match.score}")
                if hasattr(match, 'metadata') and match.metadata:
                    logger.info(f"Match metadata: {match.metadata}")
                else:
                    logger.warning("Match has no metadata")
            
            # Extract and return text from matches
            texts = []
            for match in results.matches:
                if hasattr(match, 'metadata') and match.metadata and 'text' in match.metadata:
                    texts.append(match.metadata['text'])
                else:
                    logger.warning(f"Match missing text in metadata: {match}")
            
            logger.info(f"Returning {len(texts)} text chunks")
            return texts
            
        except Exception as e:
            logger.error(f"Error during Pinecone query: {str(e)}")
            logger.error(f"Query parameters: namespace={namespace}, top_k={top_k}")
            return []
            
    except Exception as e:
        logger.error(f"Error in query_pinecone: {str(e)}")
        logger.error(f"Full error details: {str(e)}")
        return []

def generate_openai_message(chunks, year, quarter, query):
    # Limit the number of chunks to avoid token limit errors
    max_chunks = 5  # Limit to top 5 most relevant chunks
    chunks = chunks[:max_chunks]
    
    prompt = f"""
    Below are relevant excerpts from a NVDIA quarterly financial report for year {year} and quarter {quarter} that may help answer the query.

    --- User Query ---
    {query}

    --- Relevant Document Chunks ---
    {chr(10).join([f'Chunk {i+1}: {chunk}' for i, chunk in enumerate(chunks)])}

    Based on the provided document chunks, generate a comprehensive response to the query. If needed, synthesize the information and ensure clarity.
    """
    return prompt

def generate_openai_message_document(query, chunks):
    # Limit the number of chunks to avoid token limit errors
    max_chunks = 5  # Limit to top 5 most relevant chunks
    chunks = chunks[:max_chunks]
    
    prompt = f"""
    Below are relevant excerpts from a document uploaded by the user that may help answer the user query.

    --- User Query ---
    {query}

    --- Relevant Document Chunks ---
    {chr(10).join([f'Chunk {i+1}: {chunk}' for i, chunk in enumerate(chunks)])}

    Based on the provided document chunks, generate a comprehensive response to the query. If needed, synthesize the information and ensure clarity.
    """
    return prompt

def generate_model_response(message):
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. You are given excerpts from NVDIA's quarterly financial report. Use them to answer the user query."},
                {"role": "user", "content": message}
            ],
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response from OpenAI Model: {str(e)}")

def main():
    """Main function to process markdown files and store them in Pinecone."""
    base_path = "nvidia/"
    s3_obj = S3FileManager(AWS_S3_BUCKET, base_path)
    files = list({file for file in s3_obj.list_files() if file.endswith('.md')})
    logger.info(f"Files to process: {files}")

    for i, file in enumerate(files):
        logger.info(f"Processing File {i+1}: {file}")
        try:
            content = read_markdown_file(file, s3_obj)
            
            logger.info("Using markdown chunking strategy...")
            chunks = markdown_chunking(content, heading_level=2)
            logger.info(f"Chunk size: {len(chunks)}")
            create_pinecone_vector_store(file, chunks, "markdown")
            
            logger.info("Using semantic chunking strategy...")
            chunks = semantic_chunking(content, max_sentences=10)
            logger.info(f"Chunk size: {len(chunks)}")
            create_pinecone_vector_store(file, chunks, "semantic")
            
            logger.info("Using sliding window chunking strategy...")
            chunks = sliding_window_chunking(content, chunk_size=1000, overlap=150)
            logger.info(f"Chunk size: {len(chunks)}")
            create_pinecone_vector_store(file, chunks, "slidingwindow")
        except Exception as e:
            logger.error(f"Error processing file {file}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
