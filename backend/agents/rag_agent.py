import os
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from core.s3_client import S3FileManager
from features.chunking_stratergy import markdown_chunking, semantic_chunking, sliding_window_chunking
from typing import List, Optional, Dict
import requests
import time
from fastapi import HTTPException
import re

# Add backend directory to Python path
backend_dir = str(Path(__file__).parent.parent)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# Create logs directory if it doesn't exist
logs_dir = os.path.join(backend_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(logs_dir, "rag_agent.log"))
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

class AgenticResearchAssistant:
    def __init__(self):
        """Initialize the RAG agent with necessary clients and configurations."""
        try:
            # Initialize OpenAI client
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            self.pinecone_index = self.pc.Index(PINECONE_INDEX_NAME)
            
            # Initialize S3 client
            self.s3_manager = S3FileManager(AWS_BUCKET_NAME)
            
            logger.info("RAG Agent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG Agent: {str(e)}")
            raise

    def check_pinecone_data(self, year: str, quarter: str) -> bool:
        """
        Check if data exists in Pinecone for the specified year and quarter.
        
        Args:
            year (str): The year to check
            quarter (str): The quarter to check
            
        Returns:
            bool: True if data exists, False otherwise
        """
        try:
            logger.info(f"Checking Pinecone data for {year} {quarter}")
            
            # Create a dummy vector for querying
            dummy_vector = [0.0] * 1536  # OpenAI embedding dimension
            
            # Check in all chunking strategies
            strategies = ["markdown", "semantic", "slidingwindow"]
            data_found = False
            
            for strategy in strategies:
                namespace = f"mistral_{strategy}"  # Consistent namespace format
                try:
                    logger.info(f"Checking namespace: {namespace}")
                    results = self.pinecone_index.query(
                        vector=dummy_vector,
                        top_k=1,
                        namespace=namespace,
                        filter={
                            "year": year,
                            "quarter": quarter
                        }
                    )
                    
                    if results.matches:
                        logger.info(f"Found data in namespace {namespace} for {year} {quarter}")
                        logger.info(f"Match metadata: {results.matches[0].metadata}")
                        data_found = True
                        break  # Found data, no need to check other namespaces
                    else:
                        logger.info(f"No matches found in namespace {namespace}")
                        
                except Exception as e:
                    logger.warning(f"Error checking namespace {namespace}: {str(e)}")
                    continue
            
            if not data_found:
                logger.warning(f"No data found in any namespace for {year} {quarter}")
            return data_found
            
        except Exception as e:
            logger.error(f"Error in check_pinecone_data: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            return False

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def create_pinecone_vector_store(self, file_name: str, chunks: List[str], strategy: str = "markdown") -> None:
        """Create a vector store in Pinecone from the given chunks"""
        try:
            # Extract year and quarter from filename
            match = re.search(r'nvidia_earnings_call_(\d{4})_Q(\d)', file_name)
            if not match:
                raise ValueError(f"Invalid filename format: {file_name}")
            
            year, quarter = match.groups()
            
            # Get embeddings for chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = []
            for chunk in chunks:
                embedding = self.get_embedding(chunk)
                embeddings.append(embedding)
            
            # Prepare metadata
            metadata_list = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    'text': chunk,
                    'file': file_name,
                    'year': year,
                    'quarter': quarter,
                    'chunk_index': i,
                    'strategy': 'markdown',
                    'folder_path': self.s3_manager.get_report_folder_path(year, quarter)
                }
                metadata_list.append(metadata)
            
            # Prepare vectors for upsert
            vectors = []
            for i, (embedding, metadata) in enumerate(zip(embeddings, metadata_list)):
                vector = {
                    'id': f"{file_name}_chunk_{i}",
                    'values': embedding,
                    'metadata': metadata
                }
                vectors.append(vector)
            
            # Upsert to Pinecone with correct namespace
            namespace = f"mistral_{strategy.lower()}"  # Use consistent namespace format
            logger.info(f"Upserting {len(vectors)} vectors to Pinecone in namespace: {namespace}")
            self.pinecone_index.upsert(vectors=vectors, namespace=namespace)
            
            logger.info("Successfully created vector store in Pinecone")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            raise

    def query_pinecone(self, file: str, parser: str, chunking_strategy: str, 
                      query: str, top_k: int = 5) -> List[str]:
        """Query Pinecone for relevant chunks based on query."""
        try:
            logger.info(f"Starting query_pinecone with file: {file}")
            logger.info(f"Parser: {parser}, chunking_strategy: {chunking_strategy}")
            logger.info(f"Original query: {query}")
            
            # Enhance query for breakthrough-related searches
            enhanced_query = query
            if "breakthrough" in query.lower() or "innovation" in query.lower():
                enhanced_query = f"{query} technology advancement development achievement milestone progress"
                logger.info(f"Enhanced query for breakthrough search: {enhanced_query}")
            
            # Get query embedding
            logger.info("Generating query embedding...")
            query_embedding = self.get_embedding(enhanced_query)
            logger.info(f"Generated query embedding of size {len(query_embedding)}")
            
            # Connect to index
            logger.info("Connecting to Pinecone index...")
            index = self.pc.Index(PINECONE_INDEX_NAME)
            if not index:
                logger.error("Failed to connect to Pinecone index")
                return []
            
            # Use consistent namespace format
            namespace = f"mistral_{chunking_strategy.lower()}"  # Always use mistral prefix
            logger.info(f"Querying namespace: {namespace}")
            
            # Extract year and quarter from file name
            try:
                year = file.split('_')[-2]
                quarter = file.split('_')[-1].replace('.md', '')
                logger.info(f"Extracted year: {year}, quarter: {quarter}")
            except Exception as e:
                logger.error(f"Error extracting year/quarter from filename: {str(e)}")
                year = None
                quarter = None
            
            # Query index with proper filters
            try:
                logger.info("Executing Pinecone query...")
                filter_dict = {}
                if year and quarter:
                    filter_dict = {
                        "year": year,
                        "quarter": quarter
                    }
                    logger.info(f"Using filter: {filter_dict}")
                
                # Try querying with filters first
                results = index.query(
                    vector=query_embedding,
                    namespace=namespace,
                    top_k=top_k,
                    include_metadata=True,
                    min_score=0.05,  # Lower threshold for more results
                    filter=filter_dict
                )
                
                # If no results with filters, try without filters
                if not results.matches:
                    logger.info("No results with filters, trying without filters...")
                    results = index.query(
                        vector=query_embedding,
                        namespace=namespace,
                        top_k=top_k,
                        include_metadata=True,
                        min_score=0.05  # Keep consistent threshold
                    )
                
                # Log results for debugging
                logger.info(f"Query returned {len(results.matches)} matches")
                for match in results.matches:
                    logger.info(f"Match score: {match.score}")
                    if hasattr(match, 'metadata') and match.metadata:
                        logger.info(f"Match metadata: {match.metadata}")
                    else:
                        logger.warning("Match has no metadata")
                
                # Extract texts
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
                logger.error(f"Query parameters: namespace={namespace}, top_k={top_k}, filter={filter_dict}")
                logger.error("Full error details:", exc_info=True)
                return []
                
        except Exception as e:
            logger.error(f"Error in query_pinecone: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            return []

    def generate_openai_message(self, chunks: List[str], year: str, 
                              quarter: str, query: str) -> Dict:
        """Generate message for OpenAI based on query and context."""
        try:
            # Combine chunks into context
            context = "\n\n".join(chunks)
            
            # Create system message
            system_message = f"""You are a research assistant analyzing NVIDIA's {quarter} {year} earnings call.
            Use the provided context to answer the query accurately and comprehensively.
            If the information is not available in the context, say so explicitly.
            
            Guidelines:
            1. Break down complex questions into smaller, focused sub-questions
            2. Prioritize the most relevant information
            3. Be concise while maintaining accuracy
            4. Use bullet points for multiple items
            5. Include specific numbers and metrics when available"""
            
            # Create user message with improved formatting
            user_message = f"""Context from NVIDIA's {quarter} {year} earnings call:
            {context}
            
            Query: {query}
            
            Please provide a structured response with:
            1. Key findings
            2. Supporting data
            3. Relevant metrics
            4. Important context"""
            
            return {
                "system": system_message,
                "user": user_message
            }
        except Exception as e:
            logger.error(f"Error generating OpenAI message: {str(e)}")
            raise

    def generate_model_response(self, messages: Dict[str, str], query: str) -> str:
        """Generate response using OpenAI model with improved handling of lengthy responses."""
        try:
            # Initialize response parts
            response_parts = []
            current_part = ""
            
            # Generate initial response
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": messages["system"]},
                    {"role": "user", "content": messages["user"]}
                ],
                temperature=0.7,
                max_tokens=2000,
                top_p=0.9,
                frequency_penalty=0.2,
                presence_penalty=0.2
            )
            
            # Get the initial response text
            current_part = response.choices[0].message.content
            
            # If response is too long, split it into parts
            if len(current_part.split()) > 1000:  # Rough estimate of token count
                # Split into sentences
                sentences = current_part.split('. ')
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_length = len(sentence.split())
                    if current_length + sentence_length > 1000:
                        response_parts.append('. '.join(current_chunk) + '.')
                        current_chunk = [sentence]
                        current_length = sentence_length
                    else:
                        current_chunk.append(sentence)
                        current_length += sentence_length
                
                if current_chunk:
                    response_parts.append('. '.join(current_chunk) + '.')
            else:
                response_parts.append(current_part)
            
            # If we have multiple parts, generate a summary
            if len(response_parts) > 1:
                summary_prompt = f"""Please provide a concise summary of the following information about {query}:
                {' '.join(response_parts)}"""
                
                summary_response = self.openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes information concisely."},
                        {"role": "user", "content": summary_prompt}
                    ],
                    temperature=0.5,
                    max_tokens=500
                )
                
                return summary_response.choices[0].message.content
            
            return current_part
            
        except Exception as e:
            logger.error(f"Error generating model response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def search_pinecone_db(self, year_quarter_dict: Dict[str, str], query: str, chunk_strategy: str = "markdown") -> Dict[str, Any]:
        """Search Pinecone database for relevant content."""
        try:
            logger.info(f"Starting search_pinecone_db with year_quarter_dict: {year_quarter_dict}")
            logger.info(f"Query: {query}")
            logger.info(f"Chunk strategy: {chunk_strategy}")
            
            # Validate year and quarter
            year = year_quarter_dict.get('year')
            quarter = year_quarter_dict.get('quarter')
            if not year or not quarter:
                logger.error("Missing year or quarter in year_quarter_dict")
                return {
                    "status": "error",
                    "answer": "Missing year or quarter information",
                    "metadata": {
                        "chunks_used": 0,
                        "error": "Missing year or quarter information"
                    }
                }
            
            # Check if markdown exists
            markdown_path = self.s3_manager.get_markdown_path(year, quarter)
            logger.info(f"Checking markdown at path: {markdown_path}")
            
            if not self.s3_manager.check_markdown_exists(year, quarter):
                logger.info(f"Markdown not found for {year} {quarter}, processing PDF")
                pdf_path = self.s3_manager.get_pdf_path(year, quarter)
                logger.info(f"Loading PDF from path: {pdf_path}")
                
                # Load PDF content
                pdf_content = self.s3_manager.load_s3_pdf(pdf_path)
                if not pdf_content:
                    logger.error(f"Failed to load PDF content from {pdf_path}")
                    return {
                        "status": "error",
                        "answer": f"Failed to load PDF content for {year} {quarter}",
                        "metadata": {
                            "chunks_used": 0,
                            "error": "PDF content loading failed"
                        }
                    }
                
                # Extract text from PDF
                pdf_text = self.s3_manager.get_file_content(pdf_path)
                if not pdf_text:
                    logger.error("Failed to extract text from PDF")
                    return {
                        "status": "error",
                        "answer": f"Failed to extract text from PDF for {year} {quarter}",
                        "metadata": {
                            "chunks_used": 0,
                            "error": "PDF text extraction failed"
                        }
                    }
                
                # Create markdown content
                logger.info("Creating markdown content from PDF text")
                chunks = markdown_chunking(pdf_text)
                if not chunks:
                    logger.error("No chunks created from PDF text")
                    return {
                        "status": "error",
                        "answer": f"Failed to create chunks from PDF text for {year} {quarter}",
                        "metadata": {
                            "chunks_used": 0,
                            "error": "No chunks created"
                        }
                    }
                
                logger.info(f"Created {len(chunks)} chunks from PDF text")
                
                # Save markdown content
                markdown_content = "\n\n".join(chunks)
                self.s3_manager.save_markdown_content(year, quarter, markdown_content)
                logger.info(f"Saved markdown content to {markdown_path}")
                
                # Create vector store
                self.create_pinecone_vector_store(markdown_path, chunks, chunk_strategy)
                logger.info("Created vector store in Pinecone")
            
            # Query Pinecone
            logger.info("Querying Pinecone for relevant chunks")
            relevant_chunks = self.query_pinecone(
                file=markdown_path,
                parser="mistral",
                chunking_strategy=chunk_strategy,
                query=query
            )
            
            if not relevant_chunks:
                logger.warning("No relevant chunks found in Pinecone")
                return {
                    "status": "success",
                    "answer": "No relevant information found for the query.",
                    "metadata": {
                        "chunks_used": 0,
                        "warning": "No relevant chunks found"
                    }
                }
            
            # Generate response
            logger.info(f"Generating response using {len(relevant_chunks)} chunks")
            message = self.generate_openai_message(relevant_chunks, year, quarter, query)
            response = self.generate_model_response(message, query)
            
            return {
                "status": "success",
                "answer": response,
                "metadata": {
                    "chunks_used": len(relevant_chunks),
                    "year": year,
                    "quarter": quarter
                }
            }
            
        except Exception as e:
            logger.error(f"Error in search_pinecone_db: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            return {
                "status": "error",
                "answer": f"An error occurred while processing your query: {str(e)}",
                "metadata": {
                    "chunks_used": 0,
                    "error": str(e)
                }
            }

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
        
        client = OpenAI()
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
    
    # Extract year and quarter from filename
    try:
        # Handle different file naming patterns
        if 'earnings_call' in file:
            # Pattern: nvidia_earnings_call_YYYY_QX.md
            year = file.split('_')[-2]
            quarter = file.split('_')[-1].replace('.md', '')
        elif 'report' in file:
            # Pattern: nvidia_report_YYYY_QX.md
            year = file.split('_')[-2]
            quarter = file.split('_')[-1].replace('.md', '')
        else:
            # Default pattern
            year = file.split('_')[-2]
            quarter = file.split('_')[-1].replace('.md', '')
            
        logger.info(f"Extracted year: {year}, quarter: {quarter} from file: {file}")
        
        # Validate year and quarter
        if not year.isdigit() or len(year) != 4:
            raise ValueError(f"Invalid year format: {year}")
        if not quarter.startswith('Q') or not quarter[1:].isdigit() or int(quarter[1:]) not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid quarter format: {quarter}")
            
    except Exception as e:
        logger.error(f"Error extracting year/quarter from filename {file}: {str(e)}")
        year = "unknown"
        quarter = "unknown"
    
    records = 0
    # Ensure consistent namespace naming
    namespace = f"mistral_{chunk_strategy.lower()}"  # Convert to lowercase for consistency

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
                    "parser": "mistral",
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

def save_chunks_to_s3(s3_manager: S3FileManager, file: str, chunks: List[str], chunk_strategy: str) -> bool:
    """Save chunked content back to S3 as markdown files."""
    try:
        # Extract year and quarter from filename
        year = file.split('_')[-2]
        quarter = file.split('_')[-1].replace('.md', '').replace('.pdf', '')
        
        # Create a markdown file combining all chunks
        chunked_content = f"# NVIDIA Earnings Call {year} {quarter}\n\n"
        for i, chunk in enumerate(chunks):
            chunked_content += f"\n## Chunk {i+1}\n\n{chunk}\n"
        
        # Generate filename for chunked content
        base_name = os.path.splitext(os.path.basename(file))[0]
        chunked_filename = f"{base_name}_{chunk_strategy}_chunks.md"
        
        # Save to S3
        success = s3_manager.upload_file(
            bucket_name=s3_manager.bucket_name,
            key=f"processed/{chunked_filename}",
            content=chunked_content
        )
        
        if success:
            logger.info(f"Successfully saved chunked file: {chunked_filename}")
        else:
            logger.error(f"Failed to save chunked file: {chunked_filename}")
            
        return success
    except Exception as e:
        logger.error(f"Error saving chunks to S3: {str(e)}")
        return False

def main():
    """Main function to process markdown and PDF files and store them in Pinecone."""
    base_path = "nvidia/"
    s3_obj = S3FileManager(AWS_BUCKET_NAME, base_path)
    files = list({file for file in s3_obj.list_files() if file.endswith(('.md', '.pdf'))})
    logger.info(f"Files to process: {files}")

    for i, file in enumerate(files):
        logger.info(f"Processing File {i+1}: {file}")
        try:
            content = s3_obj.get_file_content(file)
            if not content:
                logger.error(f"Failed to get content for file {file}")
                continue
            
            # Process with markdown chunking only
            logger.info("Using markdown chunking strategy...")
            chunks = markdown_chunking(content, heading_level=2)
            logger.info(f"Chunk size: {len(chunks)}")
            create_pinecone_vector_store(file, chunks, "markdown")
            save_chunks_to_s3(s3_obj, file, chunks, "markdown")
            
        except Exception as e:
            logger.error(f"Error processing file {file}: {str(e)}")
            continue

if __name__ == "__main__":
    main()