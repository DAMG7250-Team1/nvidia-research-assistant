import os
import sys
from io import BytesIO
from dotenv import load_dotenv
from backend.core.s3_client import S3FileManager
from backend.features.mistral_parser import pdf_mistralocr_converter
from backend.features.chunking_stratergy import markdown_chunking, semantic_chunking, sliding_window_chunking
from backend.agents.rag_agent import connect_to_pinecone_index, get_embedding, query_pinecone
from backend.agents.rag_agent import create_pinecone_vector_store

load_dotenv()
print("OpenAI Key:", os.getenv("OPENAI_API_KEY")[:5] + "..." + os.getenv("OPENAI_API_KEY")[-4:])

def test_s3_connection():
    """Test S3 connection and basic operations"""
    print("\n=== Testing S3 Connection ===")
    try:
        bucket_name = os.getenv("AWS_BUCKET_NAME")
        s3_obj = S3FileManager(bucket_name)
        files = s3_obj.list_files()
        print(f"Successfully connected to S3. Found {len(files)} files:")
        for file in files:
            print(f"- {file}")
        return True
    except Exception as e:
        print(f"S3 connection failed: {str(e)}")
        return False

def test_pdf_processing(filename: str):
    """Test PDF processing with Mistral parser"""
    print(f"\n=== Testing PDF Processing for {filename} ===")
    try:
        bucket_name = os.getenv("AWS_BUCKET_NAME")
        s3_obj = S3FileManager(bucket_name)
        
        # Load PDF
        pdf_content = s3_obj.load_s3_pdf(filename)
        if not pdf_content:
            print(f"Failed to load PDF: {filename}")
            return False
            
        pdf_stream = BytesIO(pdf_content)
        base_name = filename.split('.')[0]
        
        # Test Mistral Parser
        print("\nTesting Mistral Parser...")
        try:
            output_path = f"nvidia-reports/processed/mistral/{base_name}"
            file_name, markdown_content = pdf_mistralocr_converter(pdf_stream, output_path, s3_obj)
            print(f"Mistral parsing successful. Output file: {file_name}")
            print(f"Content length: {len(markdown_content)} characters")
            return markdown_content  # Return the content directly
        except Exception as e:
            print(f"Mistral parser failed: {str(e)}")
            return None
            
    except Exception as e:
        print(f"PDF processing test failed: {str(e)}")
        return None

def test_chunking(markdown_content: str):
    """Test different chunking strategies"""
    print("\n=== Testing Chunking Strategies ===")
    try:
        # Test markdown chunking
        print("\nTesting Markdown Chunking...")
        md_chunks = markdown_chunking(markdown_content)
        print(f"Generated {len(md_chunks)} markdown chunks")
        
        # Test semantic chunking
        print("\nTesting Semantic Chunking...")
        sem_chunks = semantic_chunking(markdown_content)
        print(f"Generated {len(sem_chunks)} semantic chunks")
        
        # Test sliding window chunking
        print("\nTesting Sliding Window Chunking...")
        sw_chunks = sliding_window_chunking(markdown_content)
        print(f"Generated {len(sw_chunks)} sliding window chunks")
        
        return True
    except Exception as e:
        print(f"Chunking test failed: {str(e)}")
        return False

def test_vector_store(chunks: list, file_name: str):
    """Test vector store operations"""
    print("\n=== Testing Vector Store Operations ===")
    try:
        # Test Pinecone connection
        print("\nTesting Pinecone connection...")
        index = connect_to_pinecone_index()
            
        # Test embedding generation
        print("\nTesting embedding generation...")
        sample_chunk = chunks[0]
        embedding = get_embedding(sample_chunk)
        print(f"Successfully generated embedding of size {len(embedding)}")
        
        # Extract year and quarter from filename
        # Example filename: nvidia-reports/nvidia_raw_pdf_2020_Q1.pdf
        file_parts = file_name.split('/')
        identifier = file_parts[-1]  # nvidia_raw_pdf_2020_Q1.pdf
        year = identifier.split('_')[-2]  # 2020
        quarter = identifier.split('_')[-1].split('.')[0]  # Q1
        
        # Use actual document info for namespace
        parser_name = "mistral"
        strategy_name = "markdown"
        namespace = f"{parser_name}_{strategy_name}"
        
        print(f"\nProcessing document from {year} {quarter}")
        print(f"Using namespace: {namespace}")
        print(f"Parser: {parser_name}, Strategy: {strategy_name}")
        
        # Test vector storage
        print("\nTesting vector storage...")
        vectors = []
        for i, chunk in enumerate(chunks[:5]):  # Test with first 5 chunks
            print(f"\nProcessing chunk {i+1}/5:")
            print(f"Chunk length: {len(chunk)} characters")
            print(f"First 100 chars: {chunk[:100]}...")
            
            embedding = get_embedding(chunk)[:1024]  # Trim to 1024 to match index dimension
            print(f"Generated embedding of size {len(embedding)}")
            
            vectors.append((
                f"{file_name}_chunk_{i}",
                embedding,
                {
                    "text": chunk,
                    "year": year,
                    "quarter": quarter,
                    "parser": parser_name,
                    "strategy": strategy_name
                }
            ))
        
        print(f"\nUploading {len(vectors)} vectors to Pinecone...")
        index.upsert(vectors=vectors, namespace=namespace)
        print(f"Successfully stored {len(vectors)} vectors")
        
        # Test query
        print("\nTesting vector query...")
        test_query = "What was the revenue?"
        print(f"Query: '{test_query}'")
        print(f"Using namespace: {namespace}")
        print(f"Year: {year}, Quarter: {quarter}")
        
        results = query_pinecone(
            parser=parser_name,
            chunking_strategy=strategy_name,
            query=test_query,
            year=year,
            quarter=[quarter]
        )
        
        print(f"\nQuery returned {len(results)} results")
        if len(results) > 0:
            print("\nFirst result:")
            print(results[0][:500] + "..." if len(results[0]) > 500 else results[0])
        else:
            print("No results found. This could be because:")
            print("1. The vectors were not properly stored")
            print("2. The query parameters don't match the stored metadata")
            print("3. The content doesn't contain relevant information")
        
        return True
    except Exception as e:
        print(f"Vector store test failed: {str(e)}")
        print(f"Error details: {str(e.__class__.__name__)}")
        return False

def main():
    """Main test function"""
    print("Starting pipeline tests...")
    
    # Test S3
    if not test_s3_connection():
        print("S3 test failed. Stopping tests.")
        return
        
    # Get a test file
    bucket_name = os.getenv("AWS_BUCKET_NAME")
    s3_obj = S3FileManager(bucket_name)
    files = s3_obj.list_files()
    if not files:
        print("No files found in S3. Stopping tests.")
        return
        
    test_file = files[0]
    print(f"\nUsing test file: {test_file}")
    
    # Test PDF processing and get markdown content directly
    markdown_content = test_pdf_processing(test_file)
    if not markdown_content:
        print("PDF processing test failed. Stopping tests.")
        return
        
    # Test chunking with the markdown content we got directly
    if not test_chunking(markdown_content):
        print("Chunking test failed. Stopping tests.")
        return
        
    # Get chunks for vector store test
    chunks = markdown_chunking(markdown_content)
    if not test_vector_store(chunks, test_file):
        print("Vector store test failed.")
        return
        
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main() 