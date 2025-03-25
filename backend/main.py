import os
import asyncio
import chromadb
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from core.s3_client import S3FileManager
from features.chunking_stratergy import markdown_chunking, semantic_chunking, sliding_window_chunking
from agents.rag_agent import connect_to_pinecone_index, get_embedding, query_pinecone
from features.mistral_parser import pdf_mistralocr_converter

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

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
        
        # Generate chunks using the specified strategy
        chunks = generate_chunks(markdown_content, chunk_strategy)
        print(f"Generated {len(chunks)} chunks using {chunk_strategy} strategy")
        
        if vector_store == "pinecone":
            # Create vector store and query
            await create_pinecone_vector_store(file_name, chunks, chunk_strategy, parser)
            result_chunks = query_pinecone_doc(file=file_name, parser=parser, chunking_strategy=chunk_strategy, query=query, top_k=top_k)
            
            if len(result_chunks) == 0:
                raise HTTPException(status_code=500, detail="No relevant data found in the document")
            
            # Generate response using OpenAI
            message = generate_openai_message_document(query, result_chunks)
            answer = generate_model_response(message)
            
        elif vector_store == "chromadb":
            s3_obj = await create_chromadb_vector_store(file_name, chunks, chunk_strategy, parser)
            result_chunks = query_chromadb_doc(file_name, parser, chunk_strategy, query, top_k, s3_obj)
            message = generate_openai_message_document(query, result_chunks)
            answer = generate_model_response(message)
            
        else:
            raise HTTPException(status_code=400, detail="Invalid vector store type. Supported types: pinecone, chromadb")

        return {"answer": answer}
    
    except Exception as e:
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
        base_path = "nvidia-reports"
        print(f"Using base path: {base_path}")
        s3_obj = S3FileManager(AWS_BUCKET_NAME, base_path)
        
        # Get the PDF content from S3 and process it with Mistral
        pdf_filename = f"nvidia_raw_pdf_{year}_Q{quarter[0]}.pdf"
        print(f"Attempting to load PDF from: {base_path}/{pdf_filename}")
        pdf_content = s3_obj.load_s3_pdf(pdf_filename)
        if not pdf_content:
            raise HTTPException(status_code=404, detail=f"NVIDIA report for {year} Q{quarter[0]} not found at {base_path}/{pdf_filename}")
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
            file_name = f"{year}_Q{quarter[0]}"
            print(f"Creating Pinecone vector store for file: {file_name}")
            await create_pinecone_vector_store(file_name, chunks, chunk_strategy, parser)
            
            # Query using the same namespace format
            print(f"Querying Pinecone with namespace format: {parser}_sliding_window")
            result_chunks = query_pinecone_doc(
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
    if chunk_strategy == "markdown":
        return markdown_chunking(markdown_content, heading_level=2)
    elif chunk_strategy == "semantic":
        return semantic_chunking(markdown_content, max_sentences=10)
    elif chunk_strategy == "sliding_window":
        return sliding_window_chunking(markdown_content, chunk_size=1000, overlap=150)
    else:
        raise HTTPException(status_code=400, detail="Invalid chunk strategy. Supported strategies: markdown, semantic, sliding_window")

def generate_openai_message_document(query, chunks):
    prompt = f"""
    Below are relevant excerpts from a document uploaded by the user that may help answer the user query.

    --- User Query ---
    {query}

    --- Relevant Document Chunks ---
    {chr(10).join([f'Chunk {i+1}: {chunk}' for i, chunk in enumerate(chunks)])}

    Based on the provided document chunks, generate a comprehensive response to the query. If needed, synthesize the information and ensure clarity.
    """
    return prompt
    
def generate_openai_message(chunks, year, quarter, query):
    prompt = f"""
    Below are relevant excerpts from a NVDIA quarterly financial report for year {year} and quarter {quarter} that may help answer the query.

    --- User Query ---
    {query}

    --- Relevant Document Chunks ---
    {chr(10).join([f'Chunk {i+1}: {chunk}' for i, chunk in enumerate(chunks)])}

    Based on the provided document chunks, generate a comprehensive response to the query. If needed, synthesize the information and ensure clarity.
    """
    return prompt

async def create_pinecone_vector_store(file, chunks, chunk_strategy, parser):
    index = connect_to_pinecone_index()
    # Use consistent namespace format with underscore
    namespace = f"{parser}_sliding_window" if chunk_strategy == "sliding_window" else f"{parser}_{chunk_strategy}"
    print(f"Creating vectors in namespace: {namespace}")
    
    vectors = []
    records = 0
    
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        vectors.append((
            f"{file}_chunk_{i}",  # Unique ID
            embedding,  # Embedding vector
            {
                "file": file,
                "text": chunk,
                "parser": parser,
                "strategy": chunk_strategy
            }  # Metadata
        ))
        
        if len(vectors) >= 20:
            records += len(vectors)
            upsert_vectors(index, vectors, namespace)
            print(f"Inserted {len(vectors)} chunks into Pinecone.")
            vectors.clear()
    
    if vectors:
        upsert_vectors(index, vectors, namespace)
        print(f"Inserted {len(vectors)} chunks into Pinecone.")
        records += len(vectors)
    
    print(f"Total inserted {records} chunks into Pinecone namespace {namespace}")

def upsert_vectors(index, vectors, namespace):
    index.upsert(vectors=vectors, namespace=namespace)

def query_pinecone_doc(file, parser, chunking_strategy, query, top_k=10):
    index = connect_to_pinecone_index()
    dense_vector = get_embedding(query)
    # Use consistent namespace format with underscore
    namespace = f"{parser}_sliding_window" if chunking_strategy == "sliding_window" else f"{parser}_{chunking_strategy}"
    
    print(f"Querying Pinecone with namespace: {namespace}")
    
    # First try with file filter
    results = index.query(
        namespace=namespace,
        vector=dense_vector,
        filter={"file": {"$eq": file}},  # Filter by file name
        top_k=top_k,
        include_metadata=True
    )
    
    matches = [match['metadata']['text'] for match in results["matches"]]
    print(f"Found {len(matches)} matches in namespace {namespace} with file filter")
    
    # If no matches found, try without file filter
    if not matches:
        print("No matches found with file filter, trying without filter...")
        results = index.query(
            namespace=namespace,
            vector=dense_vector,
            top_k=top_k,
            include_metadata=True
        )
        matches = [match['metadata']['text'] for match in results["matches"]]
        print(f"Found {len(matches)} matches in namespace {namespace} without file filter")
    
    if not matches:
        # List all vectors in the namespace to help debug
        try:
            stats = index.describe_index_stats()
            print(f"Index stats: {stats}")
            raise HTTPException(
                status_code=500, 
                detail=f"No relevant data found in the document (namespace: {namespace}, vectors in namespace: {stats.namespaces.get(namespace, {'vector_count': 0})['vector_count']})"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"No relevant data found in the document (namespace: {namespace})")
        
    return matches

async def create_chromadb_vector_store(file, chunks, chunk_strategy, parser):
    with tempfile.TemporaryDirectory() as temp_dir:
        chroma_client = chromadb.PersistentClient(path=temp_dir)
        file_name = file.split('/')[2]
        base_path = "/".join(file.split('/')[:-1])
        s3_obj = S3FileManager(AWS_BUCKET_NAME, base_path)
        
        collection_file = chroma_client.get_or_create_collection(name=f"{file_name}_{parser}_{chunk_strategy}")
        base_metadata = {"file": file_name}
        metadata = [base_metadata for _ in range(len(chunks))]
        
        embeddings = get_chroma_embeddings(chunks)
        ids = [f"{file_name}_{parser}_{chunk_strategy}_{i}" for i in range(len(chunks))]
        
        collection_file.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadata,
            documents=chunks
        )
        
        upload_directory_to_s3(temp_dir, s3_obj, "chroma_db")
        print("ChromaDB has been uploaded to S3.")
        return s3_obj

def upload_directory_to_s3(local_dir, s3_obj, s3_prefix):
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = f"{s3_obj.base_path}/{os.path.join(s3_prefix, relative_path)}".replace("\\", "/")
            
            with open(local_path, "rb") as f:
                s3_obj.upload_file(AWS_BUCKET_NAME, s3_key, f.read())

def download_chromadb_from_s3(s3_obj, temp_dir):
    s3_prefix = f"{s3_obj.base_path}/chroma_db"
    s3_files = [f for f in s3_obj.list_files() if f.startswith(s3_prefix)]
    
    for s3_file in s3_files:
        relative_path = s3_file[len(s3_prefix):].lstrip('/')
        local_path = os.path.join(temp_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        content = s3_obj.load_s3_pdf(s3_file)
        with open(local_path, 'wb') as f:
            f.write(content if isinstance(content, bytes) else content.encode('utf-8'))

def query_chromadb_doc(file_name, parser, chunking_strategy, query, top_k, s3_obj):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            download_chromadb_from_s3(s3_obj, temp_dir)
            chroma_client = chromadb.PersistentClient(path=temp_dir)
            file_name = file_name.split('/')[2]

            try:
                collection = chroma_client.get_collection(f"{file_name}_{parser}_{chunking_strategy}")
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")
            
            query_embeddings = get_chroma_embeddings([query])
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k
            )
            
            return results["documents"][0]  # Return first list of documents
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error querying ChromaDB: {str(e)}")

def generate_model_response(message):
    try:
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
