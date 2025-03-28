import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# RAG Configuration
RAG_CONFIG = {
    "chunking_strategies": {
        "markdown": {
            "heading_level": 2,
            "max_tokens": 2000
        },
        "semantic": {
            "max_sentences": 10,
            "max_tokens": 2000
        },
        "sliding_window": {
            "chunk_size": 1000,
            "overlap": 150,
            "max_tokens": 2000
        }
    },
    "embedding_model": "text-embedding-3-small",
    "embedding_dimension": 1536,
    "max_chunks_per_query": 5,
    "similarity_threshold": 0.7,
    "batch_size": 100,
    "max_files": 100,
    "supported_file_types": [".pdf", ".md"],
    "pinecone": {
        "index_name": os.getenv("PINECONE_INDEX_NAME", "nvidia-rag-pipeline"),
        "environment": os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
        "metric": "cosine",
        "cloud": "aws",
        "region": "us-east-1"
    },
    "openai": {
        "model": "gpt-4",
        "max_tokens": 2048,
        "temperature": 0.7
    }
}

# S3 Configuration
S3_CONFIG = {
    "bucket_name": os.getenv("AWS_S3_BUCKET", "nvidia-research"),
    "base_path": "nvidia-reports",
    "supported_file_types": [".pdf", ".md"],
    "max_file_size": 10 * 1024 * 1024  # 10MB
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console", "file"],
    "file_path": "logs/rag_agent.log"
}

# API Configuration
API_CONFIG = {
    "timeout": 30,
    "max_retries": 3,
    "retry_delay": 1,
    "rate_limit": {
        "requests_per_minute": 60,
        "requests_per_hour": 1000
    }
} 