# NVIDIA Research Assistant

A powerful RAG (Retrieval Augmented Generation) system for analyzing NVIDIA's quarterly earnings reports and financial documents.

## Features

- **Document Processing**: Efficiently processes NVIDIA's quarterly earnings reports and financial documents
- **Multiple Chunking Strategies**: Supports various text chunking methods:
  - Markdown-based chunking
  - Semantic chunking
  - Sliding window chunking
- **Vector Storage**: Uses Pinecone for efficient vector storage and retrieval
- **Advanced Querying**: Natural language queries with context-aware responses
- **Multiple Parser Support**: Currently supports Mistral parser for document processing

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env`:
   ```
   MISTRAL_API_KEY=your_mistral_api_key
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   AWS_BUCKET_NAME=your_bucket_name
   ```

## Usage

1. Start the backend server:
   ```bash
   cd backend
   python main.py
   ```

2. Start the frontend:
   ```bash
   cd frontend
   streamlit run app.py
   ```

3. Access the application at `http://localhost:8501`

## Project Structure

```
nvidia-research-assistant/
├── backend/
│   ├── agents/
│   │   └── rag_agent.py
│   ├── core/
│   │   └── s3_client.py
│   ├── features/
│   │   └── mistral_parser.py
│   └── main.py
├── frontend/
│   └── app.py
└── requirements.txt
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request