# NVIDIA Research Assistant

A comprehensive research assistant that analyzes NVIDIA's quarterly reports and historical data using RAG (Retrieval-Augmented Generation), Snowflake data, and web search capabilities.

## Features

- 📊 Historical Financial Data Analysis (Snowflake)
- 📚 Quarterly Reports Analysis (Pinecone RAG)
- 🌐 Real-time Web Data Search
- 🔄 Combined Analysis Mode
- 📈 Interactive Visualizations
- 🔍 Multiple Chunking Strategies

## Prerequisites

- Python 3.8+
- AWS Account with S3 access
- Pinecone Account
- OpenAI API Key
- Snowflake Account

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
AWS_S3_BUCKET=nvidia-research

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_environment
PINECONE_INDEX_NAME=nvidia-rag-pipeline

# OpenAI Configuration
OPENAI_API_KEY=your_openai_key

# FastAPI Configuration
FASTAPI_URL=http://localhost:8000
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nvidia-research-assistant.git
cd nvidia-research-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt')"
```

## Running the Application

1. Start the FastAPI backend:
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Start the Streamlit frontend:
```bash
cd frontend
streamlit run app.py
```

3. Open your browser and navigate to:
```
http://localhost:8501
```

## Usage

1. Select the research mode:
   - Pinecone (Quarterly Reports)
   - Snowflake (Financial Data)
   - Web Search (Real-time Data)
   - Combined Analysis

2. Choose the year and quarter for analysis

3. For Pinecone and Combined modes, select a chunking strategy:
   - Markdown
   - Semantic
   - Sliding Window

4. Enter your research question

5. Click "Generate Research Report"

## Project Structure

```
nvidia-research-assistant/
├── backend/
│   ├── agents/
│   │   ├── rag_agent.py
│   │   ├── snowflake_agent.py
│   │   └── web_agent.py
│   ├── core/
│   │   ├── s3_client.py
│   │   └── text_chunking.py
│   ├── config/
│   │   └── rag_config.py
│   └── main.py
├── frontend/
│   └── app.py
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.