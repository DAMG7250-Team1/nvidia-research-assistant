import streamlit as st
import requests
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="NVIDIA Research Assistant",
    page_icon="🚀",
    layout="wide"
)

# Title and description
st.title("🚀 NVIDIA Research Assistant")
st.markdown("""
This research assistant analyzes NVIDIA's quarterly reports to provide insights and answer questions about their financial performance.

**Available Reports**: 2020-2025 quarterly reports
""")

# Sidebar for filters
with st.sidebar:
    st.header("Filters")
    
    # Year selection
    years = list(range(2020, 2026))  # Show all years from 2020 to 2025
    selected_year = st.selectbox("Select Year", years, index=2)  # Default to 2022
    
    # Quarter selection
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    selected_quarter = st.selectbox("Select Quarter", quarters)
    
    # Chunking strategy
    st.header("RAG Settings")
    chunk_strategy = st.selectbox(
        "Chunking Strategy",
        ["markdown", "semantic", "sliding_window"],
        help="Choose how to split the documents for analysis"
    )

# Main content area
st.header("Research Query")
query = st.text_area(
    "Enter your research question",
    placeholder="e.g., What was NVIDIA's revenue breakdown by segment in Q2 2022?",
    height=100
)

if st.button("Generate Research Report"):
    if not query:
        st.error("Please enter a research question")
    else:
        with st.spinner("Generating research report..."):
            try:
                # Extract quarter number from selected_quarter (e.g., "Q2" -> "2")
                quarter_number = selected_quarter[1]
                # Ensure we're sending the correct year
                year = str(selected_year)
                print(f"Sending request for year: {year}, quarter: {quarter_number}")  # Debug print
                response = requests.post(
                    f"{FASTAPI_URL}/query_nvdia_documents",
                    json={
                        "year": year,
                        "quarter": [f"Q{quarter_number}"],  # Send with Q prefix
                        "parser": "mistral",
                        "chunk_strategy": chunk_strategy,
                        "vector_store": "pinecone",
                        "query": query
                    }
                )
                if response.status_code == 200:
                    st.markdown(response.json()["answer"])
                else:
                    error_msg = response.text
                    if "404" in error_msg:
                        st.error(f"Report not found. Please select a year between 2020-2025 and ensure the report exists for that period.")
                    else:
                        st.error(f"Error: {error_msg}")
            except Exception as e:
                st.error(f"Failed to get response: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using FastAPI and Streamlit</p>
    <p>NVIDIA Research Assistant v1.0</p>
</div>
""", unsafe_allow_html=True)
