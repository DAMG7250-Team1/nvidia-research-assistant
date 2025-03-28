# import streamlit as st
# import requests
# import json
# from datetime import datetime
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Constants
# FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# # Page config
# st.set_page_config(
#     page_title="NVIDIA Research Assistant",
#     page_icon="🚀",
#     layout="wide"
# )

# # Title and description
# st.title("🚀 NVIDIA Research Assistant")
# st.markdown("""
# This research assistant analyzes NVIDIA's quarterly reports to provide insights and answer questions about their financial performance.

# **Available Reports**: 2020-2025 quarterly reports
# """)

# # Sidebar for filters
# with st.sidebar:
#     st.header("Filters")
    
#     # Year selection
#     years = list(range(2020, 2026))  # Show all years from 2020 to 2025
#     selected_year = st.selectbox("Select Year", years, index=2)  # Default to 2022
    
#     # Quarter selection
#     quarters = ["Q1", "Q2", "Q3", "Q4"]
#     selected_quarter = st.selectbox("Select Quarter", quarters)
    
#     # Chunking strategy
#     st.header("RAG Settings")
#     chunk_strategy = st.selectbox(
#         "Chunking Strategy",
#         ["markdown", "semantic", "sliding_window"],
#         help="Choose how to split the documents for analysis"
#     )

# # Main content area
# st.header("Research Query")
# query = st.text_area(
#     "Enter your research question",
#     placeholder="e.g., What was NVIDIA's revenue breakdown by segment in Q2 2022?",
#     height=100
# )

# if st.button("Generate Research Report"):
#     if not query:
#         st.error("Please enter a research question")
#     else:
#         with st.spinner("Generating research report..."):
#             try:
#                 # Extract quarter number from selected_quarter (e.g., "Q2" -> "2")
#                 quarter_number = selected_quarter[1]
#                 # Ensure we're sending the correct year
#                 year = str(selected_year)
#                 print(f"Sending request for year: {year}, quarter: {quarter_number}")  # Debug print
#                 response = requests.post(
#                     f"{FASTAPI_URL}/query_nvdia_documents",
#                     json={
#                         "year": year,
#                         "quarter": [f"Q{quarter_number}"],  # Send with Q prefix
#                         "parser": "mistral",
#                         "chunk_strategy": chunk_strategy,
#                         "vector_store": "pinecone",
#                         "query": query
#                     }
#                 )
#                 if response.status_code == 200:
#                     st.markdown(response.json()["answer"])
#                 else:
#                     error_msg = response.text
#                     if "404" in error_msg:
#                         st.error(f"Report not found. Please select a year between 2020-2025 and ensure the report exists for that period.")
#                     else:
#                         st.error(f"Error: {error_msg}")
#             except Exception as e:
#                 st.error(f"Failed to get response: {str(e)}")

# # Footer
# st.markdown("---")
# st.markdown("""
# <div style='text-align: center'>
#     <p>Built with ❤️ using FastAPI and Streamlit</p>
#     <p>NVIDIA Research Assistant v1.0</p>
# </div>
# """, unsafe_allow_html=True)


import streamlit as st
import requests
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import io

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
This research assistant analyzes NVIDIA's quarterly reports and historical data to provide insights and answer questions about their financial performance.

**Available Reports**: 2020-2023 quarterly reports
**Available Data Sources**: 
- 📊 Historical Financial Data (Snowflake) - 2020-2023
- 📚 Quarterly Reports (Pinecone RAG) - 2020-2023
- 🌐 Real-time Web Data (Web Search) - Current information
""")

# Sidebar for filters
with st.sidebar:
    st.header("Research Configuration")
    
    # Year selection (only show available years)
    years = list(range(2020, 2025))  # Show years from 2020 to 2024
    selected_year = st.selectbox("Select Year", years, index=2)  # Default to 2022
    
    # Quarter selection
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    selected_quarter = st.selectbox("Select Quarter", quarters)
    
    # Add validation for current year
    current_year = datetime.now().year
    current_quarter = (datetime.now().month - 1) // 3 + 1
    
    if selected_year == current_year:
        # If current year, only show quarters up to current quarter
        available_quarters = [f"Q{i}" for i in range(1, current_quarter + 1)]
        if selected_quarter not in available_quarters:
            st.warning(f"⚠️ Data for {selected_year} {selected_quarter} is not yet available.")
            selected_quarter = available_quarters[0]  # Reset to first available quarter
    
    # Research Mode selection
    st.header("Research Mode")
    selected_agent = st.selectbox(
        "Select Agent",
        ["Pinecone (Quarterly Reports)", "Snowflake (Financial Data)", "Web Search (Real-time Data)", "Combined Analysis"],
        help="Choose which agent to use for analysis"
    )

# Main content area
st.header("Research Query")
if selected_agent == "Snowflake (Financial Data)":
    st.info("""
    You are a structured data extraction agent. Your task is to identify relevant financial metrics from a user prompt and map them to the provided Snowflake schema.

    ### Available Metrics:
    - **Close**: Daily closing price
    - **Open**: Daily opening price
    - **High**: Highest price during the day
    - **Low**: Lowest price during the day
    - **Volume**: Number of shares traded
    - **Adjusted Close**: Closing price adjusted for splits and dividends
    """)
    
    query = st.text_area(
        "Enter your research question",
        placeholder="e.g., How is NVIDIA performing in terms of closing prices?",
        height=100
    )
else:
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
                # Map frontend agent names to backend modes
                agent_to_mode = {
                    "Pinecone (Quarterly Reports)": "rag",
                    "Snowflake (Financial Data)": "snowflake",
                    "Web Search (Real-time Data)": "web",
                    "Combined Analysis": "combined"
                }
                
                mode = agent_to_mode.get(selected_agent, "combined")
                
                # Prepare metadata filters
                metadata_filters = {
                    "year": str(selected_year),
                    "quarter": selected_quarter,
                    "chunk_strategy": "markdown"  # Always use markdown chunking
                }
                
                # Add Snowflake-specific filters
                if mode == "snowflake":
                    # Remove metric and chart type selection for Snowflake agent
                    st.info("""
                    The Snowflake agent will automatically analyze all available metrics:
                    - CLOSE: Daily closing price
                    - OPEN: Daily opening price
                    - HIGH: Highest price during the day
                    - LOW: Lowest price during the day
                    - VOLUME: Number of shares traded
                    - ADJUSTEDCLOSE: Closing price adjusted for splits and dividends
                    """)
                    metadata_filters.update({
                        "query": query  # Pass the user's query for analysis
                    })
                
                # Debug print
                print("\n" + "="*80)
                print("📤 SENDING REQUEST TO BACKEND")
                print("="*80)
                print(f"URL: {FASTAPI_URL}/research")
                print(f"Query: {query}")
                print(f"Mode: {mode}")
                print(f"Metadata filters: {metadata_filters}")
                print("="*80 + "\n")
                
                # Call the research graph endpoint
                response = requests.post(
                    f"{FASTAPI_URL}/research",
                    json={
                        "query": query,
                        "year": str(selected_year),
                        "quarter": selected_quarter,
                        "agent_type": mode,
                        "metadata_filters": metadata_filters
                    }
                )
                
                # Debug print response
                print("\n" + "="*80)
                print("📥 RECEIVED RESPONSE FROM BACKEND")
                print("="*80)
                print(f"Status code: {response.status_code}")
                print(f"Response: {response.text}")
                print("="*80 + "\n")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Received response: {result}")  # Debug print
                    
                    # Display the response with better formatting
                    st.markdown("### Research Report")
                    
                    # Display final report
                    if "final_report" in result and result["final_report"]:
                        st.markdown(result["final_report"])
                    else:
                        st.warning("No final report generated")
                    
                    # Display RAG results if available
                    if "rag_results" in result and result["rag_results"]:
                        st.markdown("### RAG Results")
                        if "answer" in result["rag_results"]:
                            st.markdown(result["rag_results"]["answer"])
                        if "error" in result["rag_results"]:
                            st.error(f"RAG Error: {result['rag_results']['error']}")
                    
                    # Display Snowflake results if available
                    if "snowflake_results" in result and result["snowflake_results"]:
                        st.markdown("### Financial Data")
                        if "summary" in result["snowflake_results"]:
                            st.markdown(result["snowflake_results"]["summary"])
                        if "error" in result["snowflake_results"]:
                            st.error(f"Snowflake Error: {result['snowflake_results']['error']}")
                    
                    # Display Web results if available
                    if "web_results" in result and result["web_results"]:
                        st.markdown("### Web Search Results")
                        if "answer" in result["web_results"]:
                            st.markdown(result["web_results"]["answer"])
                        if "error" in result["web_results"]:
                            st.error(f"Web Search Error: {result['web_results']['error']}")
                    
                    # Display visualizations if available
                    if "snowflake_results" in result and "visualizations" in result["snowflake_results"]:
                        st.markdown("### Visualizations")
                        for viz in result["snowflake_results"]["visualizations"]:
                            if viz["type"] == "chart":
                                st.markdown(f"#### {viz['title']}")
                                if "url" in viz and viz["url"]:
                                    st.image(viz["url"], caption=viz["title"])
                                else:
                                    st.warning(f"Visualization URL not available for {viz['title']}")
                    
                    # Display raw data if available
                    if "snowflake_results" in result and "data" in result["snowflake_results"]:
                        st.markdown("### Raw Data")
                        df = pd.DataFrame(result["snowflake_results"]["data"])
                        st.dataframe(df)
                    
                else:
                    error_msg = response.text
                    print(f"Error response: {error_msg}")
                    try:
                        error_json = json.loads(error_msg)
                        error_detail = error_json.get("detail", error_msg)
                        
                        # Show different error messages based on the error type
                        if "No data available" in error_detail:
                            st.error(f"""
                            ⚠️ No data available for the selected period.
                            
                            Please try:
                            1. Select a different year/quarter
                            2. Use a different research mode (Web Search or Snowflake)
                            3. Contact support if the issue persists
                            
                            Error details: {error_detail}
                            """)
                        else:
                            st.error(f"Error generating report: {error_detail}")
                    except json.JSONDecodeError:
                        st.error(f"Error: {error_msg}")
                        
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend server. Please make sure the FastAPI server is running.")
            except Exception as e:
                st.error(f"Failed to get response: {str(e)}")
                print(f"Exception details: {str(e)}")
                import traceback
                traceback.print_exc()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using FastAPI, Streamlit, and LangGraph</p>
    <p>NVIDIA Research Assistant v2.0</p>
</div>
""", unsafe_allow_html=True)
































