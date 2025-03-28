import sys
from pathlib import Path

# Add project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from dotenv import load_dotenv
import os
import pandas as pd
import snowflake.connector
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from datetime import datetime
import io
from typing import Dict, Any

# Load credentials from .env
load_dotenv()

def query_snowflake(query: str) -> pd.DataFrame:
    """Query Snowflake and return results as a DataFrame or cursor."""
    try:
        print("\n🔍 Attempting to connect to Snowflake...")
        # Configure Snowflake connection for Windows
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse='COMPUTE_WH',
            database='FINANCE_MARKET_DATA',
            schema='YAHOO_MARKET_DATA',
            client_session_keep_alive=True,
            login_timeout=60,
            network_timeout=60,
            application='NVIDIA_RESEARCH_ASSISTANT'
        )
        
        print("✅ Successfully connected to Snowflake")
        
        # Create cursor
        cursor = conn.cursor()
        
        # Execute query
        print(f"\n🔍 Executing query: {query}")
        cursor.execute(query)
        
        # For SHOW TABLES and similar commands, return the cursor
        if query.upper().startswith("SHOW"):
            print("✅ Query executed successfully")
            return cursor
            
        # For SELECT queries, return DataFrame
        df = pd.DataFrame.from_records(iter(cursor), columns=[x[0] for x in cursor.description])
        cursor.close()
        conn.close()
        print("✅ Query executed successfully")

        # Ensure column names are always uppercase
        df.columns = df.columns.str.upper()
        return df
        
    except snowflake.connector.errors.DatabaseError as e:
        print(f"\n❌ Database Error: {str(e)}")
        raise
    except snowflake.connector.errors.ProgrammingError as e:
        print(f"\n❌ Programming Error: {str(e)}")
        raise
    except Exception as e:
        print(f"\n❌ Error connecting to Snowflake: {str(e)}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()
        raise

def generate_chart(df: pd.DataFrame, metric: str, year: str, quarter: str, chart_type: str = "line") -> str:
    """Generate a chart from the DataFrame and save it to S3."""
    try:
        df = df.sort_values("ASOFDATE")
        df["ASOFDATE"] = pd.to_datetime(df["ASOFDATE"])

        # Set style for better visualization
        plt.style.use('seaborn-v0_8')
        
        # Create figure with larger size
        plt.figure(figsize=(15, 8))

        # Plot metric
        if chart_type == "line":
            plt.plot(df["ASOFDATE"], df[metric], marker="o", linewidth=2, label=metric)
        elif chart_type == "bar":
            plt.bar(df["ASOFDATE"], df[metric], label=metric)

        # Customize plot
        plt.title(f"NVIDIA Stock {metric} Over Time", fontsize=14, pad=20)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)

        # Format y-axis for large numbers
        if metric == "VOLUME":
            plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
        else:
            plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:.2f}'))

        plt.tight_layout()
        
        # Create a BytesIO object to store the image
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Reset buffer position
        img_buffer.seek(0)
        
        # Initialize S3 client
        from backend.core.s3_client import S3FileManager
        s3_manager = S3FileManager("nvidia-research")
        
        # Generate S3 path
        s3_path = f"nvidia-reports/processed/snowflake_agents_chart/{metric}_{year}_{quarter}.png"
        
        # Upload to S3
        s3_manager.upload_file(
            bucket_name="nvidia-research",
            key=s3_path,
            content=img_buffer.getvalue()
        )
        
        # Return the S3 URL
        return f"s3://nvidia-research/{s3_path}"
        
    except Exception as e:
        print(f"Error generating chart: {str(e)}")
        raise

def get_nvidia_historical(year: str, quarter: str) -> Dict[str, Any]:
    """Get NVIDIA historical stock data and generate charts."""
    try:
        # Initialize Snowflake connection
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database="FINANCE_MARKET_DATA",
            schema="YAHOO_MARKET_DATA"
        )
        
        # Determine date range for the quarter
        if quarter == 'Q1':
            start_date = f"{year}-01-01"
            end_date = f"{year}-03-31"
        elif quarter == 'Q2':
            start_date = f"{year}-04-01"
            end_date = f"{year}-06-30"
        elif quarter == 'Q3':
            start_date = f"{year}-07-01"
            end_date = f"{year}-09-30"
        else:  # Q4
            start_date = f"{year}-10-01"
            end_date = f"{year}-12-31"
        
        # Query historical data
        query = f"""
        SELECT 
            ASOFDATE,
            OPEN,
            HIGH,
            LOW,
            CLOSE,
            VOLUME,
            ADJUSTEDCLOSE,
            DIVIDENDS,
            SPLITS
        FROM NVIDIA_HISTORICAL
        WHERE ASOFDATE BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY ASOFDATE
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Generate charts for all metrics
        chart_paths = {}
        metrics = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "ADJUSTEDCLOSE"]
        
        for metric in metrics:
            if metric in df.columns:
                try:
                    chart_path = generate_chart(df, metric=metric, year=year, quarter=quarter, chart_type="line")
                    chart_paths[metric] = chart_path
                    print(f"✅ Chart saved for {metric}: {chart_path}")
                except Exception as e:
                    print(f"❌ Error generating chart for {metric}: {str(e)}")
                    chart_paths[metric] = None
        
        # Generate summary
        summary = f"NVIDIA Stock Historical Data Analysis for {year} {quarter}\n\n"
        summary += f"Data Points: {len(df)}\n"
        summary += f"Date Range: {df['ASOFDATE'].min().strftime('%Y-%m-%d')} to {df['ASOFDATE'].max().strftime('%Y-%m-%d')}\n\n"
        
        for metric in metrics:
            if metric in df.columns:
                summary += f"{metric}:\n"
                summary += f"  Min: {df[metric].min():.2f}\n"
                summary += f"  Max: {df[metric].max():.2f}\n"
                summary += f"  Avg: {df[metric].mean():.2f}\n"
                summary += f"  Std: {df[metric].std():.2f}\n\n"
        
        return {
            "summary": summary,
            "chart_paths": chart_paths
        }
        
    except Exception as e:
        print(f"Error in get_nvidia_historical: {str(e)}")
        raise

def generate_snowflake_insights(query: str, metadata_filters: dict) -> dict:
    """
    Generate insights from Snowflake data for the research report.
    """
    try:
        year = metadata_filters.get("year", "2024")
        quarter = metadata_filters.get("quarter", "Q1")
        
        # Get historical data
        result = get_nvidia_historical(
            year=year,
            quarter=quarter
        )
        
        # Format the response
        return {
            "summary": result["summary"],
            "visualizations": [
                {
                    "title": f"NVIDIA Stock {metric} Trend",
                    "url": path,
                    "type": "chart",
                    "columns": [metric]
                }
                for metric, path in result["chart_paths"].items()
                if path is not None
            ]
        }
        
    except Exception as e:
        print(f"Error generating Snowflake insights: {str(e)}")
        raise

























