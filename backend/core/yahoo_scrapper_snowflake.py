# from yahooquery import Ticker
# import pandas as pd

# # Get NVIDIA data
# nvda = Ticker("NVDA")

# # Fetch quarterly key statistics
# stats = nvda.valuation_measures

# # Convert to DataFrame
# df = pd.DataFrame(stats).T

# # Reset index to clean structure
# df = df.reset_index()

# # Drop 'periodType' row if it exists
# if "periodType" in df["index"].values:
#     df = df[df["index"] != "periodType"]
#     print("Dropped 'periodType' row.")

# # Rename index column for clarity
# df = df.rename(columns={"index": "Metric"})  

# # Pivot: Convert Metrics into Columns (Dates in Rows)
# df_pivoted = df.set_index("Metric").T  # Transpose the table

# # Reset index to get the correct format for Snowflake
# df_pivoted.reset_index(inplace=True)

# # Rename the first column to 'Date'
# df_pivoted = df_pivoted.rename(columns={"index": "Date"})

# # Drop the 'symbol' column if it exists
# if "symbol" in df_pivoted.columns:
#     df_pivoted = df_pivoted.drop(columns=["symbol"])
#     print("Removed 'symbol' column.")

# # Save as CSV (for Snowflake ingestion)
# csv_filename = "nvidia_pivoted_cleaned_data.csv"
# df_pivoted.to_csv(csv_filename, index=False)  # No extra index column

# print(f"✅ Data saved successfully to: {csv_filename} (without 'symbol' column).")





from yahooquery import Ticker
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
import yfinance as yf
import snowflake.connector
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_snowflake_connection():
    """Establish connection to Snowflake"""
    try:
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),  # Load password from environment variable
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse='COMPUTE_WH',
            database='FINANCE_MARKET_DATA',
            schema='YAHOO_MARKET_DATA',
            client_session_keep_alive=True,
            login_timeout=60,
            network_timeout=60,
            application='NVIDIA_RESEARCH_ASSISTANT'
        )
        logger.info("Successfully connected to Snowflake")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to Snowflake: {str(e)}")
        raise

def fetch_yahoo_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance"""
    try:
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        
        # Download data from Yahoo Finance
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            logger.warning(f"No data available for {ticker} in the specified date range")
            return pd.DataFrame()
        
        # Reset index and rename columns to match Snowflake schema
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'ASOFDATE',
            'Open': 'OPEN',
            'High': 'HIGH',
            'Low': 'LOW',
            'Close': 'CLOSE',
            'Volume': 'VOLUME',
            'Adj Close': 'ADJUSTEDCLOSE'
        })
        
        # Ensure all required columns exist
        required_columns = ['ASOFDATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'ADJUSTEDCLOSE']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # Convert date to datetime and remove timezone information
        df['ASOFDATE'] = pd.to_datetime(df['ASOFDATE']).dt.tz_localize(None)
        
        # Convert all numeric columns to float64
        numeric_columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'ADJUSTEDCLOSE']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data from Yahoo Finance: {str(e)}")
        raise

def check_data_exists(year: str, quarter: str) -> bool:
    """Check if data exists in Snowflake for a specific year and quarter"""
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # First check if table exists
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = 'YAHOO_MARKET_DATA' 
            AND TABLE_NAME = 'NVIDIA_HISTORICAL'
        """)
        table_exists = cursor.fetchone()[0] > 0
        
        if not table_exists:
            logger.info("Table NVIDIA_HISTORICAL does not exist")
            return False
        
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
        
        # Check if data exists
        query = f"""
        SELECT COUNT(*) 
        FROM NVIDIA_HISTORICAL 
        WHERE ASOFDATE BETWEEN '{start_date}' AND '{end_date}'
        """
        cursor.execute(query)
        count = cursor.fetchone()[0]
        
        logger.info(f"Found {count} records for {year} {quarter}")
        return count > 0
        
    except Exception as e:
        logger.error(f"Error checking data existence: {str(e)}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def load_data_to_snowflake(df: pd.DataFrame, table_name: str = 'NVIDIA_HISTORICAL') -> bool:
    """Load data into Snowflake table"""
    try:
        if df.empty:
            logger.warning("No data to load into Snowflake")
            return False
            
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            ASOFDATE TIMESTAMP_NTZ,
            OPEN FLOAT,
            HIGH FLOAT,
            LOW FLOAT,
            CLOSE FLOAT,
            VOLUME FLOAT,
            ADJUSTEDCLOSE FLOAT
        )
        """
        cursor.execute(create_table_query)
        
        # Prepare the data for bulk insert
        values = []
        for _, row in df.iterrows():
            # Convert timestamp to string in Snowflake format
            asofdate = row['ASOFDATE'].strftime('%Y-%m-%d %H:%M:%S')
            # Convert values to appropriate types and handle NULLs
            values.append((
                asofdate,
                float(row['OPEN']) if pd.notna(row['OPEN']) else None,
                float(row['HIGH']) if pd.notna(row['HIGH']) else None,
                float(row['LOW']) if pd.notna(row['LOW']) else None,
                float(row['CLOSE']) if pd.notna(row['CLOSE']) else None,
                float(row['VOLUME']) if pd.notna(row['VOLUME']) else None,
                float(row['ADJUSTEDCLOSE']) if pd.notna(row['ADJUSTEDCLOSE']) else None
            ))
        
        # Insert data in batches
        batch_size = 1000
        for i in range(0, len(values), batch_size):
            batch = values[i:i + batch_size]
            insert_query = f"""
            INSERT INTO {table_name} 
            (ASOFDATE, OPEN, HIGH, LOW, CLOSE, VOLUME, ADJUSTEDCLOSE)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.executemany(insert_query, batch)
        
        # Commit the transaction
        conn.commit()
        logger.info(f"Successfully loaded {len(values)} records into {table_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading data into Snowflake: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def fetch_and_load_missing_data(year: str, quarter: str) -> bool:
    """Fetch and load missing data from Yahoo Finance"""
    try:
        # Check if data already exists
        if check_data_exists(year, quarter):
            logger.info(f"Data already exists for {year} {quarter}")
            return True
        
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
        
        # Fetch data from Yahoo Finance
        df = fetch_yahoo_data('NVDA', start_date, end_date)
        
        if df.empty:
            logger.warning(f"No data available from Yahoo Finance for {year} {quarter}")
            return False
        
        # Load data into Snowflake
        return load_data_to_snowflake(df)
        
    except Exception as e:
        logger.error(f"Error in fetch_and_load_missing_data: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    fetch_and_load_missing_data("2023", "Q1")
