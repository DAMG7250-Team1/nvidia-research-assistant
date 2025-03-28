import os
import sys
from pathlib import Path
import boto3
import logging
from dotenv import load_dotenv
import re
from collections import defaultdict

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.s3_client import S3FileManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_s3_files():
    """Check the contents of the S3 bucket and display file information."""
    try:
        # Load environment variables from project root
        env_path = os.path.join(parent_dir, '.env')
        print(f"\n🔍 Loading environment variables from: {env_path}")
        load_dotenv(env_path)
        
        # Get bucket name from environment
        bucket_name = os.getenv("AWS_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("AWS_BUCKET_NAME not found in environment variables")
            
        print(f"\n🔍 Checking AWS credentials...")
        
        # Initialize S3 client with explicit credentials
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        
        # Debug print environment variables (without showing actual values)
        print(f"AWS_ACCESS_KEY_ID exists: {bool(aws_access_key_id)}")
        print(f"AWS_SECRET_ACCESS_KEY exists: {bool(aws_secret_access_key)}")
        print(f"AWS_REGION: {aws_region}")
        print(f"AWS_BUCKET_NAME: {bucket_name}")
        
        if not all([aws_access_key_id, aws_secret_access_key]):
            raise ValueError("AWS credentials not found in environment variables")
            
        sts = boto3.client(
            'sts',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        
        # Verify AWS credentials
        identity = sts.get_caller_identity()
        print(f"✅ Successfully authenticated as: {identity['Arn']}")
        
        print(f"\n📦 Checking bucket: {bucket_name}")
        
        # Initialize S3FileManager
        s3_manager = S3FileManager(bucket_name, "nvidia-reports/")
        
        # List files
        print("\n📄 Listing files in bucket...")
        files = s3_manager.list_files()
        
        if not files:
            print("❌ No files found in bucket")
            return
            
        print(f"\n✅ Found {len(files)} files")
        
        # Count files by year and quarter
        year_counts = defaultdict(lambda: defaultdict(int))
        file_types = defaultdict(int)
        processed_files = defaultdict(int)
        
        print("\n📊 Analyzing files...")
        for file in files:
            # Extract year and quarter from PDF files
            if file.endswith('.pdf'):
                match = re.search(r'nvidia_raw_pdf_(\d{4})_Q(\d)\.pdf$', file)
                if match:
                    year = match.group(1)
                    quarter = f"Q{match.group(2)}"
                    year_counts[year][quarter] += 1
                    processed_files['pdf'] += 1
            
            # Count file types
            ext = os.path.splitext(file)[1].lower()
            file_types[ext] += 1
            
        # Print summary
        print("\n📈 Files by Year and Quarter:")
        for year in sorted(year_counts.keys()):
            print(f"\n{year}:")
            for quarter in sorted(year_counts[year].keys()):
                print(f"  {quarter}: {year_counts[year][quarter]} files")
                
        print("\n📁 Files by Type:")
        for ext, count in file_types.items():
            print(f"  {ext}: {count} files")
            
        print("\n📊 Processed Files Summary:")
        print(f"  PDF Reports: {processed_files['pdf']} files")
        print(f"  Images: {file_types['.png']} files")
        print(f"  Markdown: {file_types['.md']} files")
            
    except ValueError as ve:
        print(f"\n❌ Configuration Error: {str(ve)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error checking S3 files: {str(e)}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    check_s3_files() 