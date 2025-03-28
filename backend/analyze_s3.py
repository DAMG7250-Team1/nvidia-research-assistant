import os
import boto3
import logging
from dotenv import load_dotenv
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_s3_structure():
    """Analyze the structure of files in the S3 bucket."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get AWS credentials
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        
        logger.info("\n🔍 Analyzing S3 Bucket Structure")
        logger.info("=" * 50)
        
        # Initialize S3 client
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        
        # List objects in bucket
        bucket_name = "nvidia-research"
        response = s3.list_objects_v2(Bucket=bucket_name)
        
        if 'Contents' not in response:
            logger.error("No files found in bucket")
            return
            
        # Analyze file structure
        files_by_type = defaultdict(list)
        files_by_year = defaultdict(list)
        
        for obj in response['Contents']:
            key = obj['Key']
            
            # Categorize by file type
            if key.endswith('.pdf'):
                files_by_type['pdf'].append(key)
            elif key.endswith('.md'):
                files_by_type['markdown'].append(key)
            elif key.endswith('.png'):
                files_by_type['image'].append(key)
            else:
                files_by_type['other'].append(key)
            
            # Try to extract year from path
            import re
            year_match = re.search(r'(\d{4})', key)
            if year_match:
                year = year_match.group(1)
                files_by_year[year].append(key)
        
        # Print summary
        logger.info("\n📊 File Type Summary:")
        for file_type, files in files_by_type.items():
            logger.info(f"- {file_type.upper()}: {len(files)} files")
            
        logger.info("\n📅 Files by Year:")
        for year, files in sorted(files_by_year.items()):
            logger.info(f"- {year}: {len(files)} files")
            
        # Print sample files
        logger.info("\n📁 Sample Files:")
        for file_type, files in files_by_type.items():
            if files:
                logger.info(f"\n{file_type.upper()} files:")
                for file in files[:3]:  # Show first 3 files of each type
                    logger.info(f"- {file}")
                    
    except Exception as e:
        logger.error(f"\n❌ Error analyzing S3 structure: {str(e)}")
        logger.error("Full error details:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_s3_structure() 