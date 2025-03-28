import boto3
import requests
import logging
from typing import List, Dict, Optional, Union
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Load environment variables from project root
env_path = os.path.join(parent_dir, '.env')
load_dotenv(env_path)

logger = logging.getLogger(__name__)

class S3FileManager:
    def __init__(self, bucket_name: str, base_path: str = ""):
        # Get AWS credentials from environment variables
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        
        # Debug print environment variables (without showing actual values)
        logger.info(f"AWS_ACCESS_KEY_ID exists: {bool(aws_access_key_id)}")
        logger.info(f"AWS_SECRET_ACCESS_KEY exists: {bool(aws_secret_access_key)}")
        logger.info(f"AWS_REGION: {aws_region}")
        
        if not all([aws_access_key_id, aws_secret_access_key]):
            raise ValueError("AWS credentials not found in environment variables")
            
        # Initialize S3 client with explicit credentials
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        
        self.bucket_name = bucket_name
        self.base_path = base_path.rstrip('/')  # Remove trailing slash if present
        self.base_reports_path = "nvidia-reports/processed/mistral"
        logger.info(f"S3FileManager initialized for bucket: {bucket_name}, base_path: {base_path}")
        logger.info(f"Using AWS region: {aws_region}")

    def get_full_path(self, filename: str) -> str:
        """Get full S3 path by combining base_path and filename"""
        try:
            # Remove any leading/trailing slashes and clean the path
            filename = filename.lstrip('/').rstrip('/')
            base_path = self.base_path.rstrip('/')
            
            # If the filename already contains the base path, return it as is
            if filename.startswith(base_path):
                return filename
                
            # Otherwise, combine base path and filename
            full_path = f"{base_path}/{filename}" if base_path else filename
            logger.debug(f"Generated full path: {full_path}")
            return full_path
        except Exception as e:
            logger.error(f"Error generating full path for {filename}: {str(e)}")
            raise

    def list_files(self, base_path: str = "", file_types: Optional[List[str]] = None) -> List[str]:
        """List files in the S3 bucket with optional filtering."""
        try:
            if file_types is None:
                file_types = ['.pdf', '.md']
            
            # Ensure base_path ends with a slash if not empty
            prefix = f"{base_path}/" if base_path else ""
            
            logger.info(f"Listing files in bucket {self.bucket_name} with prefix: {prefix}")
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                logger.warning(f"No files found in bucket {self.bucket_name} with prefix {prefix}")
                return []
            
            # Filter files by type and exclude processed directory
            files = []
            for obj in response['Contents']:
                key = obj['Key']
                if any(key.endswith(ft) for ft in file_types) and 'processed' not in key:
                    files.append(key)
            
            logger.info(f"Found {len(files)} files matching types {file_types}")
            return files
            
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            logger.error("Full error details:")
            import traceback
            traceback.print_exc()
            return []

    def load_s3_pdf(self, filename: str) -> Optional[bytes]:
        """Load PDF content from S3"""
        try:
            full_path = self.get_full_path(filename)
            print(f"Attempting to load PDF from: {full_path}")  # Debug print
            response = self.s3.get_object(Bucket=self.bucket_name, Key=full_path)
            content = response['Body'].read()
            print(f"Successfully loaded PDF, size: {len(content)} bytes")  # Debug print
            return content
        except Exception as e:
            print(f"Error loading PDF {filename}: {str(e)}")  # Debug print
            logger.error(f"Failed to load PDF {filename}: {str(e)}")
            return None

    def load_s3_file_content(self, filename: str) -> Optional[str]:
        """Load text file content from S3"""
        try:
            full_path = self.get_full_path(filename)
            logger.info(f"Loading file from: {full_path}")
            response = self.s3.get_object(Bucket=self.bucket_name, Key=full_path)
            return response['Body'].read().decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to load file {filename}: {str(e)}")
            return None

    def upload_file(self, bucket_name: str, key: str, content: Union[bytes, str]) -> bool:
        """Upload content to S3"""
        try:
            if isinstance(content, str):
                content = content.encode('utf-8')
            
            full_path = self.get_full_path(key)
            logger.info(f"Uploading to: {full_path}")
            
            # Validate content before upload
            if not content:
                logger.error("Cannot upload empty content")
                return False
                
            self.s3.put_object(
                Bucket=bucket_name,
                Key=full_path,
                Body=content
            )
            logger.info(f"Successfully uploaded {full_path} to {bucket_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {key}: {str(e)}")
            logger.error("Full error details:")
            import traceback
            traceback.print_exc()
            return False

    def upload_single_report(self, report: Dict) -> bool:
        """Upload a single report from a URL to S3"""
        try:
            response = requests.get(report['url'], timeout=30)  # Increased timeout
            response.raise_for_status()
        
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=f"nvidia_reports/{report['filename']}",  # Changed path
                Body=response.content
            )
            logger.info(f"Successfully uploaded {report['filename']}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload report {report['filename']}: {str(e)}")
            return False

    def upload_reports(self, reports: List[Dict]) -> Dict[str, int]:
        """Upload multiple reports and return success/failure counts"""
        results = {"success": 0, "failed": 0}
        for report in reports:
            if self.upload_single_report(report):
                results["success"] += 1
            else:
                results["failed"] += 1
        return results

    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get content of a file from S3."""
        try:
            logger.info(f"Fetching content of file: {file_path}")
            
            # For PDF files, we need to convert them to text
            if file_path.endswith('.pdf'):
                logger.info("Processing PDF file")
                response = self.s3.get_object(
                    Bucket=self.bucket_name,
                    Key=file_path
                )
                import io
                import PyPDF2
                
                # Read PDF content
                pdf_content = io.BytesIO(response['Body'].read())
                pdf_reader = PyPDF2.PdfReader(pdf_content)
                
                # Extract text from all pages
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                    
                logger.info(f"Successfully extracted text from PDF: {file_path}")
                return text
                
            else:  # For markdown or text files
                logger.info("Processing text file")
                response = self.s3.get_object(
                    Bucket=self.bucket_name,
                    Key=file_path
                )
                content = response['Body'].read().decode('utf-8')
                logger.info(f"Successfully loaded text file: {file_path}")
                return content
                
        except Exception as e:
            logger.error(f"Error fetching file content: {str(e)}")
            logger.error("Full error details:")
            import traceback
            traceback.print_exc()
            return None

    def get_report_folder_path(self, year: str, quarter: str) -> str:
        """Get the folder path for a specific year and quarter."""
        return f"{self.base_reports_path}/nvidia_raw_pdf_{year}_{quarter}"
        
    def get_pdf_path(self, year: str, quarter: str) -> str:
        """Get the path for the PDF file."""
        # Remove any existing Q prefix from quarter if present
        quarter = quarter.replace('Q', '') if quarter.startswith('Q') else quarter
        return f"nvidia-reports/nvidia_raw_pdf_{year}_Q{quarter}.pdf"
        
    def get_markdown_path(self, year: str, quarter: str) -> str:
        """Get the path for the markdown file."""
        # Ensure quarter is in the format Q1, Q2, etc.
        if not quarter.startswith('Q'):
            quarter = f"Q{quarter}"
        base_folder = "nvidia-reports/processed/mistral"
        folder_name = f"nvidia_raw_pdf_{year}_{quarter}"
        return f"{base_folder}/{folder_name}/nvidia_earnings_call_{year}_{quarter}.md"
        
    def get_images_folder_path(self, year: str, quarter: str) -> str:
        """Get the path for the images folder."""
        folder_path = self.get_report_folder_path(year, quarter)
        return f"{folder_path}/images"
        
    def ensure_folder_exists(self, folder_path: str) -> bool:
        """Ensure a folder exists in S3 by creating an empty marker object."""
        try:
            # S3 doesn't have real folders, but we can create a marker object
            marker_path = f"{folder_path.rstrip('/')}/.folder"
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=marker_path,
                Body=b''
            )
            logger.info(f"Created/verified folder: {folder_path}")
            return True
        except Exception as e:
            logger.error(f"Error ensuring folder exists {folder_path}: {str(e)}")
            return False
            
    def save_markdown_content(self, year: str, quarter: str, content: str) -> bool:
        """Save markdown content to the appropriate location."""
        try:
            # Create the base folder path
            base_folder = "nvidia-reports/processed/mistral"
            folder_name = f"nvidia_raw_pdf_{year}_{quarter}"
            full_folder_path = f"{base_folder}/{folder_name}"
            
            # Ensure the base folder exists
            self.ensure_folder_exists(base_folder)
            
            # Ensure the specific report folder exists
            self.ensure_folder_exists(full_folder_path)
            
            # Create images folder
            images_folder = f"{full_folder_path}/images"
            self.ensure_folder_exists(images_folder)
            
            # Save the markdown file
            markdown_path = f"{full_folder_path}/nvidia_earnings_call_{year}_{quarter}.md"
            return self.upload_file(
                bucket_name=self.bucket_name,
                key=markdown_path,
                content=content
            )
        except Exception as e:
            logger.error(f"Error saving markdown content for {year} {quarter}: {str(e)}")
            return False
            
    def save_chunk_image(self, year: str, quarter: str, image_name: str, image_data: bytes) -> bool:
        """Save a chunked image to the images folder."""
        try:
            # Ensure the images folder exists
            images_folder = self.get_images_folder_path(year, quarter)
            self.ensure_folder_exists(images_folder)
            
            # Save the image
            image_path = f"{images_folder}/{image_name}"
            return self.upload_file(
                bucket_name=self.bucket_name,
                key=image_path,
                content=image_data
            )
        except Exception as e:
            logger.error(f"Error saving chunk image {image_name}: {str(e)}")
            return False
            
    def check_markdown_exists(self, year: str, quarter: str) -> bool:
        """Check if the markdown file exists for the given year and quarter."""
        try:
            markdown_path = self.get_markdown_path(year, quarter)
            self.s3.head_object(
                Bucket=self.bucket_name,
                Key=markdown_path
            )
            return True
        except Exception:
            return False

def upload_to_s3(reports: List[Dict], bucket_name: str) -> Dict[str, int]:
    """Helper function to upload reports to S3"""
    s3_manager = S3FileManager(bucket_name)
    return s3_manager.upload_reports(reports)
