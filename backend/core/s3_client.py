import boto3
import requests
import logging
from typing import List, Dict, Optional, Union
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class S3FileManager:
    def __init__(self, bucket_name: str, base_path: str = ""):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
        self.base_path = base_path
        logging.info(f"S3FileManager initialized for bucket: {bucket_name}, base_path: {base_path}")

    def get_full_path(self, filename: str) -> str:
        """Get full S3 path by combining base_path and filename"""
        if self.base_path:
            return f"{self.base_path}/{filename}"  # Ensure proper path joining
        return filename

    def list_files(self) -> List[str]:
        """List all files in the bucket under base_path"""
        try:
            prefix = self.base_path if self.base_path else None
            kwargs = {'Bucket': self.bucket_name}
            if prefix:
                kwargs['Prefix'] = prefix
            
            response = self.s3.list_objects_v2(**kwargs)
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except Exception as e:
            logger.error(f"Failed to list files: {str(e)}")
            return []

    def load_s3_pdf(self, filename: str) -> Optional[bytes]:
        """Load PDF content from S3"""
        try:
            full_path = self.get_full_path(filename)
            logger.info(f"Loading PDF from: {full_path}")
            response = self.s3.get_object(Bucket=self.bucket_name, Key=full_path)
            return response['Body'].read()
        except Exception as e:
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
            self.s3.put_object(
                Bucket=bucket_name,
                Key=full_path,
                Body=content
            )
            logger.info(f"Successfully uploaded {full_path} to {bucket_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {key}: {str(e)}")
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

def upload_to_s3(reports: List[Dict], bucket_name: str) -> Dict[str, int]:
    """Helper function to upload reports to S3"""
    s3_manager = S3FileManager(bucket_name)
    return s3_manager.upload_reports(reports)
