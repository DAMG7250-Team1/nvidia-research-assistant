import boto3
import requests
import logging
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class S3Uploader:
    def __init__(self, bucket_name: str):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
        logging.info(f"S3Uploader initialized for bucket: {bucket_name}")

    def upload_single_report(self, report: Dict) -> bool:
        try:
            response = requests.get(report['url'], timeout=30)  # Increased timeout
            response.raise_for_status()
        
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=f"nvidia_reports/{report['filename']}",  # Changed path
                Body=response.content,
                ContentType='application/pdf'
            )
            logger.info(f"Uploaded {report['filename']}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {report['filename']}: {str(e)}")
            return False

    def upload_reports(self, reports: List[Dict]) -> Dict[str, int]:
        """Upload multiple reports"""
        results = {'total': len(reports), 'success': 0, 'failed': 0}
        
        for report in reports:
            if self.upload_single_report(report):
                results['success'] += 1
            else:
                results['failed'] += 1
                
        logging.info(f"Upload results: {results}")
        return results

def upload_to_s3(reports: List[Dict], bucket_name: str) -> Dict[str, int]:
    """Main upload interface"""
    return S3Uploader(bucket_name).upload_reports(reports)
    return S3Uploader(bucket_name).upload_reports(reports)