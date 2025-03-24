import os
import sys
import time
import boto3
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv

load_dotenv()

class S3Uploader:
    """Handles all S3 upload operations"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
    
    def upload_file(self, bucket_name, key, content):
        """Upload content to S3 bucket"""
        self.s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=content,
            ContentType='application/pdf'
        )

def setup_driver():
    """Configure and return Chrome WebDriver"""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    return webdriver.Chrome(options=options)

def scrape_nvidia_reports():
    """Scrape all NVIDIA quarterly reports from their investor site"""
    driver = setup_driver()
    try:
        driver.get("https://investor.nvidia.com/financial-info/quarterly-results/default.aspx")
        
        # Wait for page to load completely
        WebDriverWait(driver, 50).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#_ctrl0_ctl75_selectEvergreenFinancialAccordionYear"))
        )
        
        reports = []
        years = ["2025", "2024", "2023", "2022", "2021", "2020"]
        
        # Get year dropdown
        year_dropdown = Select(driver.find_element(By.ID, "_ctrl0_ctl75_selectEvergreenFinancialAccordionYear"))
        
        for year in years:
            try:
                year_dropdown.select_by_visible_text(year)
                print(f"\nProcessing year: {year}")
                time.sleep(2)
                
                # Find all accordion items
                items = driver.find_elements(By.CSS_SELECTOR, ".evergreen-accordion.evergreen-financial-accordion-item")
                
                for item in items:
                    try:
                        # Get quarter title
                        title_element = item.find_element(By.CSS_SELECTOR, ".evergreen-accordion-title")
                        quarter_title = title_element.text.strip()
                        print(f"Found section: {quarter_title}")
                        
                        # Determine quarter from title
                        if "First" in quarter_title:
                            quarter = "Q1"
                        elif "Second" in quarter_title:
                            quarter = "Q2"
                        elif "Third" in quarter_title:
                            quarter = "Q3"
                        elif "Fourth" in quarter_title or "Annual" in quarter_title:
                            quarter = "Q4"
                        else:
                            continue
                        
                        # Click to expand if not expanded
                        try:
                            toggle = item.find_element(By.CSS_SELECTOR, "button.evergreen-financial-accordion-toggle")
                            if not toggle.get_attribute("aria-expanded") == "true":
                                toggle.click()
                                time.sleep(1)
                        except:
                            pass
                        
                        # Find PDF links
                        content = item.find_element(By.CSS_SELECTOR, ".evergreen-accordion-content")
                        links = content.find_elements(By.CSS_SELECTOR, "a[href*='.pdf']")
                        
                        for link in links:
                            try:
                                url = link.get_attribute("href")
                                if url and url.endswith(".pdf"):
                                    report_type = "10-K" if "Form 10-K" in link.text else "10-Q"
                                    
                                    reports.append({
                                        "year": year,
                                        "quarter": quarter,
                                        "type": report_type,
                                        "url": url,
                                        "filename": f"nvidia_raw_pdf_{year}_{quarter}.pdf"
                                    })
                                    print(f"Found {year} {quarter} ({report_type}): {url}")
                            except Exception as e:
                                print(f"Error processing link: {str(e)}")
                                continue
                    
                    except Exception as e:
                        print(f"Error processing quarter section: {str(e)}")
                        continue
            
            except Exception as e:
                print(f"Error processing year {year}: {str(e)}")
                continue
        
        if not reports:
            print("No reports found for any year!")
        else:
            print(f"\nTotal reports found: {len(reports)}")
            
        return reports
    
    except Exception as e:
        print(f"Error during scraping: {str(e)}")
        return []
    finally:
        driver.quit()

def download_and_upload_report(report, bucket_name):
    """Download a single report and upload to S3"""
    try:
        # Download PDF
        response = requests.get(report['url'], timeout=10)
        response.raise_for_status()
        
        # Upload to S3
        uploader = S3Uploader()
        uploader.upload_file(
            bucket_name,
            f"nvidia-reports/{report['filename']}",
            response.content
        )
        print(f"Uploaded {report['filename']} to S3")
        return True
    except Exception as e:
        print(f"Failed to process {report['filename']}: {str(e)}")
        return False

def main():
    """Main execution function"""
    try:
        # Scrape all reports
        print("Starting NVIDIA report scraping...")
        all_reports = scrape_nvidia_reports()
        
        if not all_reports:
            print("No reports found!")
            sys.exit(1)
        
        print(f"\nFound {len(all_reports)} reports:")
        for report in all_reports:
            print(f"{report['year']} {report['quarter']}: {report['url']}")
        
        # Get S3 bucket name
        bucket_name = os.getenv("AWS_BUCKET_NAME")
        if not bucket_name:
            print("AWS_BUCKET_NAME environment variable not set")
            sys.exit(1)
        
        # Upload reports in parallel
        print("\nUploading reports to S3...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(
                lambda r: download_and_upload_report(r, bucket_name),
                all_reports
            ))
        
        success_count = sum(results)
        print(f"\nUpload complete. Successfully uploaded {success_count}/{len(all_reports)} reports")
        sys.exit(0 if success_count == len(all_reports) else 1)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        # For single report mode (maintains compatibility with your original script)
        if len(sys.argv) != 4:
            print("Usage: python script.py --single <year> <quarter>")
            sys.exit(1)
        year = sys.argv[2]
        quarter = sys.argv[3]
        
        # Filter reports for the requested year/quarter
        all_reports = scrape_nvidia_reports()
        filtered_reports = [
            r for r in all_reports 
            if r['year'] == year and r['quarter'] == quarter
        ]
        
        if not filtered_reports:
            print(f"No report found for {year} {quarter}")
            sys.exit(1)
            
        bucket_name = os.getenv("AWS_BUCKET_NAME")
        if download_and_upload_report(filtered_reports[0], bucket_name):
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        # Default mode - scrape and upload all reports
        main()
