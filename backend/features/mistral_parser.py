import os
import io
import base64
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from mistralai import Mistral
from mistralai import DocumentURLChunk
from mistralai.models import OCRResponse
from backend.core.s3_client import S3FileManager

# Load environment variables
load_dotenv()

# Environment Variables
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
MISTRALAI_API_KEY = os.getenv("MISTRAL_API_KEY")


def pdf_mistralocr_converter(pdf_content: bytes, base_path, s3_obj):
    """
    Convert PDF to markdown using Mistral OCR.

    Args:
        pdf_content (bytes): The PDF file content as bytes.
        base_path (str): The base path in S3 for storing the output.
        s3_obj (S3FileManager): The S3 file manager object.

    Returns:
        tuple: (markdown_file_name, markdown_content)
    """
    import time
    import requests
    from requests.exceptions import SSLError, ConnectionError, Timeout
    
    # Convert bytes to BytesIO
    pdf_stream = io.BytesIO(pdf_content)
    
    # Load environment variables and set API key
    mistral_api_key = os.getenv('MISTRAL_API_KEY')
    if not mistral_api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment variables")
    os.environ['MISTRAL_API_KEY'] = mistral_api_key
    print("Mistral API key loaded successfully")

    # Initialize Mistral client
    client = Mistral(api_key=MISTRALAI_API_KEY)

    # Read the PDF file
    pdf_stream.seek(0)
    pdf_bytes = pdf_stream.read()

    # Implement retry logic for potential SSL or connectivity issues
    max_retries = 3
    retry_count = 0
    backoff_factor = 2  # Exponential backoff
    
    # Upload PDF file to Mistral's OCR service
    while retry_count < max_retries:
        try:
            uploaded_file = client.files.upload(
                file={
                    "file_name": "temp.pdf",
                    "content": pdf_bytes,
                },
                purpose="ocr",
            )
            print(f"DEBUG: Uploaded file ID: {uploaded_file.id}")
            break  # Break if successful
        except (SSLError, ConnectionError, Timeout) as e:
            retry_count += 1
            wait_time = backoff_factor ** retry_count
            print(f"Connection error (attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed to upload file after {max_retries} attempts: {str(e)}")
        except Exception as e:
            print(f"Error uploading file to Mistral OCR: {str(e)}")
            raise

    # Reset retry counter for getting the signed URL
    retry_count = 0
    
    # Get signed URL for the uploaded file
    while retry_count < max_retries:
        try:
            signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
            print(f"DEBUG: Signed URL: {signed_url.url}")
            break  # Break if successful
        except (SSLError, ConnectionError, Timeout) as e:
            retry_count += 1
            wait_time = backoff_factor ** retry_count
            print(f"Connection error getting signed URL (attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed to get signed URL after {max_retries} attempts: {str(e)}")
        except Exception as e:
            print(f"Error getting signed URL: {str(e)}")
            raise

    # Reset retry counter for OCR processing
    retry_count = 0
    
    # Process PDF with OCR, including embedded images
    while retry_count < max_retries:
        try:
            pdf_response = client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model="mistral-ocr-latest",
                include_image_base64=True
            )
            print(f"DEBUG: OCR Response received successfully")
            break  # Break if successful
        except (SSLError, ConnectionError, Timeout) as e:
            retry_count += 1
            wait_time = backoff_factor ** retry_count
            print(f"Connection error during OCR processing (attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed OCR processing after {max_retries} attempts: {str(e)}")
        except Exception as e:
            print(f"Error processing PDF with Mistral OCR: {str(e)}")
            raise

    # Combine markdown content and replace image placeholders
    final_md_content = get_combined_markdown(pdf_response, s3_obj, base_path)

    # Define the markdown file name
    md_file_name = f"{base_path}/extracted_data.md"

    # Upload the markdown file to S3 with retry logic
    retry_count = 0
    while retry_count < max_retries:
        try:
            s3_obj.upload_file(s3_obj.bucket_name, md_file_name, final_md_content.encode('utf-8'))
            print(f"DEBUG: Uploaded markdown file to S3: {md_file_name}")
            break  # Break if successful
        except Exception as e:
            retry_count += 1
            wait_time = backoff_factor ** retry_count
            print(f"Error uploading markdown to S3 (attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed to upload markdown to S3 after {max_retries} attempts: {str(e)}")

    return md_file_name, final_md_content


def replace_images_in_markdown(markdown_str: str, images_dict: dict, s3_obj, base_path) -> str:
    """
    Replace image placeholders in markdown with actual image links.

    Args:
        markdown_str (str): The markdown content with image placeholders.
        images_dict (dict): A dictionary of image IDs and base64-encoded images.
        s3_obj (S3FileManager): The S3 file manager object.
        base_path (str): The base path in S3 for storing images.

    Returns:
        str: The updated markdown content with image links.
    """
    for img_name, base64_str in images_dict.items():
        print(f"Replacing image {img_name}")

        # Decode the base64 image
        try:
            base64_str = base64_str.split(';')[1].split(',')[1]
            image_data = base64.b64decode(base64_str)
        except Exception as e:
            print(f"Error decoding base64 image: {str(e)}")
            continue

        # Define the image filename in S3
        element_image_filename = f"{base_path}/images/{img_name.split('.')[0]}.png"
        print(f"DEBUG: Image filename: {element_image_filename}")

        # Convert the image to PNG format
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            output_buffer = io.BytesIO()
            image.save(output_buffer, format="PNG")
            output_buffer.seek(0)
        except Exception as e:
            print(f"Error converting image to PNG: {str(e)}")
            continue

        # Upload the image to S3
        try:
            s3_obj.upload_file(s3_obj.bucket_name, element_image_filename, output_buffer.read())
            element_image_link = f"https://{s3_obj.bucket_name}.s3.amazonaws.com/{element_image_filename}"
            markdown_str = markdown_str.replace(
                f"![{img_name}]({img_name})", f"![{img_name}]({element_image_link})"
            )
        except Exception as e:
            print(f"Error uploading image to S3: {str(e)}")
            continue

    return markdown_str


def get_combined_markdown(ocr_response: OCRResponse, s3_obj, base_path) -> str:
    """
    Combine markdown content from all pages of the PDF.

    Args:
        ocr_response (OCRResponse): The OCR response from Mistral.
        s3_obj (S3FileManager): The S3 file manager object.
        base_path (str): The base path in S3 for storing images.

    Returns:
        str: The combined markdown content.
    """
    markdowns = []
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        markdowns.append(replace_images_in_markdown(page.markdown, image_data, s3_obj, base_path))
    return "\n\n".join(markdowns)


def main():
    """
    Main function to process PDF files using Mistral OCR.
    """
    # Define the base path in S3
    base_path = "nvidia/raw_pdf_files"

    # Print diagnostic info
    print(f"Using S3 bucket: {AWS_BUCKET_NAME}")
    print(f"Using base path: {base_path}")

    # Initialize S3 file manager
    s3_obj = S3FileManager(AWS_BUCKET_NAME, base_path)

    try:
        # List all PDF files in the S3 bucket
        print("Listing files in S3 bucket...")
        files = list({file for file in s3_obj.list_files() if file.endswith('.pdf')})
        print(f"Found {len(files)} PDF files: {files}")

        # Process each PDF file
        for file in files:
            print(f"Processing file: {file}")
            try:
                # Load the PDF file from S3
                pdf_file = s3_obj.load_s3_pdf(file)
                pdf_bytes = io.BytesIO(pdf_file)

                # Define the output path in S3
                output_path = f"{s3_obj.base_path}/mistral/{file.split('/')[-1].split('.')[0]}"
                print(f"Output path: {output_path}")

                # Convert the PDF to markdown
                file_name, markdown_content = pdf_mistralocr_converter(pdf_bytes, output_path, s3_obj)
                print(f"Successfully processed file: {file_name}")
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
    except Exception as e:
        print(f"Error listing files in S3: {str(e)}")
        print(f"Make sure S3 bucket {AWS_BUCKET_NAME} exists and credentials are correct.")


if __name__ == "__main__":
    main()