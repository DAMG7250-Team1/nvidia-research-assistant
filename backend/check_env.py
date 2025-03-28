import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Check all required environment variables."""
    # Load environment variables
    load_dotenv()
    
    # List of required variables
    required_vars = {
        "AWS_ACCESS_KEY_ID": "AWS Access Key ID",
        "AWS_SECRET_ACCESS_KEY": "AWS Secret Access Key",
        "AWS_REGION": "AWS Region",
        "PINECONE_API_KEY": "Pinecone API Key",
        "PINECONE_ENVIRONMENT": "Pinecone Environment",
        "PINECONE_INDEX_NAME": "Pinecone Index Name",
        "OPENAI_API_KEY": "OpenAI API Key"
    }
    
    logger.info("\n🔍 Checking Environment Variables")
    logger.info("=" * 50)
    
    missing_vars = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        exists = bool(value)
        logger.info(f"{description}: {'✅' if exists else '❌'}")
        if not exists:
            missing_vars.append(var)
    
    if missing_vars:
        logger.error("\n❌ Missing environment variables:")
        for var in missing_vars:
            logger.error(f"- {var}")
    else:
        logger.info("\n✅ All required environment variables are set!")

if __name__ == "__main__":
    check_environment() 