import re
from typing import List
import logging
from nltk.tokenize import sent_tokenize
import nltk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {str(e)}")

def chunk_markdown(text: str, heading_level: int = 2) -> List[str]:
    """Split markdown text into chunks based on headings."""
    try:
        # Create heading pattern based on level
        heading_pattern = r'^#{1,' + str(heading_level) + r'}\s+.+$'
        
        # Split text into lines
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        
        for line in lines:
            if re.match(heading_pattern, line, re.MULTILINE):
                # If we have a current chunk, save it
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                # Start new chunk with heading
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        logger.info(f"Created {len(chunks)} markdown chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Error in chunk_markdown: {str(e)}")
        return [text]  # Return original text as single chunk

def chunk_semantic(text: str, max_sentences: int = 10) -> List[str]:
    """Split text into chunks based on semantic boundaries (sentences)."""
    try:
        # Split text into sentences
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= max_sentences:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Error in chunk_semantic: {str(e)}")
        return [text]  # Return original text as single chunk

def chunk_sliding_window(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks of specified size."""
    try:
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # If text is shorter than chunk size, return as is
        if len(text) <= chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk of text
            end = start + chunk_size
            chunk = text[start:end]
            
            # If not at the end, try to break at a space
            if end < len(text):
                # Find last space in chunk
                last_space = chunk.rfind(' ')
                if last_space != -1:
                    end = start + last_space
                    chunk = text[start:end]
            
            chunks.append(chunk)
            # Move start position by chunk size minus overlap
            start = end - overlap
            
        logger.info(f"Created {len(chunks)} sliding window chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Error in chunk_sliding_window: {str(e)}")
        return [text]  # Return original text as single chunk 