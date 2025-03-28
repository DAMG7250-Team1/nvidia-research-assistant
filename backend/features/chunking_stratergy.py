import re
import spacy
import spacy.cli
import tiktoken
from typing import List
import logging

TOKEN_LIMIT = 8192  # OpenAI's embedding model limit
SUB_CHUNK_SIZE = 2000  # Safe sub-chunk size to avoid exceeding limits

logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    
# Load tokenizer for token counting
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

def count_tokens(text):
    """Returns the token count of a given text."""
    return len(tokenizer.encode(text))

def split_chunk(chunk, max_tokens=SUB_CHUNK_SIZE):
    """Splits a chunk into smaller sub-chunks if it exceeds max_tokens."""
    tokens = tokenizer.encode(chunk)
    if len(tokens) <= max_tokens:
        return [chunk]  # Already within the limit

    sub_chunks = []
    for i in range(0, len(tokens), max_tokens):
        sub_tokens = tokens[i:i + max_tokens]
        sub_chunks.append(tokenizer.decode(sub_tokens))
    
    return sub_chunks

def markdown_chunking(text: str, heading_level: int = 2) -> List[str]:
    """
    Split markdown text into chunks based on headings.
    
    Args:
        text (str): The markdown text to split
        heading_level (int): The heading level to split on (1-6)
        
    Returns:
        List[str]: List of text chunks
    """
    try:
        # Validate input
        if text is None:
            logger.warning("Input text is None")
            return []
            
        # Validate heading level
        if not 1 <= heading_level <= 6:
            raise ValueError("Heading level must be between 1 and 6")
        
        # Create regex patterns for both markdown and plain text headings
        markdown_pattern = f"^#{{{heading_level}}}\\s+(.+)$"
        plain_text_pattern = r"^([A-Z][A-Z\s]+)$"  # Matches all-caps lines
        
        # Split text into sections based on headings
        sections = re.split(f"({markdown_pattern}|{plain_text_pattern})", text, flags=re.MULTILINE)
        
        # Process sections
        chunks = []
        current_heading = None
        
        for i, section in enumerate(sections):
            if section is None:
                continue
                
            if i % 2 == 0:  # This is the content
                if current_heading and section and section.strip():
                    # Add heading to content
                    chunk = f"## {current_heading}\n\n{section.strip()}"
                    chunks.append(chunk)
            else:  # This is the heading
                current_heading = section.strip() if section else None
                # Remove any markdown # characters from the heading
                if current_heading:
                    current_heading = re.sub(r'^#+\s*', '', current_heading)
        
        logger.info(f"Split text into {len(chunks)} chunks using markdown headings")
        return chunks
        
    except Exception as e:
        logger.error(f"Error in markdown_chunking: {str(e)}")
        logger.error("Full error details:", exc_info=True)
        return []

def semantic_chunking(text, max_sentences=5):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
   
    chunks = []
    current_chunk = []
    for i, sent in enumerate(sentences):
        current_chunk.append(sent)
        if (i + 1) % max_sentences == 0:
            merged_chunk = " ".join(current_chunk)
            chunks.extend(split_chunk(merged_chunk, max_tokens=TOKEN_LIMIT // 2))  # Ensure token limits
            current_chunk = []
   
    if current_chunk:
        merged_chunk = " ".join(current_chunk)
        chunks.extend(split_chunk(merged_chunk, max_tokens=TOKEN_LIMIT // 2))
   
    return chunks

def sliding_window_chunking(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks using a sliding window approach.
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk.strip())
        start = start + chunk_size - overlap

    print(f"Generated {len(chunks)} chunks using sliding window strategy")
    return chunks