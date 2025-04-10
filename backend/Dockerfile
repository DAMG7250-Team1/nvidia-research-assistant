# Use Python 3. as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install critical dependencies first
RUN pip install --prefer-binary --no-cache-dir numpy==1.26.4 pyarrow==18.1.0

# Install langchain dependencies with specific versions to avoid lengthy resolution
RUN pip install --prefer-binary --no-cache-dir \
    langchain==0.1.16 \
    langchain-openai==0.0.2 \
    langgraph==0.0.40 \
    langchain-core==0.1.30 \
    langchain-text-splitters==0.0.1

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install remaining requirements with --no-deps to avoid resolving dependencies again
RUN grep -v "langchain\|numpy\|pyarrow" requirements.txt > remaining_requirements.txt && \
    pip install --prefer-binary --no-cache-dir -r remaining_requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
