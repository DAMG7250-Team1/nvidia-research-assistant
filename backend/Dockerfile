# Use Python 3. as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

 
# Copy requirements first to leverage Docker cache
COPY requirements.txt .

RUN pip install --prefer-binary --no-cache-dir -r requirements.txt


# Copy the rest of the application
COPY . .
COPY .env .
# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
