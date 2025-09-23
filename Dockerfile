# Use official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install SSL certificates
RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates

# Copy application files
COPY requirements.txt .
COPY . .

# Install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Hugging Face cache directory
RUN mkdir -p /app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

# Container internal port (fallback to 8000 if not set)
EXPOSE ${PORT:-8000}

# Startup command
CMD ["sh", "-c", "uvicorn app:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
