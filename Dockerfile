# Use Python 3.9 slim image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create temp directories with appropriate permissions
RUN mkdir -p /tmp/uploads /tmp/outputs \
    && chmod 777 /tmp/uploads /tmp/outputs

# Set environment variables
ENV UPLOAD_FOLDER=/tmp/uploads
ENV OUTPUT_FOLDER=/tmp/outputs
ENV PORT=8080

# Use gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app 