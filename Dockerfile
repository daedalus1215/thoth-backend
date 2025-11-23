FROM nvidia/cuda:12.4.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir websockets uvicorn[standard]  # Explicitly install websockets

# Copy your application code
COPY . .

# Create SSL directory
RUN mkdir -p /app/ssl

# Expose the port
EXPOSE 8000

# Run the FastAPI application with explicit websocket support
# SSL configuration is handled via environment variables in docker-compose.yml or .env file
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
