FROM python:3.10-slim

# Metadata
LABEL maintainer="Flight Delay Prediction Platform"
LABEL version="1.0.0"
LABEL description="End-to-End ML System for Flight Delay Prediction"

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required directories
RUN mkdir -p data models reports mlflow_tracking

# Create __init__ files
RUN touch etl/__init__.py features/__init__.py \
         models/__init__.py api/__init__.py monitoring/__init__.py

# Expose Gradio + FastAPI port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health')" \
    || exit 1

# Run the Gradio app (includes FastAPI)
CMD ["python", "app.py"]
