FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements_serving.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY 3.prometheus_exporter.py app.py
COPY wine_quality_preprocessing/ wine_quality_preprocessing/

# Expose ports
# 8000: FastAPI model serving
# 8001: Prometheus metrics
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the server
CMD ["python", "app.py"]
