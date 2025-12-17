# Dockerfile for Fake News Detection Streamlit app
# Uses slim Python 3.11 image

FROM python:3.11-slim

# Reduce logs from TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for some Python wheels/builds
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       g++ \
       git \
       libopenblas-dev \
       liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY . /app

# Create a directory for models to be mounted (optional)
RUN mkdir -p /app/models

# Install Python dependencies
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Ensure entrypoint script has execute permission
RUN chmod +x /app/entrypoint.sh

# Expose Streamlit default port
EXPOSE 8501

# Use entrypoint to download NLTK data then run the container CMD
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]

# Default command runs the Streamlit app on all network interfaces
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
