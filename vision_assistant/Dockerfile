# ============================================================
# AI Vision Assistant — Dockerfile
# ============================================================
# Builds a containerised version of the vision assistant.
# Runs in headless (--no-display) mode by default.
#
# Build:  docker build -t vision-assistant .
# Run:    docker run --device /dev/video0 vision-assistant
# Run (lite): docker run --device /dev/video0 vision-assistant --lite
# ============================================================

FROM python:3.10-slim

# Install system dependencies for OpenCV and audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    espeak \
    libespeak1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model weights at build time (cached in Docker layer)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
RUN python -c "import torch; torch.hub.load('intel-isl/MiDaS', 'DPT_Small', trust_repo='check')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models logs audio

# Default: run in headless mode
ENTRYPOINT ["python", "main.py", "--no-display"]
