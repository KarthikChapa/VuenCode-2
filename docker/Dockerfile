# VuenCode Multi-Stage Dockerfile
# Optimized for both local development and GPU competition deployment

# Build arguments for deployment mode selection
ARG DEPLOYMENT_MODE=local
ARG PYTHON_VERSION=3.11

# === Stage 1: Base Dependencies ===
FROM python:${PYTHON_VERSION}-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash vuencode
USER vuencode

# === Stage 2: Local Development Build ===
FROM base AS local

# Copy requirements for local development
COPY --chown=vuencode:vuencode docker/requirements-local.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --user --no-cache-dir --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# Copy application source
COPY --chown=vuencode:vuencode . /app/

# Set environment variables for local development
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PATH="/home/vuencode/.local/bin:$PATH"
ENV VUENCODE_DEPLOYMENT_MODE=local

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# === Stage 3: GPU Production Build ===
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 AS gpu-base

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    ffmpeg \
    wget \
    curl \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash vuencode
USER vuencode

FROM gpu-base AS gpu

# Copy GPU requirements
COPY --chown=vuencode:vuencode docker/requirements-gpu.txt /app/requirements.txt

# Install Python dependencies with GPU support
RUN pip install --user --no-cache-dir --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# Copy application source
COPY --chown=vuencode:vuencode . /app/

# Set environment variables for GPU deployment
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PATH="/home/vuencode/.local/bin:$PATH"
ENV VUENCODE_DEPLOYMENT_MODE=gpu
ENV CUDA_VISIBLE_DEVICES=0

# Expose port
EXPOSE 8000

# Health check with longer timeout for GPU initialization
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with gunicorn for production
CMD ["gunicorn", "api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# === Stage 4: Competition Optimized Build ===
FROM gpu AS competition

# Additional competition-specific optimizations
USER root

# Install TensorRT (placeholder - would need actual TensorRT packages)
# RUN wget -qO - https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/8.6.1/tars/tensorrt-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz | tar -xz

# Copy competition configuration
COPY --chown=vuencode:vuencode configs/competition.env /app/.env

USER vuencode

# Set competition environment
ENV VUENCODE_DEPLOYMENT_MODE=competition
ENV VUENCODE_PERFORMANCE_PROFILE=competition
ENV COMPETITION_MODE=true

# Competition-optimized startup
CMD ["gunicorn", "api.main:app", "-w", "8", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--worker-connections", "1000", "--max-requests", "1000", "--preload"]

# === Final Stage Selection ===
FROM ${DEPLOYMENT_MODE} AS final