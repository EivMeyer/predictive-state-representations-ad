# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# Set environment variable to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Use BuildKit's cache mount to speed up pip installs
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir torch torchvision

# Install PyTorch Geometric and its dependencies
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    --extra-index-url https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Combine all apt-get commands into one layer and use cache mount
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y -qq --no-install-recommends \
    # Basic dependencies
    git \
    build-essential \
    cmake \
    git-lfs \
    unzip \
    libboost-dev \
    libboost-thread-dev \
    libboost-test-dev \
    libboost-filesystem-dev \
    libeigen3-dev \
    libomp-dev \
    freeglut3-dev \
    libglib2.0-0 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    libxrender1 \
    wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set OpenGL environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Clone repositories and install dependencies
RUN git clone https://github.com/CommonRoad/crgeo.git /app/crgeo && \
    pip install -e /app/crgeo

# Create necessary directories
RUN mkdir -p /app/psr-ad/output /app/psr-ad/scenarios

# Set environment variables
ENV PYTHONPATH=/app/psr-ad:/app/crgeo
ENV CUDA_VISIBLE_DEVICES=0

# Set working directory
WORKDIR /app/psr-ad

# Default command
CMD ["bash"]
