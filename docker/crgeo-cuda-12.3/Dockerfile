FROM python:3.10-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    PYGLET_HEADLESS=1 \
    DISPLAY=:99 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    wget git git-lfs curl ca-certificates \
    xvfb freeglut3-dev libglib2.0-0 libgl1 libglx0 \
    libegl1 libxext6 libx11-6 libxrender1 \
    libgeos-dev libproj-dev libgdal-dev gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 (adjust versions as needed)
RUN pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install PyG from wheels
RUN pip install \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Install Numpy first since polygon3 needs it
RUN pip install numpy

# Install other dependencies required by crgeo
RUN pip install shapely cartopy pyqt5 pyglet freetype-py rtree pyopengl pyopengl-accelerate

# Install crgeo (if you know a stable version, specify it with @<tag>)
RUN pip install git+https://github.com/CommonRoad/crgeo.git

# Create directories if needed
RUN mkdir -p /app/psr-ad/output /app/psr-ad/scenarios /app/crgeo
WORKDIR /app/psr-ad

# Xvfb startup script if needed
RUN echo '#!/usr/bin/env bash\n' \
    'set -e\n' \
    'Xvfb :99 -screen 0 1024x768x16 &\n' \
    'exec "$@"' > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
