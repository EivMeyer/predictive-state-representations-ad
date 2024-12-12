# syntax=docker/dockerfile:1.4

##############################################
# Stage 1: PyG Builder - Build PyG wheels
##############################################
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as pyg-builder

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX" \
    MAX_JOBS=8 \
    PIP_NO_CACHE_DIR=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    MAKEFLAGS="-j8"

# Install system dependencies for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    build-essential \
    ninja-build \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Set up ccache
RUN mkdir -p /root/.ccache && \
    echo "max_size = 5.0G" > /root/.ccache/ccache.conf && \
    echo "compression = true" >> /root/.ccache/ccache.conf

# Install PyTorch and dependencies to build PyG wheels
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    mkdir -p /wheels && \
    CC="ccache gcc" CXX="ccache g++" \
    pip wheel \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html \
    -w /wheels

##############################################
# Stage 2: Main Builder - Setup Conda and env
##############################################
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX" \
    MAX_JOBS=8 \
    PATH=/opt/conda/bin:$PATH \
    CONDA_OVERRIDE_CUDA="12.1" \
    CONDA_PKGS_DIRS=/opt/conda/pkgs \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    git-lfs \
    bzip2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean --all --yes && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Create a conda environment and install basic packages
# Adjust python version and packages as needed
RUN conda install -n base -c conda-forge mamba -y && \
    mamba create -n myenv python=3.10 -y && \
    conda clean --all --yes

##############################################
# Stage 3: Runtime Stage
##############################################
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    CUDA_VISIBLE_DEVICES=0 \
    PYGLET_HEADLESS=1 \
    DISPLAY=:99 \
    PYTHONPATH=/app/psr-ad:/app/crgeo

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    freeglut3-dev \
    libglib2.0-0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy conda environment from builder
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Copy PyG wheels from builder
COPY --from=pyg-builder /wheels /wheels

# Install PyG wheels into myenv
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate myenv && \
    pip install /wheels/*.whl && \
    rm -rf /wheels

# Create application directories
RUN mkdir -p /app/psr-ad/output /app/psr-ad/scenarios /app/crgeo

WORKDIR /app/psr-ad

# Create an entrypoint script to activate conda env & start Xvfb
# This ensures that when the container starts, we have a working display and environment.
RUN echo '#!/usr/bin/env bash\n' \
    'set -e\n' \
    'source /opt/conda/etc/profile.d/conda.sh\n' \
    'conda activate myenv\n' \
    'Xvfb :99 -screen 0 1024x768x16 &\n' \
    'exec "$@"' > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

