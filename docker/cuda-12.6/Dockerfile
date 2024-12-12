# syntax=docker/dockerfile:1.4

# Stage 1: PyG Builder - only for building PyG wheels
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04 as pyg-builder

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX" \
    MAX_JOBS=8 \
    PIP_NO_CACHE_DIR=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    MAKEFLAGS="-j8"

# Install build essentials
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
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

# Install PyTorch and build wheels
RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
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

# Stage 2: Main Builder
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX" \
    MAX_JOBS=8 \
    PATH=/opt/conda/bin:$PATH \
    CONDA_OVERRIDE_CUDA="12.1" \
    CONDA_PKGS_DIRS=/opt/conda/pkgs \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MAMBA_NO_BANNER=1 \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PYTHONDONTWRITEBYTECODE=1

# Install basic dependencies and micromamba
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    wget git git-lfs && \
    wget -q https://micromamba.snakepit.net/api/micromamba/linux-64/latest -O /usr/bin/micromamba && \
    chmod +x /usr/bin/micromamba && \
    mkdir -p /opt/conda/pkgs && \
    micromamba shell init --shell=bash --root-prefix=/opt/conda && \
    rm -rf /var/lib/apt/lists/*

# Copy environment.yml and install dependencies
COPY environment.yml /tmp/environment.yml
RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba create -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Copy pre-built wheels and install PyG
COPY --from=pyg-builder /wheels /wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    source /opt/conda/etc/profile.d/micromamba.sh && \
    micromamba activate myenv && \
    pip install /wheels/*.whl

# Stage 3: Runtime
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04 as runtime

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
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    freeglut3-dev \
    libglib2.0-0 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files from builder
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root/.bashrc /root/.bashrc

# Create directories and start Xvfb in one layer
RUN mkdir -p /app/psr-ad/output /app/psr-ad/scenarios /app/crgeo && \
    Xvfb :99 -screen 0 1024x768x16 &

# Clean up unnecessary files
RUN rm -rf /opt/conda/pkgs/* && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.pyc' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete

WORKDIR /app/psr-ad

SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["micromamba", "run", "-n", "myenv"]
CMD ["bash"]