# ==========================================
# 1. Base Image with CUDA 12.6 Support
# ==========================================
# Ubuntu 24.04 isn't fully standardized for all ML bases yet, 
# so we use NVIDIA's official Ubuntu 22.04 base which runs perfectly on both host OS systems.
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set up system dependencies needed for Git submodules, OpenCV, and Conda
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    bzip2 \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# 2. Install Miniconda
# ==========================================
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Add conda to path
ENV PATH=$CONDA_DIR/bin:$PATH

# ==========================================
# 3. Setup Project Directory
# ==========================================
WORKDIR /workspace/blur_detection

# Copy ONLY the dependency file first to take advantage of Docker layer caching
COPY conda.yaml .

# Accept Anaconda Terms of Service non-interactively and build the environment
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f conda.yaml --yes && \
    conda clean -afy

# ==========================================
# 4. Copy Project Files
# ==========================================
# Copy the rest of the source code (Submodules, main app, etc.)
COPY . .

# Ensure scripts have execution permissions
ENV PYTHONPATH=/workspace/blur_detection:/workspace/blur_detection/blur_detection:/workspace/blur_detection/sam3:/workspace/blur_detection/dinov3
RUN chmod +x setup.sh start_app.sh

# ==========================================
# 5. Shell Environment Configuration
# ==========================================
# Force Docker to use bash login shell so conda activates correctly
SHELL ["/bin/bash", "--login", "-c"]
RUN echo "conda activate blur_detection" >> ~/.bashrc

# Expose Streamlit's default port
EXPOSE 8501

# ==========================================
# 6. Execution Entrypoint
# ==========================================
# We use the explicit python environment path to call Streamlit safely
CMD ["/opt/conda/envs/blur_detection/bin/streamlit", "run", "blur_detection/main.py", "--server.port=8501", "--server.address=0.0.0.0"]