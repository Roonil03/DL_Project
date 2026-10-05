# Financial DL Benchmark — Jupyter + PyTorch container with isolated venv.
# Base: Jupyter pytorch-notebook (Python 3.11 with PyTorch preinstalled).
# Everything (venv, deps, kernel) lives INSIDE the image. The host is never
# written to outside of mounted data/results/notebooks volumes at *run* time.
# Build/run are manual steps for the user (not executed by the migration):
#   docker build -t fin-dl-benchmark .
#   docker compose up jupyter
FROM quay.io/jupyter/pytorch-notebook:python-3.11

LABEL org.opencontainers.image.title="fin-dl-benchmark" \
      org.opencontainers.image.description="Isolated Jupyter+PyTorch env for the Financial DL Benchmark (venv at /opt/venv, kernel fin-dl-benchmark)" \
      org.opencontainers.image.source="local-checkout"

# --- System layer (as root, minimal) ---
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/venv \
    && chown -R ${NB_USER}:${NB_GID} /opt/venv

# --- Project env (as non-root jovyan) ---
USER ${NB_USER}
WORKDIR /home/jovyan/work

# Isolated virtual environment INSIDE the image (not on the host).
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Install pinned deps into the venv first (better layer caching).
COPY --chown=${NB_USER}:${NB_GID} requirements.docker.txt ./requirements.docker.txt
# Reuse the scientific/PyTorch stack already supplied by the base image.  A
# plain venv would hide those packages and make pip download a second PyTorch
# plus a complete CUDA toolchain, adding several gigabytes without improving
# the runtime.  The environment is still isolated from the host: both the
# venv and its inherited site-packages live inside this image.
RUN python -m venv --system-site-packages /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.docker.txt \
    && /opt/venv/bin/python -m ipykernel install --prefix=/opt/venv --name fin-dl-benchmark --display-name "Financial DL Benchmark"

# Copy project code (data/*.csv and heavy artifacts are excluded via .dockerignore
# and provided at run time through mounted volumes instead).
COPY --chown=${NB_USER}:${NB_GID} configs/ ./configs/
COPY --chown=${NB_USER}:${NB_GID} src/ ./src/
COPY --chown=${NB_USER}:${NB_GID} notebooks/ ./notebooks/
COPY --chown=${NB_USER}:${NB_GID} tests/ ./tests/
COPY --chown=${NB_USER}:${NB_GID} README.md DOCKER.md MIGRATION_SUMMARY.md ./

EXPOSE 8888

# Start JupyterLab/Notebook as jovyan. Authentication is disabled for this local
# Compose workflow, whose host port is bound to 127.0.0.1 only.
CMD ["start-notebook.sh", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--IdentityProvider.token=''"]
