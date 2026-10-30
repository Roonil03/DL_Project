# CUDA-enabled, reproducible image for the Financial DL Benchmark.
#
# The base image is deliberately pinned to a CUDA *development* image. Mamba
# and causal-conv1d contain CUDA extensions and must be built against the same
# CUDA/PyTorch toolchain that is used at runtime. The application itself runs
# as an unprivileged user and all Python packages live in /opt/venv.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# uv is copied from a pinned upstream image rather than installed into the
# project environment with pip. This keeps Python interpreter selection
# deterministic and avoids depending on the host Python installation.
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /usr/local/bin/

ARG PYTHON_VERSION=3.11.11
ARG UV_VERSION=0.6.14
ARG NB_UID=1000
ARG NB_GID=1000

LABEL org.opencontainers.image.title="fin-dl-benchmark" \
      org.opencontainers.image.description="CUDA 12.4 + Python 3.11 benchmark environment" \
      org.opencontainers.image.source="local-checkout" \
      org.opencontainers.image.version="cuda12.4-python3.11-uv${UV_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=1

# Keep compiler/build tooling in the image: mamba-ssm and causal-conv1d are
# installed from pinned source distributions when a matching wheel is absent.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        build-essential \
        ninja-build \
        zip \
        unzip \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${NB_GID}" jovyan \
    && useradd --uid "${NB_UID}" --gid "${NB_GID}" --create-home --shell /bin/bash jovyan \
    && mkdir -p /opt/uv-python /opt/venv /home/jovyan/work \
    && chown -R jovyan:jovyan /opt/uv-python /opt/venv /home/jovyan

# Install managed CPython and seed the venv with pip. The venv is created by
# uv; package installation below is intentionally done with pip3 so
# requirements.docker.txt remains a normal, inspectable lock-style file.
RUN uv python install "${PYTHON_VERSION}" \
    && uv venv --python "${PYTHON_VERSION}" --seed /opt/venv \
    && /opt/venv/bin/pip3 install --no-cache-dir \
        pip==24.3.1 \
        setuptools==75.6.0 \
        wheel==0.45.1 \
        packaging==24.2 \
        ninja==1.11.1.3

WORKDIR /home/jovyan/work

# Install the CUDA wheel and the remaining pinned packages in separate layers.
# mamba-ssm is installed with --no-build-isolation so its extension build sees
# this exact CUDA-enabled torch instead of pulling an unrelated CPU torch.
COPY requirements.docker.txt /tmp/requirements.docker.txt
RUN /opt/venv/bin/pip3 install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.4.1+cu124 \
    && /opt/venv/bin/pip3 install --no-cache-dir --no-build-isolation \
        -r /tmp/requirements.docker.txt \
    && rm -f /tmp/requirements.docker.txt \
    && /opt/venv/bin/python -m ipykernel install \
        --prefix=/opt/venv \
        --name fin-dl-benchmark \
        --display-name "Financial DL Benchmark (CUDA)"

# mamba-ssm's language-model helpers import the Transformers generation API.
# Pin the last compatible 4.x matrix; current 5.x releases require newer
# PyTorch and removed symbols imported by mamba-ssm 2.2.4.
COPY requirements.mamba-compat.txt /tmp/requirements.mamba-compat.txt
RUN /opt/venv/bin/pip3 install --no-cache-dir --force-reinstall --no-deps \
        -r /tmp/requirements.mamba-compat.txt \
    && rm -f /tmp/requirements.mamba-compat.txt

ENV BENCHMARK_IMAGE=fin-dl-benchmark:cuda12.4

# Copy source and orchestration assets into the image. Compose mounts data,
# notebooks, source, configs and outputs for live local runs.
COPY --chown=jovyan:jovyan configs/ ./configs/
COPY --chown=jovyan:jovyan src/ ./src/
COPY --chown=jovyan:jovyan notebooks/ ./notebooks/
COPY --chown=jovyan:jovyan tests/ ./tests/
COPY --chown=jovyan:jovyan run_all_models.sh ./run_all_models.sh
COPY --chown=jovyan:jovyan README.md RunningNotebooks.md DOCKER.md MIGRATION_SUMMARY.md ./

RUN chmod 0755 /home/jovyan/work/run_all_models.sh \
    && mkdir -p /home/jovyan/.local/share/jupyter/runtime \
    && chown -R jovyan:jovyan /home/jovyan

USER jovyan
WORKDIR /home/jovyan/work

EXPOSE 8888

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token=", "--ServerApp.allow_origin=*"]
