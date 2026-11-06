# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=ubhub.ubtrobot.com/ogi/chen-liao-world-model:v0.0.3
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG BASE_IMAGE
ARG CONDA_HOME=/home/fengodchen/miniconda3
ARG ENV_NAME=motus311
ARG PYTHON_VERSION=3.11.14
ARG CONDA_MAIN_CHANNEL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG FLASH_ATTN_VERSION=2.8.3
ARG MAX_JOBS=8
ARG TORCH_CUDA_ARCH_LIST=9.0

ENV CUDA_HOME=/usr/local/cuda \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MOTUS_PYTHON_ENV=${ENV_NAME}

COPY requirements-training-py311.lock /tmp/motus-image/
COPY requirements-egoverse.lock /tmp/motus-image/
COPY validate_image.py /tmp/motus-image/

RUN "${CONDA_HOME}/bin/conda" create -y \
      --override-channels \
      --channel "${CONDA_MAIN_CHANNEL}" \
      -n "${ENV_NAME}" \
      "python=${PYTHON_VERSION}" \
      "pip=25.3" \
      "setuptools=80.9.0" \
      "wheel=0.45.1" \
    && "${CONDA_HOME}/bin/conda" clean -afy

ENV PATH=${CONDA_HOME}/envs/${ENV_NAME}/bin:${PATH}

# Install the CUDA build first. It supplies the matching triton and NVIDIA
# runtime wheels. The remaining baseline is then restored without dependency
# re-resolution.
RUN python -m pip install \
      --index-url "${PYTORCH_INDEX_URL}" \
      "torch==2.7.1+cu128" \
      "torchvision==0.22.1+cu128"

RUN python -m pip install \
      --index-url "${PIP_INDEX_URL}" \
      --no-deps \
      -r /tmp/motus-image/requirements-training-py311.lock

# The v0.0.3 environment records both OpenCV distributions. Reinstall the
# headless build last so cv2 does not require libGL inside training pods.
RUN python -m pip install \
      --index-url "${PIP_INDEX_URL}" \
      --no-deps \
      --force-reinstall \
      "opencv-python-headless==4.12.0.88"

RUN python -m pip install \
      --index-url "${PIP_INDEX_URL}" \
      --no-deps \
      -r /tmp/motus-image/requirements-egoverse.lock

# flash-attn contains a CPython/CUDA extension and must be rebuilt for cp311.
RUN MAX_JOBS="${MAX_JOBS}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    python -m pip install \
      --index-url "${PIP_INDEX_URL}" \
      --no-build-isolation \
      --no-deps \
      "flash-attn==${FLASH_ATTN_VERSION}"

RUN python /tmp/motus-image/validate_image.py

LABEL org.motus.base-image="${BASE_IMAGE}" \
      org.motus.python-environment="${ENV_NAME}" \
      org.motus.python-version="${PYTHON_VERSION}" \
      org.motus.egoverse-zarr="3.1.5"

CMD ["bash"]
