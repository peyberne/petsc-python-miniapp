# PETSc + petsc4py + Python environment for petsc-python-miniapp
#
# CPU-only base image for local development and CI.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-petsc4py \
    petsc-dev \
    libopenmpi-dev \
    openmpi-bin \
    git \
    build-essential \
    python3-numpy \
    python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

# No pip install of numpy here, to keep ABI compatible with petsc4py
# You can still install *other* Python packages later with pip if needed.

WORKDIR /app

CMD ["/bin/bash"]
