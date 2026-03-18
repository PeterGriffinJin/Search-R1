---
name: setup-search-r1-ubuntu-22.04
description: Full environment setup for the Search-R1 project on Ubuntu 22.04. Installs Miniconda, creates the searchr1 (Python 3.9) and retriever (Python 3.10) conda environments, and installs all dependencies as specified in the Search-R1 README. Use when setting up Search-R1 from scratch on a fresh Ubuntu 22.04 machine with an NVIDIA GPU.
argument-hint: [project-path]
allowed-tools: Bash
---

# Setup Search-R1 on Ubuntu 22.04

Set up the full Search-R1 environment by following the README exactly. The project path is `$ARGUMENTS` (default: current working directory if not provided).

## Prerequisites check

First, verify the environment:
```bash
which conda 2>/dev/null || echo "conda not found"
nvidia-smi | head -5
nvcc --version 2>/dev/null || echo "nvcc not found"
```

## Step 1 — Install Miniconda (if conda not found)

```bash
cd /tmp && wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /root/miniconda3
export PATH="/root/miniconda3/bin:$PATH"
conda --version
```

Accept the Anaconda Terms of Service (required on newer Miniconda versions):
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Initialize conda for bash (so `conda activate` works in future shells):
```bash
conda init bash
source /root/miniconda3/etc/profile.d/conda.sh
```

## Step 2 — Create the `searchr1` environment (Python 3.9)

```bash
export PATH="/root/miniconda3/bin:$PATH"
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n searchr1 python=3.9 -y
```

## Step 3 — Install dependencies in `searchr1`

Activate the environment and install torch with CUDA 12.1:
```bash
conda activate searchr1
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

Install vllm:
```bash
pip3 install vllm==0.6.3
```

Install verl in editable mode from the project root:
```bash
cd $ARGUMENTS  # or the Search-R1 project root
pip install -e .
```

### Install flash-attn

The system likely has no `nvcc` even if the NVIDIA driver is present.
First install `nvcc` via conda so the flash-attn build can find CUDA:
```bash
conda install -c nvidia cuda-nvcc=12.1 -y
```

Then install flash-attn using the pre-built wheel to avoid the cross-device
link error that occurs when building from source in some container setups:
```bash
wget -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl" \
  -O "/tmp/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"
pip install "/tmp/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"
```

> **Why the pre-built wheel?**
> `pip3 install flash-attn --no-build-isolation` attempts to move the built
> wheel across filesystem boundaries (e.g. `/workspace` cache → install dir),
> which fails with `[Errno 18] Invalid cross-device link` inside containers.
> The pre-built wheel from the flash-attention GitHub releases sidesteps this.

Install wandb:
```bash
pip install wandb
```

Verify the `searchr1` environment:
```bash
python -c "import torch, vllm, flash_attn, wandb, verl; print('torch:', torch.__version__); print('vllm:', vllm.__version__); print('flash_attn:', flash_attn.__version__); print('wandb:', wandb.__version__)"
```

## Step 4 — Create the `retriever` environment (Python 3.10)

```bash
conda create -n retriever python=3.10 -y
conda activate retriever
```

Install PyTorch via conda (recommended for faiss-gpu compatibility):
```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

Install retrieval libraries:
```bash
pip install transformers datasets pyserini
```

Install faiss-gpu:
```bash
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y
```

Install API server dependencies:
```bash
pip install uvicorn fastapi
```

Verify the `retriever` environment:
```bash
python -c "import torch, transformers, datasets, faiss, uvicorn, fastapi; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('faiss available')"
```

## Summary

After this setup you have:

| Environment | Python | Key packages |
|-------------|--------|-------------|
| `searchr1`  | 3.9    | torch 2.4.0+cu121, vllm 0.6.3, verl (editable), flash-attn 2.8.3, wandb |
| `retriever` | 3.10   | torch 2.4.0+cu121 (conda), transformers, datasets, pyserini, faiss-gpu 1.8.0, uvicorn, fastapi |

Use them as:
```bash
conda activate searchr1   # RL training
conda activate retriever  # local retrieval server
```

## Known issues & workarounds

- **`conda create` fails with ToS error** — Run the two `conda tos accept` commands in Step 1 and retry.
- **`flash-attn` build fails with `CUDA_HOME not set`** — Install `cuda-nvcc=12.1` via `conda install -c nvidia cuda-nvcc=12.1 -y` inside the `searchr1` env.
- **`flash-attn` build fails with `Invalid cross-device link`** — Use the pre-built wheel approach in Step 3 instead of `pip3 install flash-attn --no-build-isolation`.
- **`vllm.__version__` shows `dev`** — This is a cosmetic issue caused by a missing `_version` module; vllm 0.6.3 is correctly installed.
