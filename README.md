# Search-R1: reproducible BM25 + GRPO training

This fork packages a verified Search-R1 workflow for training Qwen2.5-3B with
GRPO and a local Wikipedia BM25 retriever. It targets shared NVIDIA servers,
keeps persistent assets separate from host-local runtime files, records runs
with SwanLab, and supports both native and Docker-based deployment.

The implementation is based on
[PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1).
Please cite the original Search-R1 papers and retain the upstream license.

## Verified configuration

- Model: `Qwen/Qwen2.5-3B` base
- Data: `PeterJinGo/nq_hotpotqa_train`
- Retriever: `PeterJinGo/wiki-18-bm25-index`
- Corpus: `PeterJinGo/wiki-18-corpus`
- Training: GRPO, FSDP CPU offload, vLLM 0.6.3
- Hardware tested: NVIDIA A800-SXM4-40GB
- Metrics: SwanLab project `search-r1`, workspace `nileonx`

Measured results:

- 4 GPUs: stable; approximately 26.2 GB peak reserved memory per GPU.
- 2 GPUs: stable and recommended for long jobs; approximately 25.0 GB peak
  reserved memory per GPU.
- 1 GPU: unsupported for this 3B configuration. Both normal and reduced-memory
  tests OOMed while synchronizing actor weights into vLLM at about 39.4 GiB.

See [the full reproduction report](docs/reproduction_a800_bm25.md) for measured
timings, failure analysis, and experiment settings.

## Storage model

Keep large reusable assets on persistent storage and high-churn runtime files
on local memory or SSD:

```text
SEARCH_R1_ROOT=/persistent/search-r1-run
  models/ data/ indexes/ corpus/ checkpoints/ logs/ swanlab/

SEARCH_R1_RUNTIME_ROOT=/dev/shm/$USER-search-r1
  envs/ ray/ tmp/ pip-cache/ active checkpoints/ extracted corpus
```

Approximate downloads are 5.8 GB for the model, 2.3 GB for BM25, 426 MB for
training data, and 5.1 GB for the compressed corpus. The extracted corpus is
about 14.4 GB.

## Native quick start

Requirements:

- Linux with at least 2 NVIDIA GPUs with 40 GB VRAM each
- CUDA driver compatible with CUDA 12.1
- Conda or Miniconda
- OpenJDK 21
- At least 50 GB free persistent storage
- At least 50 GB free local `/dev/shm` for a two-GPU run

```bash
git clone https://github.com/Leonx01/Search-R1.git
cd Search-R1

export SEARCH_R1_ROOT=/path/to/persistent/search-r1-run
export SEARCH_R1_RUNTIME_ROOT=/dev/shm/$USER-search-r1

scripts/local_4xa800/create_envs.sh
scripts/local_4xa800/setup_envs.sh
scripts/local_4xa800/download_assets.sh
CUDA_VISIBLE_DEVICES=0,1 scripts/local_4xa800/verify_envs.sh
```

Start the retriever in one terminal:

```bash
export SEARCH_R1_ROOT=/path/to/persistent/search-r1-run
export SEARCH_R1_RUNTIME_ROOT=/dev/shm/$USER-search-r1
scripts/local_4xa800/launch_bm25.sh
```

Configure SwanLab without placing the key in shell history:

```bash
mkdir -p "$SEARCH_R1_ROOT/.secrets"
chmod 700 "$SEARCH_R1_ROOT/.secrets"
read -rsp "SwanLab API key: " key; echo
printf '%s\n' "$key" > "$SEARCH_R1_ROOT/.secrets/swanlab_api_key"
chmod 600 "$SEARCH_R1_ROOT/.secrets/swanlab_api_key"
unset key
```

Run a two-GPU experiment:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
EXPERIMENT_PRESET=a800_2gpu_overnight \
scripts/local_4xa800/run_experiment.sh
```

The generic launcher also accepts environment and Hydra overrides:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
TOTAL_STEPS=60 \
TRAIN_BATCH_SIZE=16 \
EXPERIMENT_NAME=my-search-r1-run \
scripts/local_4xa800/train_grpo.sh \
actor_rollout_ref.actor.optim.lr=2e-6
```

`CUDA_VISIBLE_DEVICES` selects physical cards. The launcher derives the GPU
count and applies tested defaults. It rejects one GPU by default.

## Reproduced experiments

Versioned presets live in
[`configs/experiments`](configs/experiments):

- `a800_4gpu_smoke.env`: five-step end-to-end validation.
- `a800_2gpu_overnight.env`: 300-step long run, validation every 25 steps,
  checkpoint every 100 steps.

Verified four-GPU smoke run:
[SwanLab metrics](https://swanlab.cn/@nileonx/search-r1/runs/cegy07pb).

Active checkpoints are written to the runtime disk. After successful training,
the latest actor checkpoint is copied sequentially to persistent storage to
avoid concurrent FSDP writes stalling NFS.

## SwanLab

SwanLab is a first-class veRL tracking backend in this fork. Each run records:

- full resolved Hydra configuration;
- hostname, visible GPU IDs, GPU count, and git commit;
- reward, KL, entropy, gradient norm, response length, valid-search ratio;
- validation scores and per-stage timings.

Online mode requires `SWANLAB_API_KEY` or
`$SEARCH_R1_ROOT/.secrets/swanlab_api_key`. The key is used with
`save=False`; it is never stored in this repository or the shared home
directory. Set `SWANLAB_MODE=offline` for disconnected diagnostics.

## Docker

The repository includes separate training and retrieval images under
[`docker/`](docker):

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d retriever
docker compose \
  -f docker/docker-compose.yml \
  -f docker/compose.2gpu.yml \
  run --rm train
```

The host must provide the NVIDIA driver and NVIDIA Container Toolkit. The
training container requires large shared memory and IPC access; the Compose
configuration sets these explicitly. Use a unique Compose project name and
runtime directory on shared servers.

Docker improves portability, but it does not allocate GPUs or solve NFS
performance. Do not build images where Docker's data root is on a full system
partition.

## BM25 API

The lightweight server exposes:

```text
GET  /health
POST /retrieve
```

Example:

```bash
curl -sS http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["Who developed relativity?"],"topk":3}'
```

The official BM25 index stores document IDs rather than source text, so the
corpus is required. On first startup, the server builds a compact persistent
line-offset index for random JSONL access.

## Troubleshooting highlights

- Do not create Conda environments on slow NFS; use local SSD or `/dev/shm`.
- Do not save FSDP checkpoints directly to NFS; stage locally and archive.
- The corpus named `wiki-18.jsonl.gz` is a gzip-compressed tar archive.
- Pyserini 0.44 imports optional encoder modules eagerly; the retriever image
  includes the required CPU-side compatibility dependencies.
- Use Java 21 and set `JAVA_HOME`.
- High-concurrency Hugging Face downloads may destabilize local proxies; this
  workflow limits workers and supports `HF_ENDPOINT`.
- Pin NumPy and OpenCV before vLLM 0.6.3 to avoid resolver backtracking.

## Upstream papers

```bibtex
@article{jin2025search,
  title={Search-r1: Training llms to reason and leverage search engines with reinforcement learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}

@article{jin2025empirical,
  title={An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents},
  author={Jin, Bowen and Yoon, Jinsung and Kargupta, Priyanka and Arik, Sercan O and Han, Jiawei},
  journal={arXiv preprint arXiv:2505.15117},
  year={2025}
}
```
