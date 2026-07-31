# Portable local training

All paths and runtime settings can be overridden with environment variables. Persistent
assets default to `/mnt/private/nijianqiao/search-r1-run`; environments, corpus, Ray
state, and active checkpoints default to `/dev/shm/njq-searchr1`.

Create and install both native environments:

```bash
scripts/local_4xa800/create_envs.sh
scripts/local_4xa800/setup_envs.sh
CUDA_VISIBLE_DEVICES=0,1 scripts/local_4xa800/verify_envs.sh
```

For repeated runs across shared servers, store it without exposing it in shell history:

```bash
mkdir -p "$SEARCH_R1_ROOT/.secrets"
chmod 700 "$SEARCH_R1_ROOT/.secrets"
read -rsp "SwanLab API key: " key; echo
printf '%s\n' "$key" > "$SEARCH_R1_ROOT/.secrets/swanlab_api_key"
chmod 600 "$SEARCH_R1_ROOT/.secrets/swanlab_api_key"
unset key
```

The key is loaded from that private file and used with `save=False`; it is not written
into this repository or the shared home directory. Metrics are recorded under project
`search-r1`, workspace `nileonx`.

Start the BM25 service:

```bash
scripts/local_4xa800/launch_bm25.sh
```

Choose GPUs with `CUDA_VISIBLE_DEVICES`; the launcher derives the GPU count and applies
safe defaults for batch sizes and vLLM memory:

```bash
# Two 40GB GPUs: recommended minimum
CUDA_VISIBLE_DEVICES=2,3 scripts/local_4xa800/train_grpo.sh

# Four GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/local_4xa800/train_grpo.sh

# Eight GPUs on another server
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 scripts/local_4xa800/train_grpo.sh
```

Qwen2.5-3B was tested with both standard and ultra-low-memory settings on one
40GB A800; both runs OOMed while synchronizing actor weights into vLLM. The launcher
therefore rejects one GPU by default. `ALLOW_SINGLE_GPU=1` exists only for developers
testing a different model or memory architecture.

Common overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TOTAL_STEPS=60 \
TRAIN_BATCH_SIZE=16 \
EXPERIMENT_NAME=qwen3b-bm25-2gpu-seed1 \
scripts/local_4xa800/train_grpo.sh
```

Hydra overrides can also be appended to the command. Set `ENABLE_SWANLAB=0` only for
offline diagnostics. Successful runs archive their latest actor checkpoint from
`/dev/shm` to the persistent checkpoint directory automatically.

Versioned presets can be run with:

```bash
EXPERIMENT_PRESET=a800_2gpu_overnight \
  scripts/local_4xa800/run_experiment.sh
```

On a server with different mount points:

```bash
SEARCH_R1_ROOT=/path/to/persistent/search-r1 \
SEARCH_R1_RUNTIME_ROOT=/dev/shm/$USER-search-r1 \
CUDA_VISIBLE_DEVICES=0,1 \
scripts/local_4xa800/train_grpo.sh
```
