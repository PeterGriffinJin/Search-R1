# Reproduction report: A800, BM25, Qwen2.5-3B

## Scope

This report records the verified path from a clean Search-R1 checkout to local
BM25 retrieval and GRPO training. Dense E5/FAISS indexes were deliberately not
downloaded.

Test host:

- 4 × NVIDIA A800-SXM4-40GB
- Linux 6.8
- CUDA-compatible driver 580.95.05
- OpenJDK 21.0.9
- persistent NFS under `/mnt`
- 197 GB host `/dev/shm`

## Assets

| Asset | Source | Observed size |
| --- | --- | ---: |
| Qwen2.5-3B | `Qwen/Qwen2.5-3B` | 5.8 GB |
| BM25 index | `PeterJinGo/wiki-18-bm25-index` | about 2.3 GB |
| train/test parquet | `PeterJinGo/nq_hotpotqa_train` | 426 MB |
| compressed corpus | `PeterJinGo/wiki-18-corpus` | 5.1 GB |
| extracted corpus | same | 14.4 GB |

The BM25 index contains Lucene document IDs but no raw passage bodies.
Search-R1 therefore also needs the corpus. Despite its filename,
`wiki-18.jsonl.gz` expands to a tar archive containing one JSONL member.

## Verified software

Training:

- Python 3.9
- PyTorch 2.4.0 + cu121
- vLLM 0.6.3
- Transformers 4.47.1
- Ray 2.51.2
- xFormers 0.0.27.post2
- FlashAttention 2
- SwanLab 0.9.2

Retrieval:

- Python 3.10
- PyTorch 2.4.0 CPU
- Pyserini 0.44.0
- Pyjnius 1.7.0
- Java 21
- FastAPI and Uvicorn

Exact constraints are stored under `constraints/`.

## GPU results

### Four GPUs

One end-to-end step completed successfully with:

- global train batch 16;
- four GRPO agents;
- response length 384;
- vLLM memory utilization 0.55;
- FSDP parameter, gradient, and optimizer CPU offload.

Observed vLLM initialization peaked near 26.2 GB reserved memory per GPU.
The complete one-step run, including startup, checkpoint, and final validation,
took 189 seconds. The actor checkpoint was 13 GB.

A corrected five-update online smoke run completed successfully and saved
`global_step_5`. SwanLab:
https://swanlab.cn/@nileonx/search-r1/runs/cegy07pb

### Two GPUs

The two-GPU configuration completed generation, BM25 calls, backward pass,
SwanLab logging, validation, and checkpoint creation.

Observed step-1 metrics:

- mean reward: 0.078;
- environment finish ratio: 0.375;
- valid-action ratio: 0.526;
- mean response length: 693.5 tokens;
- generation: 57.842 seconds;
- actor update: 21.078 seconds;
- checkpoint: 33.668 seconds;
- total step with checkpoint: 118.625 seconds;
- actor MFU: 0.161.

vLLM initialization peaked near 25.0 GB reserved memory per GPU. Two GPUs are
the recommended minimum for Qwen2.5-3B.

### One GPU

Two configurations were tested:

1. batch 8, vLLM utilization 0.35, response length 256;
2. batch 4, vLLM utilization 0.15, response length 192.

Both failed at actor-to-vLLM weight synchronization. The process used about
39.4 GiB and failed on the next small allocation. Lowering KV-cache allocation
did not solve simultaneous actor/vLLM weight residency. The launcher rejects
one GPU by default.

## Failure log and fixes

### Root filesystem was full

Package caches and model downloads could not use `/root`. Persistent assets
were moved to `SEARCH_R1_ROOT`; high-churn runtime data moved to
`SEARCH_R1_RUNTIME_ROOT`.

### Conda creation on NFS failed

Parallel package extraction produced NFS I/O errors. Environments are created
on local `/dev/shm` or SSD and can be reconstructed from scripts.

### vLLM resolver backtracking

Installing vLLM without constraints caused NumPy/OpenCV resolver churn.
`numpy<2` and `opencv-python-headless<4.11` are installed before vLLM.

### Pyserini imports failed

Pyserini's import path eagerly loads encoder integrations. The retriever
environment therefore includes CPU Torch, Transformers, Pillow, ONNX Runtime,
OpenAI, Tiktoken, and Flask even though BM25 itself does not use them.
`OPENAI_API_KEY` is set to a non-secret local placeholder to avoid import-time
client validation.

### Java detection failed

Pyjnius could not discover a compiler on a JRE-only installation.
Setting `JAVA_HOME` to Java 21 made the existing JVM load correctly.

### Proxy downloads reset TLS

Three simultaneous Hugging Face clients overwhelmed the local proxy. Downloads
now use bounded worker counts and a configurable `HF_ENDPOINT`.

### Corpus extraction appeared corrupt

Plain gzip extraction produced tar header bytes rather than JSON. The correct
operation is `tar -xzOf wiki-18.jsonl.gz`.

### NFS checkpoint writes stalled

Concurrent FSDP workers writing directly to NFS entered uninterruptible I/O.
Checkpoints now write to local runtime storage and only the latest actor is
copied sequentially to persistent storage.

### `total_training_steps` was off by one

The upstream loop incremented the global step and returned on `>=`, so a
five-step request performed only four optimizer updates and skipped the step-5
checkpoint. The termination check now uses `>` and logs final validation at
the requested step. A repeated five-step run verified the fix.

## Experiment presets

`a800_4gpu_smoke` validates the full online path in five steps.

`a800_2gpu_overnight` runs 300 steps with:

- train batch 16;
- four agents;
- validation every 25 steps;
- checkpoints at steps 100, 200, and 300;
- final actor archive to persistent storage;
- SwanLab online logging.

Four-GPU smoke:
https://swanlab.cn/@nileonx/search-r1/runs/cegy07pb

The 300-step run URL and final metric summary are appended after completion.

Active two-GPU 300-step run:
https://swanlab.cn/@nileonx/search-r1/runs/7lxgmkuc
