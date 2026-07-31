# Docker deployment

The Docker deployment uses separate training and BM25 retrieval images. This
matches the native Python 3.9/3.10 split and allows one retriever to serve
multiple sequential experiments.

## Host requirements

- NVIDIA driver compatible with CUDA 12.1
- NVIDIA Container Toolkit
- Docker Compose v2
- at least two 40GB GPUs for Qwen2.5-3B
- persistent assets downloaded under `SEARCH_R1_ROOT_HOST`
- at least 50GB free under `SEARCH_R1_RUNTIME_ROOT_HOST`

Do not build on a host where Docker's data root is full.

## Configure

```bash
cp docker/.env.example docker/.env
```

Edit `docker/.env`, then create the host runtime directory:

```bash
mkdir -p /dev/shm/$USER-search-r1
```

The SwanLab key should be stored at:

```text
$SEARCH_R1_ROOT_HOST/.secrets/swanlab_api_key
```

with mode `0600`.

## Build

```bash
docker compose --env-file docker/.env \
  -f docker/docker-compose.yml \
  -f docker/compose.2gpu.yml \
  build
```

The training image compiles FlashAttention and can take 15–30 minutes.

## Run

Start BM25:

```bash
docker compose --env-file docker/.env \
  -f docker/docker-compose.yml up -d retriever
```

Run the two-GPU overnight preset:

```bash
docker compose --env-file docker/.env \
  -f docker/docker-compose.yml \
  -f docker/compose.2gpu.yml \
  --profile train run --rm train
```

Run the four-GPU smoke preset by replacing the override with
`docker/compose.4gpu.yml`.

On shared hosts, use an isolated project name:

```bash
docker compose -p search-r1-$USER ...
```

GPU reservations prevent the training container from seeing undeclared cards.
Docker does not prevent another host process from using the same physical GPU;
check `nvidia-smi` before starting.
