#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
N_GPUS="${#GPU_IDS[@]}"
export SEARCH_R1_GPU_COUNT="$N_GPUS"
export SEARCH_R1_GIT_COMMIT="$(
  git -C "$SEARCH_R1_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown
)"

case "$N_GPUS" in
  1)
    DEFAULT_TRAIN_BATCH_SIZE=4
    DEFAULT_ACTOR_MICRO_BATCH_SIZE=4
    DEFAULT_LOG_PROB_MICRO_BATCH_SIZE=8
    DEFAULT_GPU_MEMORY_UTILIZATION=0.15
    DEFAULT_MAX_PROMPT_LENGTH=1024
    DEFAULT_MAX_RESPONSE_LENGTH=192
    DEFAULT_MAX_START_LENGTH=512
    DEFAULT_MAX_OBS_LENGTH=128
    ;;
  2)
    DEFAULT_TRAIN_BATCH_SIZE=16
    DEFAULT_ACTOR_MICRO_BATCH_SIZE=16
    DEFAULT_LOG_PROB_MICRO_BATCH_SIZE=32
    DEFAULT_GPU_MEMORY_UTILIZATION=0.45
    DEFAULT_MAX_PROMPT_LENGTH=2048
    DEFAULT_MAX_RESPONSE_LENGTH=384
    DEFAULT_MAX_START_LENGTH=1024
    DEFAULT_MAX_OBS_LENGTH=256
    ;;
  3|4)
    DEFAULT_TRAIN_BATCH_SIZE=16
    DEFAULT_ACTOR_MICRO_BATCH_SIZE=32
    DEFAULT_LOG_PROB_MICRO_BATCH_SIZE=64
    DEFAULT_GPU_MEMORY_UTILIZATION=0.55
    DEFAULT_MAX_PROMPT_LENGTH=2048
    DEFAULT_MAX_RESPONSE_LENGTH=384
    DEFAULT_MAX_START_LENGTH=1024
    DEFAULT_MAX_OBS_LENGTH=256
    ;;
  *)
    DEFAULT_TRAIN_BATCH_SIZE=32
    DEFAULT_ACTOR_MICRO_BATCH_SIZE=32
    DEFAULT_LOG_PROB_MICRO_BATCH_SIZE=128
    DEFAULT_GPU_MEMORY_UTILIZATION=0.60
    DEFAULT_MAX_PROMPT_LENGTH=2048
    DEFAULT_MAX_RESPONSE_LENGTH=384
    DEFAULT_MAX_START_LENGTH=1024
    DEFAULT_MAX_OBS_LENGTH=256
    ;;
esac

TOTAL_STEPS="${TOTAL_STEPS:-60}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$DEFAULT_TRAIN_BATCH_SIZE}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
N_AGENT="${N_AGENT:-4}"
TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-2048}"
VAL_DATA_NUM="${VAL_DATA_NUM:-64}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-20}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$((TRAIN_BATCH_SIZE * N_AGENT))}"
ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-$DEFAULT_ACTOR_MICRO_BATCH_SIZE}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-$DEFAULT_LOG_PROB_MICRO_BATCH_SIZE}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$DEFAULT_GPU_MEMORY_UTILIZATION}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-$DEFAULT_MAX_PROMPT_LENGTH}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$DEFAULT_MAX_RESPONSE_LENGTH}"
MAX_START_LENGTH="${MAX_START_LENGTH:-$DEFAULT_MAX_START_LENGTH}"
MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-$DEFAULT_MAX_OBS_LENGTH}"
ENABLE_SWANLAB="${ENABLE_SWANLAB:-1}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"

if (( N_GPUS == 1 )) && [[ "${ALLOW_SINGLE_GPU:-0}" != "1" ]]; then
  echo "Qwen2.5-3B Search-R1 OOMs on one 40GB GPU during actor/vLLM weight sync." >&2
  echo "Use at least two GPUs, or set ALLOW_SINGLE_GPU=1 only for code-level experiments." >&2
  exit 1
fi
if (( TRAIN_BATCH_SIZE % N_GPUS != 0 )); then
  echo "TRAIN_BATCH_SIZE must be divisible by the GPU count ($N_GPUS)." >&2
  exit 1
fi
if (( VAL_DATA_NUM < VAL_BATCH_SIZE )); then
  echo "VAL_DATA_NUM must be at least VAL_BATCH_SIZE." >&2
  exit 1
fi
if [[ "$ENABLE_SWANLAB" == "1" && "$SWANLAB_MODE" == "online" && -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "Export SWANLAB_API_KEY before enabling SwanLab." >&2
  exit 1
fi

HOST_TAG="$(hostname -s | tr -c '[:alnum:].-' '-')"
RUN_TAG="$(date +%Y%m%d-%H%M%S)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2.5-3b-bm25-${N_GPUS}gpu-${HOST_TAG}-${RUN_TAG}}"
LOG_FILE="$SEARCH_R1_LOGS/${EXPERIMENT_NAME}-steps${TOTAL_STEPS}.log"

test -x "$SEARCH_R1_ENV/bin/python"
test -f "$SEARCH_R1_DATA/train.parquet"
test -f "$SEARCH_R1_DATA/test.parquet"
test -f "$SEARCH_R1_MODEL/config.json"
curl --fail --silent --show-error "${RETRIEVER_URL%/retrieve}/health" >/dev/null

mkdir -p "$SEARCH_R1_CHECKPOINTS" "$SEARCH_R1_LOGS" "$RAY_TMPDIR" "$SWANLAB_LOGDIR"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export PYTHONUNBUFFERED=1

if [[ "$ENABLE_SWANLAB" == "1" ]]; then
  LOGGER_CONFIG='trainer.logger=["console","swanlab"]'
else
  LOGGER_CONFIG='trainer.logger=["console"]'
fi

cd "$SEARCH_R1_REPO"
"$SEARCH_R1_ENV/bin/python" -m verl.trainer.main_ppo \
  data.train_files="$SEARCH_R1_DATA/train.parquet" \
  data.val_files="$SEARCH_R1_DATA/test.parquet" \
  data.train_data_num="$TRAIN_DATA_NUM" \
  data.val_data_num="$VAL_DATA_NUM" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.max_start_length="$MAX_START_LENGTH" \
  data.max_obs_length="$MAX_OBS_LENGTH" \
  data.shuffle_train_dataloader=True \
  algorithm.adv_estimator=grpo \
  algorithm.no_think_rl=false \
  actor_rollout_ref.model.path="$SEARCH_R1_MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE:-1e-6}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${LR_WARMUP_RATIO:-0.285}" \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size="$ACTOR_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="$LOG_PROB_MICRO_BATCH_SIZE" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.n_agent="$N_AGENT" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size="$LOG_PROB_MICRO_BATCH_SIZE" \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.state_masking=true \
  "$LOGGER_CONFIG" \
  +trainer.val_only=false \
  +trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.project_name=search-r1 \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.total_epochs="${TOTAL_EPOCHS:-15}" \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.default_local_dir="$SEARCH_R1_CHECKPOINTS/$EXPERIMENT_NAME" \
  max_turns="${MAX_TURNS:-2}" \
  retriever.url="$RETRIEVER_URL" \
  retriever.topk="${RETRIEVER_TOPK:-3}" \
  "$@" \
  2>&1 | tee "$LOG_FILE"

if [[ "${ARCHIVE_AFTER_TRAIN:-1}" == "1" ]]; then
  "$SCRIPT_DIR/archive_checkpoint.sh" "$EXPERIMENT_NAME"
fi
