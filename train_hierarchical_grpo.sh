#!/bin/bash
# Hierarchical Thinker-Searcher Training Script
# Thinker: Qwen2.5-0.5B (trained with RL, max 2 iterations)
# Searcher: Qwen2.5-1.5B (frozen, max 3 turns per search)

export CUDA_VISIBLE_DEVICES=0
export DATA_DIR=data/hotpotqa_only

WAND_PROJECT="Search-R1-Hierarchical"

# Thinker model (smaller, will be trained)
export THINKER_MODEL='Qwen/Qwen2.5-0.5B-Instruct'
# Searcher model (larger, frozen)
export SEARCHER_MODEL='Qwen/Qwen2.5-1.5B-Instruct'

export EXPERIMENT_NAME=hotpotqa-hierarchical-thinker0.5b-searcher1.5b

export VLLM_ATTENTION_BACKEND=XFORMERS

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_hierarchical \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=256 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=300 \
    data.max_start_length=1024 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$THINKER_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=3 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=3 \
    trainer.total_training_steps=35 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    hierarchical.enabled=true \
    hierarchical.thinker_max_turns=2 \
    hierarchical.searcher_max_turns=1 \
    hierarchical.thinker_max_response_length=300 \
    hierarchical.searcher_max_response_length=400 \
    hierarchical.searcher_max_obs_length=400 \
    hierarchical.searcher_model_path=$SEARCHER_MODEL \
    hierarchical.searcher_gpu_memory=0.25 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log

