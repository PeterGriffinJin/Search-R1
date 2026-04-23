"""
Batch benchmark evaluator for Search-R1 checkpoints.

Usage example (same level as `train_grpo.sh`):
python3 benchmark_validate.py \
  actor_rollout_ref.model.path=verl_checkpoints/your_exp/actor/global_step_1000 \
  data.val_files=data/nq_search/test.parquet \
  data.val_batch_size=256 \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  do_search=true \
  max_turns=2 \
  retriever.url=http://127.0.0.1:8000/retrieve \
  retriever.topk=3
"""

import json
from pprint import pprint
from collections import defaultdict

import hydra
import ray
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig


def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    raise NotImplementedError


class RewardManager:
    """Function RM aligned with main_ppo.py."""

    def __init__(self, tokenizer, num_examine=1, format_score=0.0):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score

    def __call__(self, data):
        import torch

        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            score = compute_score_fn(solution_str=sequences_str,
                                     ground_truth=ground_truth,
                                     format_score=self.format_score)

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


_RAY_WORKER_ENV_VARS = {
    'TOKENIZERS_PARALLELISM': 'true',
    'NCCL_DEBUG': 'WARN',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
}


def validate_with_progress(trainer: RayPPOTrainer):
    """Validation flow aligned with RayPPOTrainer._validate(), with tqdm progress and per-step logs."""
    import numpy as np
    import torch

    reward_tensor_lst = []
    data_source_lst = []
    running_scores = defaultdict(list)

    gen_config = GenerationConfig(
        max_turns=trainer.config.max_turns,
        max_start_length=trainer.config.data.max_start_length,
        max_prompt_length=trainer.config.data.max_prompt_length,
        max_response_length=trainer.config.data.max_response_length,
        max_obs_length=trainer.config.data.max_obs_length,
        num_gpus=trainer.config.trainer.n_gpus_per_node * trainer.config.trainer.nnodes,
        no_think_rl=trainer.config.algorithm.no_think_rl,
        search_url=trainer.config.retriever.url,
        topk=trainer.config.retriever.topk,
    )
    generation_manager = LLMGenerationManager(
        tokenizer=trainer.tokenizer,
        actor_rollout_wg=trainer.actor_rollout_wg,
        config=gen_config,
        is_validation=True,
    )

    progress = tqdm(enumerate(trainer.val_dataloader, start=1),
                    total=len(trainer.val_dataloader),
                    desc='Benchmark validating',
                    unit='batch')

    for batch_idx, batch_dict in progress:
        test_batch = DataProto.from_single_dict(batch_dict)

        if trainer.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
            raise RuntimeError('Validation with model-based RM is not supported in this script.')

        if not trainer.config.do_search:
            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': trainer.tokenizer.eos_token_id,
                'pad_token_id': trainer.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, trainer.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = trainer.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            test_batch = test_batch.union(test_output_gen_batch)
            reward_tensor = trainer.val_reward_fn(test_batch)
        else:
            test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': trainer.tokenizer.eos_token_id,
                'pad_token_id': trainer.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
            final_gen_batch_output = generation_manager.run_llm_loop(
                gen_batch=test_gen_batch,
                initial_input_ids=first_input_ids,
            )
            test_batch = test_batch.union(final_gen_batch_output)
            for key in test_batch.batch.keys():
                test_batch.batch[key] = test_batch.batch[key].long()
            reward_tensor = trainer.val_reward_fn(test_batch)

        reward_tensor_lst.append(reward_tensor)
        cur_sources = test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0])
        data_source_lst.append(cur_sources)

        per_sample_scores = reward_tensor.sum(-1).detach().cpu().tolist()
        for src, score in zip(cur_sources, per_sample_scores):
            running_scores[src].append(score)

        batch_mean = float(np.mean(per_sample_scores)) if per_sample_scores else 0.0
        overall_mean = float(np.mean([s for arr in running_scores.values() for s in arr])) \
            if running_scores else 0.0
        progress.set_postfix(batch_mean=f'{batch_mean:.4f}', running_mean=f'{overall_mean:.4f}')
        print(f'[validate] batch={batch_idx}/{len(trainer.val_dataloader)} '
              f'batch_mean={batch_mean:.4f} running_mean={overall_mean:.4f}')

    reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()
    data_sources = np.concatenate(data_source_lst, axis=0)
    data_source_reward = {}
    for i in range(reward_tensor.shape[0]):
        data_source = data_sources[i]
        if data_source not in data_source_reward:
            data_source_reward[data_source] = []
        data_source_reward[data_source].append(reward_tensor[i].item())

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f'val/test_score/{data_source}'] = float(np.mean(rewards))

    return metric_dict


@hydra.main(config_path='verl/trainer/config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': _RAY_WORKER_ENV_VARS})

    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Benchmark only uses validation; still keep trainer initialization robust.
    with open_dict(config):
        if not config.data.get('train_files'):
            config.data.train_files = config.data.val_files
        if not config.data.get('train_data_num'):
            config.data.train_data_num = config.data.val_data_num
        config.trainer.val_before_train = False
        config.trainer.val_only = True

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)

    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)

    trainer.init_workers()
    try:
        metrics = validate_with_progress(trainer)
        print('Benchmark metrics:')
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    finally:
        trainer.shutdown()


if __name__ == '__main__':
    main()
