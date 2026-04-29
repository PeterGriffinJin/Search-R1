# Stage41 Qwen3.5-9B RL With dev veRL and vLLM 0.19.1

This note records the local-scratch environment, dev veRL integration, model
deployment, Search-R1 reward/search loop wiring, evaluation setup, and the eval
driver write-path fix used for the Qwen/Qwen3.5-9B-Base stage41 GRPO run.

## Scope

- Base model: `Qwen/Qwen3.5-9B-Base`.
- RL strategy: stage41 Search-R1 GRPO, 50 training steps.
- Training source tree: `/mnt/local-scratch/fhwu/src/verl-vllm019`.
- vLLM environment: `/mnt/local-scratch/fhwu/envs/vllm`.
- Retriever: 2018 Wikipedia dump with E5, `topk=3`.
- Final merged checkpoint:
  `/mnt/local-scratch/fhwu/search-r1/merged_models/stage41-grpo-qwen35-9b-base-devverl-vllm019-50step-gpuretriever-paperref-gpu03-nooffload-global_step_50`.

## Local-Scratch Environment

All heavy Python state, package caches, model cache, Ray temp files, vLLM cache,
Torch/Triton compile cache, and temporary files were kept under
`/mnt/local-scratch/fhwu`.

The active Python environment was reused from the existing vLLM install:

```text
Python: /mnt/local-scratch/fhwu/envs/vllm/bin/python
veRL source: /mnt/local-scratch/fhwu/src/verl-vllm019
vLLM: 0.19.1
Torch: 2.10.0+cu128
Ray: 2.55.1
Transformers: 5.5.4
FlashAttention: 2.8.3
```

The run intentionally used the local dev veRL tree rather than the vendored
Search-R1 veRL adapter, because the old adapter was not compatible with the
current vLLM 0.19.1 environment.

## Flash Attention

`flash-attn==2.8.3` was built and installed into
`/mnt/local-scratch/fhwu/envs/vllm` against CUDA 12.8 and Torch 2.10.0.

The CUDA branch of
`/mnt/local-scratch/fhwu/src/verl-vllm019/verl/utils/attention_utils.py` imports
`flash_attn.bert_padding` directly. There is no PyTorch fallback for the CUDA
path. This is intentional: if FlashAttention or `bert_padding` is missing, the
training stack should fail fast instead of silently entering a slower or
semantically different path.

The NPU branch still uses `verl.utils.npu_flash_attn_utils`.

## dev veRL Changes

The dev veRL tree was extended with a Search-R1 XML agent loop:

- `verl/experimental/agent_loop/search_r1_xml_agent_loop.py` implements the
  `<search>...</search>` and `<answer>...</answer>` loop.
- `verl/experimental/agent_loop/__init__.py` registers
  `SearchR1XMLAgentLoop`.
- `verl/utils/reward_score/search_r1_stage41.py` wraps the Search-R1 EM reward
  for dev veRL custom reward loading.
- `examples/search_r1_stage41/search_r1_xml_agent_loop.yaml` configures the
  retriever URL, `topk`, max turns, and observation budget.
- `examples/search_r1_stage41/run_qwen35_9b_base_stage41_vllm.sh` launches
  Qwen3.5-9B stage41 GRPO with dev veRL async vLLM rollout.
- `tests/experimental/agent_loop/test_search_r1_xml_agent_loop.py` covers XML
  action parsing, empty decoded generations, reward signature compatibility,
  and launcher invariants.

A Qwen3.5/Torch 2.10 nested tensor issue was fixed in
`verl/workers/utils/padding.py`: `position_ids` now uses
`tu.nested_tensor_from_tensor_list(..., ragged_idx=position_ids_list[0].dim())`
instead of `torch.nested.as_nested_tensor(..., layout=torch.jagged)`, avoiding a
wrong ragged dimension inference for Qwen3.5 4D position IDs.

The dev veRL rollout default was also adjusted so
`max_num_batched_tokens` is `null` by default. This lets vLLM use its own
default scheduler behavior unless an experiment explicitly sets the value.

## Stage41 Training Configuration

The completed 50-step run used this experiment name:

```text
stage41-grpo-qwen35-9b-base-devverl-vllm019-50step-gpuretriever-paperref-gpu03-nooffload
```

Core training settings:

```text
algorithm.adv_estimator=grpo
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.tensor_model_parallel_size=2
actor_rollout_ref.rollout.gpu_memory_utilization=0.30
actor_rollout_ref.actor.strategy=fsdp2
actor_rollout_ref.actor.fsdp_config.fsdp_size=2
actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.rollout.n=5
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285
actor_rollout_ref.actor.use_kl_loss=true
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.ppo_mini_batch_size=8
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
data.train_batch_size=16
data.val_batch_size=16
data.max_prompt_length=32256
data.max_response_length=500
trainer.total_training_steps=50
trainer.save_freq=10
trainer.test_freq=10
```

The run did not enable actor/ref CPU offload:

```text
ACTOR_PARAM_OFFLOAD=false
ACTOR_OPTIMIZER_OFFLOAD=false
ACTOR_OFFLOAD_POLICY=false
REF_PARAM_OFFLOAD=false
REF_OFFLOAD_POLICY=false
```

Reason: an earlier offload attempt failed during FSDP-to-vLLM weight update with
a CPU/CUDA storage mismatch. The successful 50-step run therefore used nooffload.

The final training log is:

```text
/mnt/local-scratch/fhwu/search-r1/logs/stage41-grpo-qwen35-9b-base-devverl-vllm019-50step-gpuretriever-paperref-gpu03-nooffload.log
```

Final step highlights:

```text
training/global_step=50
val-core/nq/acc/mean@1=0.0
val-core/hotpotqa/acc/mean@1=0.010390624999999999
val-aux/hotpotqa/reward/mean@1=0.010390624690141218
critic/score/mean=0.002624999964609742
response_length/mean=446.4624938964844
response_length/clip_ratio=0.7250000238418579
num_turns/mean=2.7
timing_s/agent_loop/tool_calls/mean=26.442905174090992
timing_s/agent_loop/tool_calls/max=112.50113329780288
timing_s/gen=121.19337344402447
timing_s/ref=45.857828529085964
timing_s/update_actor=89.57199607905932
timing_s/save_checkpoint=21.22886428690981
timing_s/testing=796.3331013799179
timing_s/step=299.8700472860364
```

The final `EngineCore died unexpectedly` message occurred after final validation
and process shutdown, and was treated as shutdown noise rather than a training
failure.

## GPU Retriever

The retriever process used the Search-R1 retrieval server with the Wiki18/E5
index:

```text
index_path=/mnt/local-scratch/fhwu/search-r1/wiki18-e5/e5_Flat.index
corpus_path=/mnt/local-scratch/fhwu/search-r1/wiki18-e5/wiki-18.jsonl
retriever_model=intfloat/e5-base-v2
topk=3
endpoint=http://127.0.0.1:8000/retrieve
```

For this run the retriever was GPU-backed. Earlier CPU/GPU retriever probes
showed that the retrieval loop remains a major latency contributor under
concurrent Search-R1 rollouts, and sharing GPU 1 with training did not produce a
clear end-to-end speedup.

## vLLM Serving Configuration

The final merged model was served through the OpenAI-compatible vLLM API:

```text
port=8080
tensor_parallel_size=2
dtype=bfloat16
gpu_memory_utilization=0.30
max_model_len=262144
language_model_only=true
enable_auto_tool_choice=true
tool_call_parser=qwen3_xml
```

No CLI `--max_num_batched_tokens` override was used for the evaluation server.
The explicit `--max-model-len 262144` matched the Qwen3.5 model-declared maximum
context length; shorter context values were not used.

## Function Call Evaluation

The control evaluation exposes Search-R1 retrieval as an OpenAI-style function
call:

1. Call `/chat/completions` with the `retrieve` tool and `tool_choice="auto"`.
2. Parse returned `tool_calls`.
3. Execute each `retrieve` tool call against the E5/Wiki18 retriever.
4. Append `role="tool"` messages with retrieved passages.
5. Ask the model for a final concise answer with `tool_choice="none"`.

The evaluation driver intentionally does not set `temperature` or `max_tokens`
in the OpenAI-compatible chat payload. Tests assert these fields are absent.

Full-eval input:

```text
eval_runs/stage44_full_eval/full_manifest_nq3610_hotpot7405.jsonl
```

Full-eval output directory:

```text
eval_runs/stage75_qwen35_9b_stage41_step50_toolcall_full
```

Final full-eval integrity check:

```text
rows=11015
unique_keys=11015
summary_total=11015
```

Final full-eval metrics:

```text
strict_em=2746 / 11015
strict_em_rate=0.24929641398093508
relaxed_em=4532 / 11015
relaxed_em_rate=0.41143894689060373
avg_token_f1=0.37629497412619156
avg_num_searches=1.4699046754425784
avg_latency_sec=17.60046890603722
```

By dataset:

```text
NQ:
  total=3610
  strict_em=897
  strict_em_rate=0.2484764542936288
  relaxed_em=1766
  relaxed_em_rate=0.489196675900277
  avg_num_searches=1.1409972299168976

HotpotQA:
  total=7405
  strict_em=1849
  strict_em_rate=0.24969615124915598
  relaxed_em=2766
  relaxed_em_rate=0.37353139770425386
  avg_num_searches=1.6302498311951383
```

Failure type counts:

```text
correct=2746
evidence_use_error=3923
retrieval_miss=4328
no_search=11
no_answer_tag=6
request_error=1
```

Search behavior:

```text
multi_search=3557
multi_search_rate=0.3229232864275987
closed_after_multi_search=3556
closed_after_multi_search_rate=0.9997188642114141
strict_after_multi_search=899
strict_after_multi_search_rate=0.2527410739387124
```

Failure taxonomy files:

```text
eval_runs/stage75_qwen35_9b_stage41_step50_toolcall_full/failure_taxonomy.csv
eval_runs/stage75_qwen35_9b_stage41_step50_toolcall_full/failure_taxonomy.json
eval_runs/stage75_qwen35_9b_stage41_step50_toolcall_full/failure_taxonomy.md
```

Top taxonomy subtype counts:

```text
hotpot_single_query_topk_miss=2209
near_alias_or_granularity=1821
numeric_or_date_specificity=1408
partial_relation_wrong_answer=766
retrieval_other=693
direct_topk_miss=653
comparison_or_multihop_logic_error=487
```

## Eval Driver Write Path

The slow write-path bug affected both XML-style and Function Call eval drivers.
The old behavior rewrote `results.jsonl` and `results.csv` from the beginning
after every completed sample. With a large `results.jsonl`, this caused the
driver to spend increasing time in disk I/O and scheduling, leaving the GPU
server idle between requests.

The patched behavior is:

- During a run, append one JSONL row per completed sample.
- During a run, update only the small `summary.json`.
- On resume, rebuild only `summary.json` from existing results.
- At normal completion, rewrite the ordered/deduplicated `results.jsonl` and
  `results.csv` once.

Changed files:

```text
scripts/stage2_eval_small.py
scripts/stage3_analyze_failure_taxonomy.py
scripts/stage72_eval_tool_calls.py
tests/test_stage2_eval_small.py
tests/test_stage3_failure_taxonomy.py
tests/test_stage72_eval_tool_calls.py
```

The failure taxonomy reader was also hardened to iterate file lines rather than
using `Path.read_text().splitlines()`. Some model outputs contain Unicode line
separator characters inside JSON strings; `splitlines()` can split those strings
and create invalid pseudo-lines even when the JSONL file is valid.

The currently running full eval was started before this patch, so it still uses
the old in-memory code. It should be allowed to finish naturally because its
`results.jsonl` can be temporarily invalid while the old driver rewrites it.
Future full eval runs and resumes use the append-only path.

## Verification Commands

Search-R1 eval driver tests:

```bash
/mnt/local-scratch/fhwu/envs/vllm/bin/python -c "import sys; sys.path=[p for p in sys.path if p != '/mnt/local-scratch/fhwu/src/verl-vllm019']; import pytest; raise SystemExit(pytest.main(['tests/test_stage2_eval_small.py','tests/test_stage72_eval_tool_calls.py','-q']))"
```

dev veRL agent loop tests:

```bash
PYTHONPATH=/mnt/local-scratch/fhwu/src/verl-vllm019 \
/mnt/local-scratch/fhwu/envs/vllm/bin/python -m pytest \
  tests/experimental/agent_loop/test_search_r1_xml_agent_loop.py -q
```
