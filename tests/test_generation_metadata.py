"""Regression coverage for generation controls across the search loop."""

import pytest
import torch

from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
from verl import DataProto


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"

    def __call__(self, texts, **kwargs):
        tokens = [[ord(char) + 2 for char in text] for text in texts]
        width = max(map(len, tokens))
        return {
            "input_ids": torch.tensor(
                [row + [0] * (width - len(row)) for row in tokens], dtype=torch.long
            )
        }

    def batch_decode(self, tokens, **kwargs):
        return [
            "".join(chr(token - 2) for token in row.tolist() if token > 1)
            for row in tokens
        ]


@pytest.mark.parametrize(
    "num_gpus,do_sample,max_turns",
    [
        (1, False, 2),
        (2, False, 2),
        (4, False, 2),
        (2, True, 2),
        (1, False, 0),
        (4, False, 0),
    ],
)
def test_search_loop_preserves_generation_controls(num_gpus, do_sample, max_turns):
    tokenizer = CharacterTokenizer()
    calls = []
    queries = []
    controls = {
        "do_sample": do_sample,
        "validate": not do_sample,
        "recompute_log_prob": do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    class ScriptedRollout:
        def generate_sequences(self, prompts):
            call_index = len(calls)
            batch_size = prompts.batch["input_ids"].shape[0]
            calls.append((batch_size, prompts.meta_info.copy()))
            # Workers may annotate the dispatched batch, without changing the
            # original generation controls used by subsequent search turns.
            prompts.meta_info["worker_annotation"] = True
            if max_turns == 0:
                texts = ["<answer>done</answer>"] * batch_size
            elif call_index == 0:
                texts = ["<answer>done</answer>", "<search>first hop</search>"]
                texts += [texts[0]] * (batch_size - len(texts))
            elif call_index == 1:
                texts = ["<search>second hop</search>"] * batch_size
            else:
                texts = ["<answer>done</answer>"] * batch_size
            return DataProto.from_dict({"responses": tokenizer(texts)["input_ids"]})

    manager = LLMGenerationManager(
        tokenizer,
        ScriptedRollout(),
        GenerationConfig(
            max_turns=max_turns,
            max_start_length=8,
            max_prompt_length=512,
            max_response_length=128,
            max_obs_length=128,
            num_gpus=num_gpus,
        ),
        is_validation=not do_sample,
    )

    def search(search_queries):
        queries.extend(search_queries)
        return ["retrieved evidence"] * len(search_queries)

    manager.batch_search = search
    input_ids = tokenizer(["A", "B"])["input_ids"]
    gen_batch = DataProto.from_dict(
        {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": torch.zeros_like(input_ids),
        },
        meta_info=controls.copy(),
    )

    output = manager.run_llm_loop(gen_batch, input_ids.clone())

    expected_sizes = [max(2, num_gpus)]
    if max_turns:
        expected_sizes += [max(1, num_gpus)] * 2
    assert [size for size, _ in calls] == expected_sizes
    assert [meta for _, meta in calls] == [controls] * len(calls)
    assert gen_batch.meta_info == controls
    assert output.batch.batch_size == torch.Size([2])
    assert output.meta_info["active_mask"] == [False, False]
    assert queries == (["first hop", "second hop"] if max_turns else [])
    assert all(
        text.endswith("<answer>done</answer>")
        for text in tokenizer.batch_decode(output.batch["responses"])
    )
