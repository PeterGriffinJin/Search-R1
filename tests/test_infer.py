"""Exercise the inference loop without model downloads, GPUs, or HTTP requests.

infer.py initializes a model at module scope, so run it with lightweight dependency
stubs. Scripted generations include the input tokens, as model.generate does.
"""

import contextlib
import io
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


INFER_SCRIPT = Path(__file__).resolve().parents[1] / "infer.py"
EOS_TOKEN_ID = 151645


class Token(int):
    def item(self):
        return int(self)


class TokenBatch(list):
    @property
    def shape(self):
        return (len(self), len(self[0]))

    def to(self, device):
        return self


def encode(text, return_tensors=None, **kwargs):
    tokens = [Token(ord(char)) for char in text]
    return TokenBatch([tokens]) if return_tensors == "pt" else tokens


def decode(tokens, **kwargs):
    return "".join(chr(token) for token in tokens if token != EOS_TOKEN_ID)


class InferenceSearchTests(unittest.TestCase):
    def run_inference(self, continuations):
        # Non-EOS turns simulate either a search stop or the generation length
        # limit. The final turn terminates normally with EOS.
        turns = iter([(text, False) for text in continuations]
                     + [("<answer>done</answer>", True)])

        def generate(input_ids, **kwargs):
            try:
                text, terminal = next(turns)
            except StopIteration:
                raise AssertionError("Inference did not stop at EOS") from None
            new_tokens = encode(text)
            if terminal:
                new_tokens.append(Token(EOS_TOKEN_ID))
            return TokenBatch([input_ids[0] + new_tokens])

        tokenizer = SimpleNamespace(
            encode=encode, decode=decode, chat_template=None,
            eos_token_id=EOS_TOKEN_ID,
        )
        model = SimpleNamespace(generate=Mock(side_effect=generate))
        post = Mock(return_value=SimpleNamespace(json=lambda: {
            "result": [[{"document": {"contents": "Title\nEvidence"}}]],
        }))
        modules = {
            "transformers": SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
                StoppingCriteria=object, StoppingCriteriaList=list,
            ),
            "torch": SimpleNamespace(
                device=lambda name: name,
                cuda=SimpleNamespace(is_available=lambda: False),
                bfloat16=object(), ones_like=lambda tokens: None,
            ),
            "datasets": SimpleNamespace(load_dataset=Mock()),
            "requests": SimpleNamespace(post=post),
        }
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(INFER_SCRIPT), run_name="__main__")

        self.assertEqual(model.generate.call_count, len(continuations) + 1)
        return [call.kwargs["json"]["queries"][0] for call in post.call_args_list]

    def test_no_search_in_current_turn_does_not_reuse_history(self):
        queries = self.run_inference([
            "<think>Need evidence.</think><search>old query</search>",
            "<think>Still reasoning when the generation length limit is reached",
        ])
        self.assertEqual(queries, ["old query"])

    def test_new_search_in_current_turn_is_executed(self):
        queries = self.run_inference([
            "<search>old query</search>",
            "<think>Need another fact.</think><search>new query</search>",
        ])
        self.assertEqual(queries, ["old query", "new query"])

    def test_no_search_does_not_execute_prompt_example(self):
        queries = self.run_inference([
            "<think>Still reasoning without requesting any search",
        ])
        self.assertEqual(queries, [])

    def test_incomplete_search_does_not_reuse_history(self):
        queries = self.run_inference([
            "<search>old query</search>",
            "<think>Need another fact.</think><search>unfinished query",
        ])
        self.assertEqual(queries, ["old query"])


if __name__ == "__main__":
    unittest.main()
