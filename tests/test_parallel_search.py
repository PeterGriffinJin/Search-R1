"""CPU runtime/wire tests; no model download, GPU, API key or live quota needed."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from uuid import UUID

import pytest
import requests
import torch
import yaml

from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
from search_r1.llm_agent import parallel_search
from verl import DataProto


class Tokenizer:
    pad_token_id = 0
    pad_token = ""

    def __call__(self, texts, **kwargs):
        rows = [[ord(c) + 1 for c in text] for text in texts]
        tokens = torch.zeros((len(rows), max(map(len, rows))), dtype=torch.long)
        for i, row in enumerate(rows):
            tokens[i, :len(row)] = torch.tensor(row)
        return {"input_ids": tokens}

    def batch_decode(self, tokens, **kwargs):
        return ["".join(chr(int(t) - 1) for t in row if t) for row in tokens]


class Actor:
    """Script model output, but run the real tokenizer/manager/DataProto loop."""

    def __init__(self):
        self.turns = {}
        self.observations = []

    def generate_sequences(self, batch):
        prompts = Tokenizer().batch_decode(batch.batch["input_ids"])
        responses = []
        for prompt in prompts:
            label = prompt[0]
            turn = self.turns.get(label, 0)
            if turn:
                assert "<information>Doc 1(Title:" in prompt
                assert "Source: https://example.org/0" in prompt
                assert "Evidence for" in prompt
                self.observations.append(prompt)
            if turn == 0 or (turn == 1 and label == "A"):
                responses.append(f"<search>query {label} turn {turn}</search>")
            else:
                responses.append("<answer>grounded answer</answer>")
            self.turns[label] = turn + 1
        return DataProto.from_dict({"responses": Tokenizer()(responses)["input_ids"]})


def manager(url="parallel", turns=3):
    config = GenerationConfig(max_turns=turns, max_start_length=64,
                              max_prompt_length=4096, max_response_length=512,
                              max_obs_length=2048, num_gpus=1, search_url=url, topk=2)
    return LLMGenerationManager(Tokenizer(), Actor(), config)


def rollout(mgr, labels=("A", "B")):
    ids = Tokenizer()(labels)["input_ids"]
    batch = DataProto.from_dict({"input_ids": ids, "attention_mask": torch.ones_like(ids),
                                 "position_ids": torch.zeros_like(ids)})
    return mgr.run_llm_loop(batch, ids)


@pytest.fixture
def wire(monkeypatch):
    state = SimpleNamespace(records=[], mode="json", failure=None)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.records.append((self.path, dict(self.headers), payload))
            method = payload["method"]
            if method == "notifications/initialized":
                self.send_response(202)
                self.end_headers()
                return
            if method == "initialize":
                result = {"protocolVersion": parallel_search.PROTOCOL_VERSION}
            elif method == "tools/list":
                result = {"tools": [{"name": "web_search"}]}
            else:
                assert method == "tools/call"
                assert payload["params"]["name"] == "web_search"
                if state.failure == "timeout":
                    time.sleep(0.15)
                if state.failure == "http":
                    self.send_response(429)
                    self.send_header("Retry-After", "60")
                    self.end_headers()
                    return
                query = payload["params"]["arguments"]["objective"]
                data = {"results": [{"url": f"https://example.org/{i}", "title": f"Page {i}",
                                      "excerpts": [f"Evidence for {query}"]} for i in range(4)]}
                result = {"structuredContent": data}
                if state.mode == "sse":
                    result = {"content": [{"type": "text", "text": json.dumps(data)}]}
                if state.failure == "tool":
                    result = {"isError": True, "content": [{"type": "text", "text": "quota exceeded"}]}
                if state.failure == "malformed":
                    result = {"structuredContent": {"results": [{"url": "bad", "excerpts": []}]}}
            response = {"jsonrpc": "2.0", "id": payload["id"], "result": result}
            if state.failure == "rpc" and method == "tools/call":
                response = {"jsonrpc": "2.0", "id": payload["id"], "error": {"code": -32000}}
            if state.failure == "id" and method == "tools/call":
                response["id"] = -1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if state.mode == "sse"
                             else "application/json")
            self.send_header("Mcp-Session-Id", "transport-session")
            self.end_headers()
            if state.mode == "sse":
                body = 'data: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n'
                body += "data: " + json.dumps(response) + "\n\n"
            else:
                body = json.dumps(response)
            try:
                self.wfile.write(body.encode())
            except BrokenPipeError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(parallel_search, "MCP_URL", f"http://127.0.0.1:{server.server_port}/mcp")
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def search_calls(wire):
    return [payload["params"]["arguments"] for _, _, payload in wire.records
            if payload["method"] == "tools/call"]


@pytest.mark.parametrize("mode", ["json", "sse"])
def test_runtime_search_consumption_and_conversation_lifetime(wire, monkeypatch, tmp_path, mode):
    wire.mode = mode
    netrc = tmp_path / "netrc"
    netrc.write_text("machine 127.0.0.1 login ambient password secret\n")
    monkeypatch.setenv("NETRC", str(netrc))
    monkeypatch.setenv("PARALLEL_API_KEY", "must-not-be-used")
    mgr = manager()
    result = rollout(mgr)
    calls = search_calls(wire)
    assert len(calls) == 3  # Two searches for A; B finishes before A's last turn.
    assert calls[0]["session_id"] == calls[2]["session_id"]
    assert calls[0]["session_id"] != calls[1]["session_id"]
    for call in calls:
        UUID(call["session_id"])
        assert call["search_queries"] == [call["objective"]]
    for path, headers, payload in wire.records:
        headers = {key.lower(): value for key, value in headers.items()}
        assert path == "/mcp"
        assert headers["user-agent"] == parallel_search.USER_AGENT
        assert "authorization" not in headers
        if payload["method"] != "initialize":
            assert headers["mcp-session-id"] == "transport-session"
            assert headers["mcp-protocol-version"] == parallel_search.PROTOCOL_VERSION
    assert mgr.actor_rollout_wg.observations
    decoded = Tokenizer().batch_decode(result.batch["responses"])
    assert all("<answer>grounded answer</answer>" in row for row in decoded)
    assert all("Doc 3" not in row for row in decoded)  # topk truncation
    assert result.meta_info["valid_search_stats"] == [2, 1]

    previous_ids = {call["session_id"] for call in calls}
    # A new run on the same manager must not reuse identities.
    mgr.actor_rollout_wg = Actor()
    rollout(mgr)
    assert previous_ids.isdisjoint({call["session_id"] for call in search_calls(wire)[3:]})


def test_interleaved_rollouts_do_not_share_conversation_ids(wire):
    mgr = manager()
    # A shared manager/provider has no mutable conversation context.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(rollout, mgr, ("A",)),
                   executor.submit(rollout, mgr, ("B",))]
        for future in futures:
            future.result()
    calls = search_calls(wire)
    a_ids = {call["session_id"] for call in calls if "query A" in call["objective"]}
    b_ids = {call["session_id"] for call in calls if "query B" in call["objective"]}
    assert len(a_ids) == len(b_ids) == 1
    assert a_ids.isdisjoint(b_ids)


def test_existing_default_and_explicit_url_are_preserved(wire, monkeypatch):
    with open("verl/trainer/config/ppo_trainer.yaml") as file:
        default = yaml.safe_load(file)["retriever"]["url"]
    assert default == "http://127.0.0.1:8000/retrieve"
    captured = []

    def retrieve(url, json):
        captured.append((url, json))
        return SimpleNamespace(json=lambda: {"result": [[{"document": {"contents": "Local\nSaved corpus"}}]]})

    monkeypatch.setattr(requests, "post", retrieve)
    for url in [default, "https://custom.example/retrieve?engine=private"]:
        observations, *_ = manager(url).execute_predictions(["<search>saved choice</search>"], "", [True])
        assert "Saved corpus" in observations[0]
        assert captured[-1] == (url, {"queries": ["saved choice"], "topk": 2, "return_scores": True})
    assert not wire.records


def test_disabled_inactive_and_answer_only_do_not_contact_provider(wire):
    mgr = manager()
    obs, *_ = mgr.execute_predictions(["<search>disabled</search>"], "", [True], do_search=False)
    assert obs == ["\n\n<information></information>\n\n"]
    mgr.execute_predictions(["<search>inactive</search>"], "", [False])
    mgr.execute_predictions(["<answer>done</answer>"], "", [True])
    assert not wire.records


@pytest.mark.parametrize("failure, error", [("http", requests.HTTPError),
                                            ("tool", parallel_search.ParallelSearchError),
                                            ("rpc", parallel_search.ParallelSearchError),
                                            ("malformed", parallel_search.ParallelSearchError),
                                            ("id", parallel_search.ParallelSearchError),
                                            ("timeout", requests.Timeout)])
def test_errors_and_timeouts_propagate_without_fallback(wire, monkeypatch, failure, error):
    wire.failure = failure
    if failure == "timeout":
        monkeypatch.setattr(parallel_search, "REQUEST_TIMEOUT", (1, 0.03))
    with pytest.raises(error):
        rollout(manager(), ("A",))
    assert len(search_calls(wire)) == 1  # No silent retry or fallback.


def test_cancellation_closes_transport_and_propagates(wire, monkeypatch):
    original = requests.Session.post
    closed = []
    original_close = requests.Session.close

    def post(http, *args, **kwargs):
        if kwargs["json"]["method"] == "tools/call":
            raise KeyboardInterrupt()
        return original(http, *args, **kwargs)

    def close(http):
        closed.append(True)
        return original_close(http)

    monkeypatch.setattr(requests.Session, "post", post)
    monkeypatch.setattr(requests.Session, "close", close)
    with pytest.raises(KeyboardInterrupt):
        rollout(manager(), ("A",))
    assert closed


def test_missing_conversation_context_is_rejected(wire):
    with pytest.raises(ValueError, match="conversation IDs"):
        manager().execute_predictions(["<search>question</search>"], "", [True])
    assert not wire.records
