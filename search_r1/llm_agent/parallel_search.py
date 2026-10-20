"""Anonymous Parallel Search MCP over synchronous Streamable HTTP.

Keep this transport independent of the MCP Python SDK so Search-R1's Python 3.9
environment can use it. Conversation IDs are supplied by the rollout, never by
the HTTP session (one connection may serve several independent trajectories).
"""

import json

import requests


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "Search-R1 (https://github.com/PeterGriffinJin/Search-R1)"
PROTOCOL_VERSION = "2025-03-26"
REQUEST_TIMEOUT = (10, 60)


class ParallelSearchError(RuntimeError):
    """An MCP or tool response could not supply search evidence."""


def _read_response(response, request_id):
    """Accept either Streamable HTTP response format, stopping at our reply."""
    if response.headers.get("Content-Type", "").split(";")[0] == "text/event-stream":
        response.encoding = "utf-8"
        data = []
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif not line and data:
                message = json.loads("\n".join(data))
                data = []
                if message.get("id") == request_id:
                    return message
        raise ParallelSearchError("MCP stream ended without a matching response")
    return response.json()


class _MCPConnection:
    def __init__(self, http):
        self.http = http
        self.request_id = 0

    def request(self, method, params=None, notification=False):
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        if not notification:
            self.request_id += 1
            payload["id"] = self.request_id
        with self.http.post(MCP_URL, json=payload, timeout=REQUEST_TIMEOUT,
                            stream=True, allow_redirects=False) as response:
            response.raise_for_status()
            if 300 <= response.status_code < 400:
                raise ParallelSearchError("Unexpected MCP redirect")
            if notification:
                return None
            if method == "initialize":
                session_id = response.headers.get("Mcp-Session-Id")
                if session_id:
                    self.http.headers["Mcp-Session-Id"] = session_id
            message = _read_response(response, payload["id"])
        if message.get("jsonrpc") != "2.0" or message.get("id") != payload["id"]:
            raise ParallelSearchError("Mismatched MCP response")
        if "error" in message:
            raise ParallelSearchError("Parallel MCP error: " + str(message["error"]))
        result = message.get("result")
        if not isinstance(result, dict):
            raise ParallelSearchError("Missing MCP result")
        return result

    def initialize(self):
        result = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "Search-R1", "version": "0.1"},
        })
        if result.get("protocolVersion") != PROTOCOL_VERSION:
            raise ParallelSearchError("Unsupported MCP protocol version")
        self.http.headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
        self.request("notifications/initialized", notification=True)
        tools = self.request("tools/list")
        if not any(tool.get("name") == "web_search" for tool in tools.get("tools", [])):
            raise ParallelSearchError("Parallel MCP does not expose web_search")


def _documents(result, topk):
    if result.get("isError"):
        raise ParallelSearchError("Parallel web_search failed: " + str(result.get("content")))
    data = result.get("structuredContent")
    if data is None:
        # Older MCP servers return the same object as a JSON text block.
        blocks = [block["text"] for block in result.get("content", [])
                  if block.get("type") == "text"]
        try:
            data = json.loads("\n".join(blocks))
        except (ValueError, TypeError) as exc:
            raise ParallelSearchError("Invalid Parallel search content") from exc
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ParallelSearchError("Missing Parallel search results")
    documents = []
    for item in data["results"][:topk]:
        url = item.get("url")
        excerpts = item.get("excerpts")
        if (not isinstance(url, str) or not url.startswith(("http://", "https://"))
                or not isinstance(excerpts, list)
                or not all(isinstance(text, str) for text in excerpts)):
            raise ParallelSearchError("Invalid Parallel search result")
        title = item.get("title") or url
        documents.append({"document": {
            "contents": "{}\nSource: {}\n{}".format(title, url, "\n".join(excerpts))
        }})
    return documents


def batch_search(queries, session_ids, topk):
    """Return the existing retriever contract; never fall back to another provider.

    Each query belongs to a separate trajectory. The caller must reuse that
    trajectory's ID on subsequent turns, including after an error or retry.
    """
    if len(queries) != len(session_ids) or any(not sid for sid in session_ids):
        raise ValueError("Parallel search needs one conversation ID per query")
    if topk < 1:
        raise ValueError("topk must be positive")
    if not queries:
        return {"result": []}
    with requests.Session() as http:
        # Suppress implicit .netrc authentication while retaining configured
        # proxies and CA bundles. No ambient API-key variables are consulted.
        http.auth = lambda request: request
        http.headers.update({"User-Agent": USER_AGENT,
                             "Accept": "application/json, text/event-stream"})
        connection = _MCPConnection(http)
        connection.initialize()
        results = []
        for query, session_id in zip(queries, session_ids):
            result = connection.request("tools/call", {
                "name": "web_search",
                "arguments": {"objective": query, "search_queries": [query],
                              "session_id": session_id},
            })
            results.append(_documents(result, topk))
    return {"result": results}
