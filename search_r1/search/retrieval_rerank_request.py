import requests

# URL for your local FastAPI server
retrieval_url = "http://10.100.20.13:8000/retrieve"
rerank_url = "http://10.100.20.20:6980/rerank"

# Example payload
retrieval_request = {
    "queries": ["What is the capital of France?", "Explain neural networks."],
    "topk": 10,
    "return_scores": True
}

# Send POST request
response = requests.post(retrieval_url, json=retrieval_request)

# Raise an exception if the request failed
response.raise_for_status()

# Get the JSON response
retrieved_data = response.json()

print("Response from retirever server:")
print(retrieved_data)

rerank_request = {
    "queries": ["What is the capital of France?", "Explain neural networks."],
    "documents": retrieved_data["result"],
    "rerank_topk": 3,
    "return_scores": True
}

response = requests.post(rerank_url, json=rerank_request)
response.raise_for_status()
rerank_data = response.json()
print("Response from reranker server:")
print(rerank_data)
# [ {"document": {"id": "xxx", "contents": "yyy"}, "score": "zzz"}}, {}]