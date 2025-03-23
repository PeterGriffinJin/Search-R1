#!/bin/bash


python search_r1/search/reranker_server.py \
                    --rerank_topk 3 \
                    --rerank_model_name_or_path "cross-encoder/ms-marco-MiniLM-L12-v2" \
                    --batch_size 32 \
                    --reranker_type "sentence_transformer" 