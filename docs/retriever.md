
## Search Engine

In this document, we provide examples of how to launch different retrievers, including local sparse retriever (e.g., BM25), local dense retriever (e.g., e5) and online search engine.
For local retrievers, we use [wiki-18](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus) corpus as an example and the corpus indexing can be found at [bm25](https://huggingface.co/datasets/PeterJinGo/wiki-18-bm25-index), [e5-flat](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index), [e5-HNSW64](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index-HNSW64).

### How to choose the retriever?

- If you have a private or domain-specific corpus, choose **local retriever**.

    - If there is no high quality embedding-based retrievers (dense retrievers) in your domain, choose **sparse local retriever** (e.g., BM25).

    - Otherwise choose **dense local retriever**.
    
        - If you do not have sufficent GPUs to conduct exact dense embedding matching, choose **ANN indexing** on CPUs.

        - If you have sufficient GPUs, choose **flat indexing** on GPUs.


- If you want to train a general LLM search agent and have enough funding, choose **online search engine** (e.g., [SerpAPI](https://serpapi.com/)).


- If you have a domain specific online search engine (e.g., PubMed search), you can refer to [link](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/serp_search_server.py) to integrate it to Search-R1 by yourself.

Search engine launching scripts can be found at [link](https://github.com/PeterGriffinJin/Search-R1/tree/main/example/retriever).

### Local Sparse Retriever

Sparse retriever (e.g., bm25) is a traditional method. The retrieval process is very efficient and no GPUs are needed. However, it may not be as accurate as dense retrievers in some specific domain.

(1) Download the indexing.
```bash
save_path=/your/path/to/save
huggingface-cli download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir $save_path
```

(2) Launch a local BM25 retriever server.
```bash
conda activate retriever

index_file=$save_path/bm25
corpus_file=$save_path/wiki-18.jsonl
retriever_name=bm25

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name
```


### Local Dense Retriever

You can also adopt some off-the-shelf dense retrievers, e.g., e5. These models are much stronger than sparse retriever in some specific domains.
If you have sufficient GPU, we would recommend the flat indexing variant below, otherwise you can adopt the ANN variant.

#### Flat indexing

Flat indexing conducts exact embedding match, which is slow but very accurate. To make it efficient enough to support online RL, we would recommend enable **GPU** usage by ```--faiss_gpu```.

(1) Download the indexing and corpus.
```bash
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

(2) Launch a local flat e5 retriever server.

```bash
conda activate retriever

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path --faiss_gpu

```


#### ANN indexing (HNSW64)

To improve the search efficient with only **CPU**, you can adopt approximate nearest neighbor (ANN) indexing, e.g., with HNSW64.
It is very efficient, but may not be as accurate as flat indexing, especially when the number of retrieved passages is small.

(1) Download the indexing.
```bash
save_path=/the/path/to/save
huggingface-cli download PeterJinGo/wiki-18-e5-index-HNSW64 --repo-type dataset --local-dir $save_path
cat $save_path/part_* > $save_path/e5_HNSW64.index
```


(2) Launch a local ANN dense retriever server.
```bash
conda activate retriever

index_file=$save_path/e5_HNSW64.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path
```


### Online Search Engine

We support both [Google Search API](https://developers.google.com/custom-search/v1/overview) and [SerpAPI](https://serpapi.com/). We would recommend [SerpAPI](https://serpapi.com/) since it integrates multiple online search engine APIs (including Google, Bing, Baidu, etc) and does not have a monthly quota limitation ([Google Search API](https://developers.google.com/custom-search/v1/overview) has a hard 10k monthly quota, which is not sufficient to fulfill online LLM RL training).

#### SerpAPI online search server

```bash
search_url=https://serpapi.com/search
serp_api_key="" # put your serp api key here (https://serpapi.com/)

python search_r1/search/serp_search_server.py --search_url $search_url --topk 3 --serp_api_key $serp_api_key
```

#### Google online search server

```bash
api_key="" # put your google custom API key here (https://developers.google.com/custom-search/v1/overview)
cse_id="" # put your google cse API key here (https://developers.google.com/custom-search/v1/overview)

python search_r1/search/google_search_server.py --api_key $api_key --topk 5 --cse_id $cse_id --snippet_only
```

