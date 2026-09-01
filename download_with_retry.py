"""
带重试的下载脚本 - 下载NQ数据和wiki-18语料
支持断点续传和自动重试
"""
import os
import sys
import time
import json
import subprocess

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'

print('='*60)
print('Search-R1 数据下载（带重试）')
print('='*60)

# Step 1: 下载NQ数据集
print('\n[Step 1] 下载NQ数据集 (RUC-NLPIR/FlashRAG_datasets)...')

max_retries = 5
for attempt in range(max_retries):
    try:
        result = subprocess.run(
            [sys.executable, '-c', """
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'
from datasets import load_dataset
print('Loading NQ dataset...')
ds = load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq', cache_dir='D:/Search-R1/data_cache')
print(f'Train: {len(ds["train"])} examples')
print(f'Test: {len(ds["test"])} examples')
print(f'Features: {list(ds["train"].features.keys())}')
# 保存前100条训练数据
import json
train_data = []
for i, item in enumerate(ds['train']):
    if i >= 100:
        break
    train_data.append(item)
with open('D:/Search-R1/data/nq_train_sample.json', 'w') as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)
print(f'Saved {len(train_data)} train examples to nq_train_sample.json')
# 保存测试数据
test_data = []
for i, item in enumerate(ds['test']):
    if i >= 50:
        break
    test_data.append(item)
with open('D:/Search-R1/data/nq_test_sample.json', 'w') as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)
print(f'Saved {len(test_data)} test examples to nq_test_sample.json')
"""],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print(result.stdout)
            print('NQ数据集下载成功!')
            break
        else:
            print(f'  尝试 {attempt+1}/{max_retries} 失败')
            print(f'  错误: {result.stderr[-200:]}')
            time.sleep(5)
    except Exception as e:
        print(f'  尝试 {attempt+1}/{max_retries} 异常: {e}')
        time.sleep(5)

# Step 2: 下载BM25索引（较小的文件）
print('\n[Step 2] 尝试下载wiki-18 BM25索引...')
for attempt in range(3):
    try:
        result = subprocess.run(
            [sys.executable, '-c', """
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download
print('Downloading BM25 index...')
hf_hub_download(
    repo_id='PeterJinGo/wiki-18-bm25-index',
    filename='bm25',
    repo_type='dataset',
    local_dir='D:/Search-R1/data',
)
print('BM25 index downloaded!')
"""],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print(result.stdout)
            print('BM25索引下载成功!')
            break
        else:
            print(f'  尝试 {attempt+1}/3 失败: {result.stderr[-200:]}')
            time.sleep(10)
    except Exception as e:
        print(f'  尝试 {attempt+1}/3 异常: {e}')
        time.sleep(10)

# Step 3: 下载wiki-18语料（只下载一部分）
print('\n[Step 3] 尝试下载wiki-18语料库...')
# wiki-18.jsonl.gz 太大(27GB)，尝试下载一个较小的版本
for attempt in range(3):
    try:
        result = subprocess.run(
            [sys.executable, '-c', """
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download
print('Downloading wiki-18 corpus (this is ~27GB, may take a while)...')
try:
    hf_hub_download(
        repo_id='PeterJinGo/wiki-18-corpus',
        filename='wiki-18.jsonl.gz',
        repo_type='dataset',
        local_dir='D:/Search-R1/data',
    )
    print('wiki-18 corpus downloaded!')
except Exception as e:
    print(f'Download failed: {e}')
    print('Will use a smaller corpus instead.')
"""],
            capture_output=True, text=True, timeout=1200
        )
        print(result.stdout[-500:])
        if result.returncode == 0 and 'downloaded!' in result.stdout:
            print('wiki-18语料下载成功!')
            break
        else:
            print(f'  尝试 {attempt+1}/3: 语料太大或网络不稳')
    except Exception as e:
        print(f'  尝试 {attempt+1}/3 异常: {e}')

# 检查下载结果
print('\n' + '='*60)
print('下载结果检查')
print('='*60)
for f in ['nq_train_sample.json', 'nq_test_sample.json', 'bm25', 'wiki-18.jsonl.gz']:
    path = f'D:/Search-R1/data/{f}'
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f'[OK] {f}: {size/1024/1024:.1f}MB')
    else:
        print(f'[MISSING] {f}')
