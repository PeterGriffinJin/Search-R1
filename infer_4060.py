"""
Search-R1 推理脚本 - 适配RTX 4060 (8GB显存)
使用Qwen2.5-3B模型进行推理测试
"""

import transformers
import torch
import re
import requests
import gc

# ==================== 配置部分 ====================
# 模型选择（3B模型适合8GB显存）
MODEL_ID = "Qwen/Qwen2.5-3B"  # 或使用本地路径 "D:/Search-R1/models/qwen2.5-3b"

# 测试问题
QUESTIONS = [
    "Who is the first president of the United States?",
    "What is the capital of France?",
    "When did World War II end?",
    "Who wrote the novel '1984'?",
    "What is the largest planet in our solar system?",
]

# 搜索引擎地址（需要先启动检索服务器）
SEARCH_URL = "http://127.0.0.1:8000/retrieve"

# ==================== 工具函数 ====================

def clear_gpu_memory():
    """清理GPU显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB / "
              f"{torch.cuda.memory_reserved()/1024**3:.2f}GB")


def get_query(text):
    """从模型输出中提取搜索查询"""
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    return None


def search(query):
    """调用搜索引擎"""
    try:
        payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
        results = requests.post(SEARCH_URL, json=payload, timeout=10).json()['result']

        def _passages2string(retrieval_result):
            format_reference = ''
            for idx, doc_item in enumerate(retrieval_result):
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            return format_reference

        return _passages2string(results[0])
    except Exception as e:
        print(f"搜索失败: {e}")
        return "搜索服务不可用，请检查检索服务器是否启动。"


# ==================== 主程序 ====================

def main():
    print("=" * 60)
    print("Search-R1 推理测试 (RTX 4060 适配版)")
    print("=" * 60)

    # 检查GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

    # 加载模型
    print(f"\n正在加载模型: {MODEL_ID}")
    print("这可能需要几分钟...")

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,  # 使用float16而非bfloat16
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print("模型加载成功！")
        clear_gpu_memory()
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请检查：")
        print("1. 模型是否已下载到本地")
        print("2. 显存是否足够")
        return

    # Prompt模板
    prompt_template = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

    # 停止条件
    curr_eos = [151645, 151643]  # Qwen2.5的EOS token
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    target_sequences = ["</search>", " </search>", "</search>\n"]
    target_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in target_sequences]
    target_lengths = [len(ids) for ids in target_ids]

    class StopOnSequence(transformers.StoppingCriteria):
        def __init__(self, target_ids, target_lengths):
            self.target_ids = target_ids
            self.target_lengths = target_lengths

        def __call__(self, input_ids, scores, **kwargs):
            if input_ids.shape[1] < min(self.target_lengths):
                return False
            for target, length in zip(self.target_ids, self.target_lengths):
                if torch.equal(input_ids[0, -length:], torch.as_tensor(target, device=input_ids.device)):
                    return True
            return False

    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_ids, target_lengths)])

    # 测试每个问题
    for q_idx, question in enumerate(QUESTIONS):
        print(f"\n{'='*60}")
        print(f"问题 {q_idx+1}: {question}")
        print("=" * 60)

        if question.strip()[-1] != '?':
            question = question.strip() + '?'
        else:
            question = question.strip()

        prompt = prompt_template.format(question=question)

        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )

        print(f"\n[Prompt]\n{prompt[:200]}...")

        turn = 0
        max_turns = 3  # 4060上限制轮数

        while turn < max_turns:
            turn += 1
            print(f"\n--- 第{turn}轮生成 ---")

            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,  # 减小以节省显存
                    stopping_criteria=stopping_criteria,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7
                )

            if outputs[0][-1].item() in curr_eos:
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                print(f"[最终回答] {output_text}")
                break

            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # 尝试搜索
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            query = get_query(full_text)

            if query:
                print(f"[搜索查询] {query}")
                search_results = search(query)
                search_text = curr_search_template.format(
                    output_text=output_text,
                    search_results=search_results
                )
                prompt += search_text
                print(f"[搜索结果] {search_results[:200]}...")
            else:
                print(f"[模型输出] {output_text}")
                prompt += output_text

        clear_gpu_memory()
        print()

    print("\n" + "=" * 60)
    print("推理测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
