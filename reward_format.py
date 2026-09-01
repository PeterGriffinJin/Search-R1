"""
Search-R1 改进：格式奖励函数
在原始EM奖励的基础上，增加格式规范性奖励

改进思路：
原始Search-R1只使用EM（Exact Match）作为奖励，奖励信号非常稀疏。
通过增加格式奖励，可以引导模型正确使用<think>、<search>、<answer>标签，
加速训练收敛，减少格式错误。

奖励组成：
- 格式奖励 (0.3分)：正确使用标签格式
- 过程奖励 (0.2分)：推理和搜索过程的质量
- 最终奖励 (0.5分)：答案的EM匹配
"""

import re


def extract_solution(text):
    """
    从模型输出中提取答案
    要求至少有2个<answer>标签（确保经过了推理/搜索阶段）
    """
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if len(matches) < 2:
        return None
    return matches[-1].strip()


def normalize_answer(answer):
    """标准化答案，用于EM匹配"""
    # 转小写
    answer = answer.lower()
    # 去除标点
    answer = re.sub(r'[^\w\s]', '', answer)
    # 去除冠词
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
    # 去除多余空格
    answer = ' '.join(answer.split())
    return answer


def exact_match_score(prediction, ground_truth):
    """计算EM分数"""
    normalized_pred = normalize_answer(prediction)
    normalized_gt = normalize_answer(ground_truth)
    return 1.0 if normalized_pred == normalized_gt else 0.0


def compute_format_reward(text):
    """
    计算格式奖励

    检查模型是否正确使用了各种标签：
    - <think>标签：必须成对出现
    - <search>标签：如果使用了搜索
    - <answer>标签：必须有至少2个（确保经过推理阶段）
    """
    reward = 0.0

    # 检查<think>标签
    think_open = text.count('<think>')
    think_close = text.count('</think>')
    if think_open >= 1 and think_open == think_close:
        reward += 0.1  # 正确使用了推理标签

    # 检查<search>标签（如果使用了搜索）
    search_matches = re.findall(r'<search>(.*?)</search>', text, re.DOTALL)
    if len(search_matches) > 0:
        # 检查搜索查询是否非空
        for query in search_matches:
            if query.strip():
                reward += 0.05  # 每个有效搜索查询加分
        reward = min(reward, 0.15)  # 搜索部分最多0.15分

    # 检查<answer>标签
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if len(answer_matches) >= 2:
        reward += 0.15  # 有至少2个answer标签，说明经过了推理阶段
    elif len(answer_matches) == 1:
        reward += 0.05  # 只有1个answer标签，部分奖励

    return min(reward, 0.3)  # 格式奖励最多0.3分


def compute_process_reward(text):
    """
    计算过程奖励

    评估推理和搜索过程的质量：
    - 推理内容是否合理
    - 搜索查询是否与问题相关
    - 是否避免了重复搜索
    """
    reward = 0.0

    # 检查推理内容
    think_matches = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_matches:
        # 简单检查推理内容长度（太短可能说明没有认真推理）
        avg_think_length = sum(len(t) for t in think_matches) / len(think_matches)
        if avg_think_length > 20:
            reward += 0.1  # 推理内容有一定长度

    # 检查搜索查询
    search_matches = re.findall(r'<search>(.*?)</search>', text, re.DOTALL)
    if len(search_matches) > 0:
        # 检查是否有重复搜索
        unique_queries = set(q.strip().lower() for q in search_matches)
        if len(unique_queries) == len(search_matches):
            reward += 0.1  # 没有重复搜索

        # 检查搜索次数是否合理（1-3次比较好）
        if 1 <= len(search_matches) <= 3:
            reward += 0.05

    return min(reward, 0.2)  # 过程奖励最多0.2分


def compute_reward_with_format(prediction, ground_truth, text):
    """
    计算综合奖励

    参数：
        prediction: 模型预测的答案
        ground_truth: 标准答案列表
        text: 模型的完整输出（包含推理、搜索、答案）

    返回：
        total_reward: 总奖励 (0-1)
        breakdown: 奖励分解详情
    """
    # 1. EM奖励 (0.5分)
    em_score = 0.0
    if prediction:
        for gt in ground_truth:
            if exact_match_score(prediction, gt) > 0:
                em_score = 0.5
                break

    # 2. 格式奖励 (0.3分)
    format_reward = compute_format_reward(text)

    # 3. 过程奖励 (0.2分)
    process_reward = compute_process_reward(text)

    # 总奖励
    total_reward = em_score + format_reward + process_reward

    breakdown = {
        'em_score': em_score,
        'format_reward': format_reward,
        'process_reward': process_reward,
        'total_reward': total_reward
    }

    return total_reward, breakdown


def compute_reward_original(prediction, ground_truth):
    """
    原始Search-R1的奖励函数（用于对比）
    只使用EM匹配，二元奖励
    """
    if prediction is None:
        return 0.0

    for gt in ground_truth:
        if exact_match_score(prediction, gt) > 0:
            return 1.0
    return 0.0


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 测试用例
    test_cases = [
        {
            "text": """<think>I need to find the first president of the United States.</think>
<search>first president of the United States</search>

<information>George Washington was the first president of the United States...</information>

<think>Based on the search results, George Washington was the first president.</think>

<answer> George Washington </answer>""",
            "ground_truth": ["George Washington", "Washington"],
            "prediction": "George Washington"
        },
        {
            "text": """<think>Let me think about this.</think>
<answer> Paris </answer>""",
            "ground_truth": ["Paris"],
            "prediction": "Paris"
        },
        {
            "text": """The capital of France is Paris.""",
            "ground_truth": ["Paris"],
            "prediction": "Paris"
        }
    ]

    print("=" * 60)
    print("格式奖励函数测试")
    print("=" * 60)

    for i, case in enumerate(test_cases):
        print(f"\n测试用例 {i+1}:")
        print(f"文本: {case['text'][:100]}...")

        # 新方法奖励
        new_reward, breakdown = compute_reward_with_format(
            case['prediction'], case['ground_truth'], case['text']
        )
        print(f"\n改进后奖励: {new_reward:.3f}")
        print(f"  - EM奖励: {breakdown['em_score']:.3f}")
        print(f"  - 格式奖励: {breakdown['format_reward']:.3f}")
        print(f"  - 过程奖励: {breakdown['process_reward']:.3f}")

        # 原始方法奖励
        old_reward = compute_reward_original(case['prediction'], case['ground_truth'])
        print(f"原始奖励: {old_reward:.3f}")

        print("-" * 40)
