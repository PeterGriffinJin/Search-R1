"""
Search-R1 对比实验：基线(EM) vs 格式奖励(Format Reward)
纯RL选择法：不做SFT训练，只通过采样+选择来对比两种奖励的效果
使用 Qwen2.5-3B
"""
import os, json, re, time, gc, random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['SAFETENSORS_FAST_MMAP'] = '0'

import torch
import transformers

# ==================== 配置 ====================
MODEL_PATH = 'D:/Search-R1/modelscope/Qwen/Qwen2___5-3B'
N_AGENT = 3          # 每题采样3次
N_STEPS = 20         # 20步
MAX_NEW_TOKENS = 150

# ==================== 数据 ====================
nq_data = [
    ("total number of death row inmates in the us", ["2,718"]),
    ("do veins carry blood to the heart or away", ["to"]),
    ("who is next in line to be the monarch of england", ["Charles, Prince of Wales"]),
    ("what is the capital of france", ["Paris"]),
    ("who wrote the novel 1984", ["George Orwell"]),
    ("when did world war 2 end", ["1945"]),
    ("who was the first president of the united states", ["George Washington"]),
    ("what is the largest planet in our solar system", ["Jupiter"]),
    ("who painted the mona lisa", ["Leonardo da Vinci"]),
    ("what is the speed of light", ["299792458"]),
    ("when was the declaration of independence signed", ["1776"]),
    ("what is the chemical formula for water", ["H2O"]),
    ("who discovered penicillin", ["Alexander Fleming"]),
    ("what is the tallest mountain in the world", ["Mount Everest"]),
    ("how many continents are there", ["7"]),
    ("what is the currency of japan", ["yen"]),
    ("who invented the telephone", ["Alexander Graham Bell"]),
    ("what is the longest river in the world", ["Nile"]),
    ("when did the berlin wall fall", ["1989"]),
    ("what is the smallest country in the world", ["Vatican City"]),
    ("who was albert einstein", ["physicist"]),
    ("what is the boiling point of water", ["100"]),
    ("how many states in the us", ["50"]),
    ("what language is spoken in brazil", ["Portuguese"]),
    ("what is the largest ocean", ["Pacific"]),
    ("when did humans first land on the moon", ["1969"]),
    ("what is the main ingredient in guacamole", ["avocado"]),
    ("who was cleopatra", ["Egyptian queen"]),
    ("what is the hardest natural substance", ["diamond"]),
    ("how many bones in the human body", ["206"]),
    ("what is the capital of australia", ["Canberra"]),
    ("who wrote romeo and juliet", ["William Shakespeare"]),
    ("what is the largest desert in the world", ["Sahara"]),
    ("what is the freezing point of water", ["0"]),
    ("who was the first person in space", ["Yuri Gagarin"]),
    ("how many planets are there in the solar system", ["8"]),
    ("what is the capital of china", ["Beijing"]),
    ("who was marie curie", ["physicist"]),
    ("what is the largest country in the world by area", ["Russia"]),
    ("what is the number 1 sport in the usa", ["American football"]),
]

prompt_template = (
    'Answer the given question. '
    'You must conduct reasoning inside <think> and </think> first every time you get new information. '
    'After reasoning, if you find you lack some knowledge, you can call a search engine by '
    '<search> query </search> and it will return the top searched results between '
    '<information> and </information>. You can search as many times as your want. '
    'If you find no further external knowledge needed, you can directly provide the answer '
    'inside <answer> and </answer>, without detailed illustrations. '
    'For example, <answer> Beijing </answer>. Question: {question}\n'
)

# ==================== 知识库 ====================
print('[1/3] Loading knowledge base...')
corpus = []
with open('D:/Search-R1/real_data/corpus.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        corpus.append(json.loads(line))
extra_docs = [
    '{"id":"45","contents":"\\"Death Penalty Statistics\\"\\nThere are approximately 2,718 inmates on death row in the US."}',
    '{"id":"46","contents":"\\"Veins and Blood Flow\\"\\nVeins carry blood toward the heart."}',
    '{"id":"47","contents":"\\"British Monarchy\\"\\nCharles, Prince of Wales is heir to the throne."}',
    '{"id":"48","contents":"\\"George Orwell 1984\\"\\nGeorge Orwell wrote 1984 in 1949."}',
    '{"id":"49","contents":"\\"Speed of Light\\"\\nThe speed of light is 299,792,458 m/s."}',
    '{"id":"50","contents":"\\"Declaration of Independence\\"\\nAdopted on July 4, 1776."}',
    '{"id":"51","contents":"\\"Penicillin Discovery\\"\\nAlexander Fleming discovered penicillin in 1928."}',
    '{"id":"52","contents":"\\"Highest Mountain\\"\\nMount Everest is the tallest mountain at 8,849 meters."}',
    '{"id":"53","contents":"\\"American Football\\"\\nAmerican football is the number 1 sport in the USA."}',
    '{"id":"54","contents":"\\"Cleopatra\\"\\nCleopatra was an Egyptian queen and pharaoh."}',
    '{"id":"55","contents":"\\"Einstein\\"\\nAlbert Einstein was a theoretical physicist."}',
    '{"id":"56","contents":"\\"Yuri Gagarin\\"\\nYuri Gagarin was the first person in space in 1961."}',
    '{"id":"57","contents":"\\"Marie Curie\\"\\nMarie Curie was a physicist and chemist."}',
    '{"id":"58","contents":"\\"Diamond\\"\\nDiamond is the hardest natural substance on Earth."}',
    '{"id":"59","contents":"\\"Human Bones\\"\\nThe human body has 206 bones."}',
    '{"id":"60","contents":"\\"Canberra\\"\\nCanberra is the capital of Australia."}',
    '{"id":"61","contents":"\\"Sahara Desert\\"\\nThe Sahara is the largest desert in the world."}',
    '{"id":"62","contents":"\\"Freezing Point\\"\\nThe freezing point of water is 0 degrees Celsius."}',
    '{"id":"64","contents":"\\"H2O\\"\\nThe chemical formula for water is H2O."}',
    '{"id":"65","contents":"\\"Continents\\"\\nThere are 7 continents on Earth."}',
    '{"id":"66","contents":"\\"Water Boiling Point\\"\\nThe boiling point of water is 100 degrees Celsius."}',
    '{"id":"67","contents":"\\"Planet Jupiter\\"\\nJupiter is the largest planet in our solar system."}',
    '{"id":"68","contents":"\\"Mona Lisa\\"\\nThe Mona Lisa was painted by Leonardo da Vinci."}',
    '{"id":"69","contents":"\\"US States\\"\\nThere are 50 states in the United States."}',
    '{"id":"70","contents":"\\"Nile River\\"\\nThe Nile is the longest river in the world."}',
    '{"id":"71","contents":"\\"Berlin Wall\\"\\nThe Berlin Wall fell in 1989."}',
    '{"id":"72","contents":"\\"Vatican City\\"\\nVatican City is the smallest country in the world."}',
    '{"id":"73","contents":"\\"Telephone\\"\\nAlexander Graham Bell invented the telephone."}',
    '{"id":"74","contents":"\\"Brazil Language\\"\\nThe language spoken in Brazil is Portuguese."}',
    '{"id":"75","contents":"\\"Moon Landing\\"\\nHumans first landed on the moon in 1969 (Apollo 11)."}',
]
for doc_str in extra_docs:
    corpus.append(json.loads(doc_str))
print(f'  {len(corpus)} documents')

# ==================== 内联检索 ====================
def simple_search(query, topk=3):
    words = re.findall(r'\w+', query.lower())
    scored = []
    for doc in corpus:
        content = doc["contents"].lower()
        score = sum(1 for w in words if w in content)
        scored.append((score, doc))
    scored.sort(key=lambda x: -x[0])
    passages = ''
    for idx, (s, d) in enumerate(scored[:topk]):
        content = d["contents"]
        title = content.split('\n')[0].strip('"')
        text = '\n'.join(content.split('\n')[1:])
        passages += f'Doc {idx+1}(Title: {title}) {text}\n'
    return passages

# ==================== 加载模型 ====================
print('[2/3] Loading Qwen2.5-3B...')
gc.collect()
torch.cuda.empty_cache()

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.float16, device_map='cuda:0', low_cpu_mem_usage=True)
model.eval()
print(f'  GPU: {torch.cuda.memory_allocated()/1024**3:.2f}GB')

# ==================== 工具函数 ====================
def build_prompt(question):
    p = prompt_template.format(question=question)
    if tokenizer.chat_template:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Always put your final answer inside <answer> and </answer> tags. Use <think> and </think> for reasoning. Use <search> and </search> for search queries."},
            {"role": "user", "content": p}
        ]
        p = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return p

def normalize_answer(ans):
    ans = ans.lower().strip()
    ans = re.sub(r'[^\w\s]', '', ans)
    ans = re.sub(r'\b(a|an|the)\b', ' ', ans)
    return ' '.join(ans.split())

def em_reward(answer, golden_answers):
    if answer is None:
        return 0.0
    norm = normalize_answer(answer)
    for g in golden_answers:
        if norm == normalize_answer(g):
            return 1.0
    return 0.0

def format_reward_score(text):
    """格式奖励: think(0.1) + search(0.1) + answer(0.1)"""
    r = 0.0
    if text.count('<think>') >= 1 and text.count('<think>') == text.count('</think>'):
        r += 0.1
    if re.search(r'<search>.*?</search>', text, re.DOTALL):
        r += 0.1
    if re.search(r'<answer>.*?</answer>', text, re.DOTALL):
        r += 0.1
    return min(r, 0.3)

def has_format_tags(text):
    ht = text.count('<think>') >= 1 and text.count('<think>') == text.count('</think>')
    hs = bool(re.search(r'<search>.*?</search>', text, re.DOTALL))
    ha = bool(re.search(r'<answer>.*?</answer>', text, re.DOTALL))
    return ht, hs, ha

def extract_answer(text):
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None

def generate_response(question, temperature=0.7):
    prompt = build_prompt(question)
    full_text = prompt
    search_count = 0
    for turn in range(3):
        input_ids = tokenizer.encode(full_text, return_tensors='pt').to(model.device)
        if input_ids.shape[1] > 4000:
            input_ids = input_ids[:, -4000:]
        attention_mask = torch.ones_like(input_ids)
        try:
            with torch.no_grad():
                outputs = model.generate(input_ids, attention_mask=attention_mask,
                                         max_new_tokens=MAX_NEW_TOKENS, do_sample=(temperature > 0),
                                         temperature=temperature if temperature > 0 else None,
                                         pad_token_id=tokenizer.eos_token_id,
                                         top_p=0.95 if temperature > 0 else None)
        except Exception as e:
            return None, '', search_count, full_text
        output_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        full_text += output_text
        search_match = re.search(r'<search>(.*?)</search>', output_text, re.DOTALL)
        if search_match and turn < 2:
            search_count += 1
            passages = simple_search(search_match.group(1).strip())
            full_text += f'\n\n<information>{passages}</information>\n\n'
        elif re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL):
            break
    answer = extract_answer(full_text[len(prompt):])
    return answer, full_text[len(prompt):], search_count, full_text

def run_experiment(mode, n_samples_per_q, log_path):
    """运行实验：对每个问题采样N次，分别用两种奖励评估"""
    print(f'\n{"="*60}')
    print(f'Experiment: {mode.upper()} | {n_samples_per_q} samples/question')
    if mode == 'format':
        print('Selection: EM(0.7) + think(0.1) + search(0.1) + answer(0.1)')
    else:
        print('Selection: EM only (0 or 1)')
    print('='*60)

    results = []
    all_format_rewards = []
    all_search_counts = []
    all_tag_usage = []

    for qi, (question, golden) in enumerate(nq_data):
        samples = []
        for _ in range(n_samples_per_q):
            temp = random.choice([0.5, 0.7, 0.9])
            answer, output_text, sc, full_text = generate_response(question, temperature=temp)
            er = em_reward(answer, golden)
            fr = format_reward_score(output_text)
            ht, hs, ha = has_format_tags(output_text)

            if mode == 'format':
                total_r = 0.7 * er + fr
            else:
                total_r = er

            samples.append({
                'answer': answer, 'output': output_text, 'em': er,
                'format_reward': fr, 'search_count': sc,
                'total_reward': total_r, 'tags': (ht, hs, ha)
            })

        # 选择最佳样本
        best = max(samples, key=lambda x: x['total_reward'])
        results.append({
            'question': question, 'golden': golden,
            'best_answer': best['answer'], 'best_em': best['em'],
            'best_format_reward': best['format_reward'],
            'best_search_count': best['search_count'],
            'all_samples': samples
        })
        all_format_rewards.append(best['format_reward'])
        all_search_counts.append(best['search_count'])
        all_tag_usage.append(best['tags'])

        if (qi + 1) % 5 == 0 or qi == 0:
            em_count = sum(1 for r in results if r['best_em'] > 0)
            avg_fmt = sum(all_format_rewards) / len(all_format_rewards)
            avg_sc = sum(all_search_counts) / len(all_search_counts)
            n_think = sum(1 for t in all_tag_usage if t[0]) / len(all_tag_usage)
            n_search = sum(1 for t in all_tag_usage if t[1]) / len(all_tag_usage)
            n_answer = sum(1 for t in all_tag_usage if t[2]) / len(all_tag_usage)
            print(f'  Q{qi+1}/{len(nq_data)} | EM={em_count}/{len(results)} ({em_count/len(results):.0%}) | Fmt={avg_fmt:.3f} | Search={avg_sc:.1f} | Tags=[T:{n_think:.0%} S:{n_search:.0%} A:{n_answer:.0%}]')
            print(f'    Best: {best["answer"]} (golden: {golden[0][:20]})')

    # 汇总
    em_count = sum(1 for r in results if r['best_em'] > 0)
    summary = {
        'mode': mode,
        'total_questions': len(nq_data),
        'correct': em_count,
        'em_accuracy': em_count / len(nq_data),
        'avg_format_reward': sum(all_format_rewards) / len(all_format_rewards),
        'avg_search_count': sum(all_search_counts) / len(all_search_counts),
        'tag_usage': {
            'think': sum(1 for t in all_tag_usage if t[0]) / len(all_tag_usage),
            'search': sum(1 for t in all_tag_usage if t[1]) / len(all_tag_usage),
            'answer': sum(1 for t in all_tag_usage if t[2]) / len(all_tag_usage),
        }
    }

    log_data = {'summary': summary, 'results': []}
    for r in results:
        log_data['results'].append({
            'question': r['question'], 'golden': r['golden'][0],
            'best_answer': r['best_answer'], 'best_em': r['best_em'],
            'best_format_reward': r['best_format_reward'],
            'best_search_count': r['best_search_count'],
            'n_samples': n_samples_per_q,
        })
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print(f'\n  === {mode.upper()} RESULTS ===')
    print(f'  EM Accuracy: {em_count}/{len(nq_data)} ({summary["em_accuracy"]:.0%})')
    print(f'  Avg Format Reward: {summary["avg_format_reward"]:.3f}')
    print(f'  Avg Search Count: {summary["avg_search_count"]:.2f}')
    print(f'  Tag Usage: Think={summary["tag_usage"]["think"]:.0%} Search={summary["tag_usage"]["search"]:.0%} Answer={summary["tag_usage"]["answer"]:.0%}')
    return summary

# ==================== 主流程 ====================
print('[3/3] Running experiments...')

print('\n>>> Phase 1: Baseline (EM reward) <<<')
baseline = run_experiment('baseline', N_AGENT, 'D:/Search-R1/baseline_results.json')

print('\n>>> Phase 2: Format Reward <<<')
fmt_result = run_experiment('format', N_AGENT, 'D:/Search-R1/format_reward_results.json')

# ==================== 对比总结 ====================
print('\n\n' + '='*70)
print('FINAL COMPARISON: Baseline(EM) vs FormatReward')
print('='*70)
print(f'{"Metric":<35} {"Baseline(EM)":<15} {"FormatReward":<15} {"Diff":<10}')
print('-'*70)
for key, label in [('em_accuracy', 'EM Accuracy'), ('avg_format_reward', 'Avg Format Reward'),
                   ('avg_search_count', 'Avg Search Count')]:
    b = baseline[key]
    f = fmt_result[key]
    diff = f - b
    sign = '+' if diff > 0 else ''
    if 'accuracy' in key:
        print(f'{label:<35} {b:<15.0%} {f:<15.0%} {sign}{diff:.0%}')
    else:
        print(f'{label:<35} {b:<15.3f} {f:<15.3f} {sign}{diff:.3f}')

print('\nFormat Tag Usage:')
for tag in ['think', 'search', 'answer']:
    b = baseline['tag_usage'][tag]
    f = fmt_result['tag_usage'][tag]
    diff = f - b
    sign = '+' if diff > 0 else ''
    print(f'  {tag:<10} Baseline: {b:.0%} | Format: {f:.0%} | Diff: {sign}{diff:.0%}')
print('='*70)

comparison = {'baseline': baseline, 'format_reward': fmt_result}
with open('D:/Search-R1/comparison_results.json', 'w') as f:
    json.dump(comparison, f, indent=2, ensure_ascii=False)
print('\nResults saved to D:/Search-R1/comparison_results.json')

del model
gc.collect()
torch.cuda.empty_cache()
