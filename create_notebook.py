import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

# Cell 1
md1 = '''# Cell 1: 环境配置与依赖安装

- 安装 openai, wikipedia, datasets, matplotlib, tqdm
- 从环境变量读取 API Key 与 base_url
- 固定随机种子，设置模型参数
'''
code1 = '''# 如需安装依赖，取消下面注释并运行一次
# !pip install openai wikipedia datasets matplotlib tqdm -q

import os
import re
import json
import random
import time
import textwrap
from typing import List, Dict, Tuple, Optional

import requests
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset

# 固定随机种子，保证数据采样可复现
random.seed(42)

# ========================
# OpenAI 兼容 API 配置
# ========================
# 请在本 cell 运行前设置环境变量：
# export KIMI_API_KEY="your_key"
# export KIMI_BASE_URL="https://api.moonshot.cn/v1"   # Kimi 兼容接口
# export MODEL_NAME="kimi-k2p5-coding"               # 或 "gpt-4o" 等

API_KEY = os.environ.get("KIMI_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("KIMI_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://api.moonshot.cn/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "kimi-k2p5-coding")

if not API_KEY:
    raise ValueError("请先设置 KIMI_API_KEY 或 OPENAI_API_KEY 环境变量")

from openai import OpenAI
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 实验超参
TEMPERATURE = 0.0          # 尽量确定性，便于复现
MAX_REACT_STEPS = 10       # ReAct 最大步数
N_SAMPLES = 50             # 测试样本数（50-100 之间可调）
TOP_N_SEARCH = 3           # Wiki 搜索返回候选数

print(f"Model: {MODEL_NAME}")
print(f"Base URL: {BASE_URL}")
'''

# Cell 2
md2 = '''# Cell 2: 数据集加载与预处理

- 从 HuggingFace `datasets` 加载 HotPotQA
- 选取 validation 集前 N 条（按固定 seed 打乱后取前 N，保证复现）
- 展示数据格式
'''
code2 = '''# 加载 HotPotQA（fullwiki 设置，问题可直接用 Wikipedia 回答）
# 该数据集规模较大，首次下载需联网
dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
print(f"Total validation samples: {len(dataset)}")

# 用固定 seed 打乱并取前 N 条
indices = list(range(len(dataset)))
random.seed(42)
random.shuffle(indices)
selected_indices = indices[:N_SAMPLES]

test_data = []
for idx in selected_indices:
    item = dataset[idx]
    test_data.append({
        "id": item.get("_id", idx),
        "question": item["question"],
        "answer": item["answer"],
        "type": item.get("type", "unknown"),
    })

print(f"Selected {len(test_data)} samples")
print("\\nExample:")
print(json.dumps(test_data[0], ensure_ascii=False, indent=2))
'''

# Cell 3
md3 = '''# Cell 3: 工具函数定义

- Wikipedia 搜索与摘要（基于 MediaWiki API，比 wikipedia 包更稳定）
- Token 计数与累加
- 答案评估函数（严格匹配 + 宽松包含匹配）
- LLM 调用封装（统一接口 + usage 提取）
'''
code3 = '''# ---------------------------
# Wikipedia 工具
# ---------------------------
WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "ReflexionReproBot/1.0 (research@example.com)"}


def wiki_search(query: str, top_n: int = TOP_N_SEARCH) -> List[str]:
    \"\"\"返回与 query 最相关的 Wikipedia 页面标题列表。\"\"\"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": top_n,
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        results = r.json().get("query", {}).get("search", [])
        return [x["title"] for x in results]
    except Exception as e:
        print(f"[Wiki search error] {e}")
        return []


def wiki_summary(title: str, sentences: int = 5) -> str:
    \"\"\"获取指定 Wikipedia 页面的前 sentences 句摘要。\"\"\"
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "explaintext": True,
        "exsentences": sentences,
        "format": "json",
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract:
                return extract.strip()
        return ""
    except Exception as e:
        print(f"[Wiki summary error] {e}")
        return ""


# ---------------------------
# LLM 调用 + Token 统计
# ---------------------------
def call_llm(system: str, user: str, temperature: float = TEMPERATURE, **kwargs) -> Tuple[str, Dict]:
    \"\"\"调用 LLM，返回 (content, usage_dict)。\"\"\"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            **kwargs
        )
        content = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "total_tokens": resp.usage.total_tokens if resp.usage else 0,
        }
        return content.strip(), usage
    except Exception as e:
        print(f"[LLM error] {e}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def normalize_answer(s: str) -> str:
    \"\"\"去除冠词、标点、多余空格，统一小写。\"\"\"
    s = s.lower().strip()
    s = re.sub(r"\\b(a|an|the)\\b", " ", s)
    s = re.sub(r"[^\\w\\s]", "", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s


def eval_answer(pred: str, gold: str) -> Dict:
    \"\"\"严格匹配 + 宽松包含匹配。\"\"\"
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    exact = pred_norm == gold_norm
    loose = gold_norm in pred_norm or pred_norm in gold_norm or any(g in pred_norm for g in gold_norm.split())
    return {"exact": exact, "loose": loose}


def extract_boxed(text: str) -> str:
    \"\"\"尝试提取 \\boxed{} 或最后的 Answer: 后的内容。\"\"\"
    m = re.search(r"\\\\boxed\\{([^}]+)\\}", text)
    if m:
        return m.group(1).strip()
    # 匹配 "Answer: xxx" 或 "答案是 xxx"
    lines = text.split("\\n")
    for line in reversed(lines):
        if "answer:" in line.lower():
            return line.split(":", 1)[1].strip()
    return text.strip()


print("Tool functions ready.")
print("Wiki search test:", wiki_search("Barack Obama", top_n=1)[:1])
'''

# Cell 4
md4 = '''# Cell 4: CoT 实现

- Prompt 要求模型逐步推理，最后以 `Answer: xxx` 给出答案
- 解析最终答案并记录 Token 消耗
'''
code4 = '''COT_SYSTEM = '''You are a helpful question-answering assistant.
Answer the following multi-hop question by reasoning step by step.
End your response with a line that starts with "Answer:" followed by the final answer.'''

COT_TEMPLATE = '''Question: {question}

Let's think step by step.'''


def run_cot(question: str) -> Dict:
    \"\"\"执行 CoT 范式。\"\"\"
    user_prompt = COT_TEMPLATE.format(question=question)
    response, usage = call_llm(COT_SYSTEM, user_prompt)
    answer = extract_boxed(response)
    return {
        "raw": response,
        "answer": answer,
        "tokens": usage["total_tokens"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
    }


# 快速自测
sample_q = test_data[0]["question"]
print("Sample question:", sample_q)
cot_res = run_cot(sample_q)
print("\\nCoT raw response (truncated):")
print(textwrap.shorten(cot_res["raw"], width=500))
print("\\nExtracted answer:", cot_res["answer"])
print("Tokens:", cot_res["tokens"])
'''

# Cell 5
md5 = '''# Cell 5: ReAct 实现

严格按论文 ReAct 格式：
- Thought: 推理内容
- Action: Search[entity] 或 Finish[answer]
- Observation: 工具返回结果
- 循环直到 Finish 或 max_steps
'''
code5 = '''REACT_SYSTEM = '''You are a question-answering agent that solves multi-hop questions by interleaving Thought, Action, and Observation.

Available actions:
- Search[entity]: search Wikipedia for "entity" and get a short summary
- Finish[answer]: submit the final answer and stop

You must strictly follow this format:
Thought: <your reasoning>
Action: Search[<query>] or Finish[<answer>]

After each Action, you will receive an Observation. Continue until you are ready to finish.'''

REACT_TEMPLATE = '''Solve the following question by interacting with Wikipedia.

Question: {question}

{scratchpad}

Thought:'''


def parse_action(text: str) -> Tuple[str, str]:
    \"\"\"从模型输出解析 Action 类型与参数。\"\"\"
    # 找到最后一个 Thought/Action 行
    action_match = re.search(r"Action:\\s*(Search|Finish)\\[(.*?)\\]", text, re.IGNORECASE | re.DOTALL)
    if not action_match:
        return "None", ""
    act_type = action_match.group(1).strip()
    act_arg = action_match.group(2).strip()
    return act_type, act_arg


def run_react(question: str, reflection: Optional[str] = None) -> Dict:
    \"\"\"执行 ReAct 范式；若提供 reflection，则将其拼入初始 prompt。\"\"\"
    scratchpad = ""
    if reflection:
        scratchpad += f"Previous reflection: {reflection}\\n\\n"

    trajectory = []   # 保存完整轨迹，用于 Reflexion
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for step in range(MAX_REACT_STEPS):
        user_prompt = REACT_TEMPLATE.format(question=question, scratchpad=scratchpad)
        response, usage = call_llm(REACT_SYSTEM, user_prompt)
        total_usage["prompt_tokens"] += usage["prompt_tokens"]
        total_usage["completion_tokens"] += usage["completion_tokens"]
        total_usage["total_tokens"] += usage["total_tokens"]

        # 记录这一步
        step_record = {"step": step, "llm_output": response}

        # 解析 Action
        act_type, act_arg = parse_action(response)

        if act_type.lower() == "finish":
            step_record["action"] = "Finish"
            step_record["observation"] = ""
            trajectory.append(step_record)
            return {
                "answer": act_arg,
                "trajectory": trajectory,
                "tokens": total_usage["total_tokens"],
                "prompt_tokens": total_usage["prompt_tokens"],
                "completion_tokens": total_usage["completion_tokens"],
                "finished": True,
                "steps": step + 1,
            }

        elif act_type.lower() == "search":
            titles = wiki_search(act_arg, top_n=1)
            if titles:
                obs = wiki_summary(titles[0], sentences=3)
            else:
                obs = "No relevant Wikipedia page found."
            step_record["action"] = f"Search[{act_arg}]"
            step_record["observation"] = obs
            trajectory.append(step_record)
            # 拼接进 scratchpad
            scratchpad += f"Thought: {response}\\nAction: Search[{act_arg}]\\nObservation: {obs}\\n\\n"
        else:
            # 未产生可解析 Action，视为失败
            step_record["action"] = "ParseError"
            step_record["observation"] = ""
            trajectory.append(step_record)
            break

    # 达到 max_steps 仍未 Finish
    return {
        "answer": "",
        "trajectory": trajectory,
        "tokens": total_usage["total_tokens"],
        "prompt_tokens": total_usage["prompt_tokens"],
        "completion_tokens": total_usage["completion_tokens"],
        "finished": False,
        "steps": MAX_REACT_STEPS,
    }


# 快速自测
print("Sample question:", sample_q)
react_res = run_react(sample_q)
print("\\nReAct answer:", react_res["answer"])
print("Finished:", react_res["finished"], "Steps:", react_res["steps"])
print("Tokens:", react_res["tokens"])
print("\\nTrajectory (first step):")
if react_res["trajectory"]:
    print(json.dumps(react_res["trajectory"][0], ensure_ascii=False, indent=2))
'''

# Cell 6
md6 = '''# Cell 6: Reflexion 实现

严格按论文流程：
1. 第一轮 ReAct → 得到答案与完整轨迹
2. 评估答案
3. 若错误，用完整轨迹 + 标准答案生成反思
4. 第二轮 ReAct，将反思拼入 prompt
5. 记录两轮总 Token 消耗
'''
code6 = '''REFLECT_SYSTEM = '''You are a reflection assistant.
Given a failed question-answering trajectory (Thought-Action-Observation) and the correct answer, identify what went wrong and produce a concise, actionable reflection.
The reflection should suggest how the agent could avoid the same mistake next time.'''

REFLECT_TEMPLATE = '''The agent failed to answer the question correctly.

Question: {question}
Correct Answer: {gold_answer}

Agent Trajectory:
{trajectory_text}

Please provide a concise reflection (1-2 sentences) explaining the error and how to fix it.
Reflection:'''


def format_trajectory(trajectory: List[Dict]) -> str:
    \"\"\"把轨迹序列转换成文本，用于反思 prompt。\"\"\"
    lines = []
    for rec in trajectory:
        lines.append(f"Step {rec['step']}:")
        lines.append(rec["llm_output"])
        if rec.get("action"):
            lines.append(f"Action: {rec['action']}")
        if rec.get("observation"):
            lines.append(f"Observation: {rec['observation']}")
        lines.append("")
    return "\\n".join(lines)


def generate_reflection(question: str, gold_answer: str, trajectory: List[Dict]) -> Tuple[str, Dict]:
    \"\"\"生成语言化反思。\"\"\"
    traj_text = format_trajectory(trajectory)
    user_prompt = REFLECT_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        trajectory_text=traj_text,
    )
    reflection, usage = call_llm(REFLECT_SYSTEM, user_prompt)
    return reflection, usage


def run_reflexion(question: str, gold_answer: str) -> Dict:
    \"\"\"执行 Reflexion 范式：ReAct → 评估 → 反思 → ReAct(带反思)。\"\"\"
    # 第一轮 ReAct
    first = run_react(question)
    eval_first = eval_answer(first["answer"], gold_answer)
    total_tokens = first["tokens"]

    if eval_first["loose"]:
        # 第一轮已正确，无需反思
        return {
            "first_answer": first["answer"],
            "second_answer": first["answer"],
            "reflection": "",
            "first_correct": True,
            "second_correct": True,
            "tokens": total_tokens,
            "prompt_tokens": first["prompt_tokens"],
            "completion_tokens": first["completion_tokens"],
            "used_reflection": False,
        }

    # 生成反思
    reflection, reflect_usage = generate_reflection(question, gold_answer, first["trajectory"])
    total_tokens += reflect_usage["total_tokens"]

    # 第二轮 ReAct，携带反思
    second = run_react(question, reflection=reflection)
    eval_second = eval_answer(second["answer"], gold_answer)
    total_tokens += second["tokens"]

    return {
        "first_answer": first["answer"],
        "second_answer": second["answer"],
        "reflection": reflection,
        "first_correct": eval_first["loose"],
        "second_correct": eval_second["loose"],
        "tokens": total_tokens,
        "prompt_tokens": first["prompt_tokens"] + reflect_usage["prompt_tokens"] + second["prompt_tokens"],
        "completion_tokens": first["completion_tokens"] + reflect_usage["completion_tokens"] + second["completion_tokens"],
        "used_reflection": True,
    }


# 快速自测
sample_a = test_data[0]["answer"]
ref_res = run_reflexion(sample_q, sample_a)
print("First answer:", ref_res["first_answer"])
print("Second answer:", ref_res["second_answer"])
print("Reflection:", ref_res["reflection"])
print("Second correct:", ref_res["second_correct"])
print("Total tokens:", ref_res["tokens"])
'''

# Cell 7
md7 = '''# Cell 7: 实验执行

- 遍历 50 条测试样本
- 每个样本分别跑 CoT、ReAct、Reflexion
- 记录正确性与 Token 消耗
'''
code7 = '''results = {
    "cot": [],
    "react": [],
    "reflexion": [],
}

for item in tqdm(test_data, desc="Running experiments"):
    q = item["question"]
    a = item["answer"]

    # CoT
    cot = run_cot(q)
    cot_eval = eval_answer(cot["answer"], a)
    results["cot"].append({
        "question": q,
        "gold": a,
        "pred": cot["answer"],
        "correct": cot_eval["loose"],
        "tokens": cot["tokens"],
        "prompt_tokens": cot["prompt_tokens"],
        "completion_tokens": cot["completion_tokens"],
    })

    # ReAct
    react = run_react(q)
    react_eval = eval_answer(react["answer"], a)
    results["react"].append({
        "question": q,
        "gold": a,
        "pred": react["answer"],
        "correct": react_eval["loose"],
        "tokens": react["tokens"],
        "prompt_tokens": react["prompt_tokens"],
        "completion_tokens": react["completion_tokens"],
    })

    # Reflexion
    reflex = run_reflexion(q, a)
    results["reflexion"].append({
        "question": q,
        "gold": a,
        "first_pred": reflex["first_answer"],
        "second_pred": reflex["second_answer"],
        "reflection": reflex["reflection"],
        "first_correct": reflex["first_correct"],
        "correct": reflex["second_correct"],
        "tokens": reflex["tokens"],
        "prompt_tokens": reflex["prompt_tokens"],
        "completion_tokens": reflex["completion_tokens"],
    })

    # 简单防限流
    time.sleep(0.5)

print("\\nExperiment finished.")
'''

# Cell 8
md8 = '''# Cell 8: 结果分析与可视化

- 准确率对比
- Token 消耗对比
- Pareto 曲线
'''
code8 = '''# 汇总统计
def summarize(method: str) -> Dict:
    rows = results[method]
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n * 100
    avg_tokens = sum(r["tokens"] for r in rows) / n
    avg_prompt = sum(r["prompt_tokens"] for r in rows) / n
    avg_completion = sum(r["completion_tokens"] for r in rows) / n
    return {
        "method": method.upper(),
        "accuracy (%)": round(acc, 2),
        "avg_tokens": round(avg_tokens, 1),
        "avg_prompt": round(avg_prompt, 1),
        "avg_completion": round(avg_completion, 1),
    }

summary = [summarize(m) for m in ["cot", "react", "reflexion"]]
import pandas as pd
df_summary = pd.DataFrame(summary)
print(df_summary.to_string(index=False))

# 准确率柱状图
methods = [s["method"] for s in summary]
accs = [s["accuracy (%)"] for s in summary]
avg_tokens = [s["avg_tokens"] for s in summary]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].bar(methods, accs, color=["#4C78A8", "#F58518", "#E45756"])
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Accuracy Comparison")
axes[0].set_ylim(0, 100)
for i, v in enumerate(accs):
    axes[0].text(i, v + 2, f"{v}%", ha="center")

axes[1].bar(methods, avg_tokens, color=["#4C78A8", "#F58518", "#E45756"])
axes[1].set_ylabel("Avg Tokens")
axes[1].set_title("Average Token Consumption")
for i, v in enumerate(avg_tokens):
    axes[1].text(i, v + max(avg_tokens)*0.02, f"{v:.0f}", ha="center")

# Pareto 曲线：准确率 vs Token 消耗
axes[2].scatter(avg_tokens, accs, s=200, c=["#4C78A8", "#F58518", "#E45756"])
for i, m in enumerate(methods):
    axes[2].annotate(m, (avg_tokens[i], accs[i]), textcoords="offset points", xytext=(10, 10), fontsize=11)
axes[2].set_xlabel("Avg Tokens")
axes[2].set_ylabel("Accuracy (%)")
axes[2].set_title("Pareto Frontier: Accuracy vs Tokens")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reflexion_results.png", dpi=150)
plt.show()
'''

# Cell 9
md9 = '''# Cell 9: 案例分析

展示 2-3 个 Reflexion 成功修正错误答案的案例。
'''
code9 = '''success_cases = [
    r for r in results["reflexion"]
    if not r["first_correct"] and r["correct"]
][:3]

for i, case in enumerate(success_cases, 1):
    print(f"===== Case {i} =====")
    print(f"Question: {case['question']}")
    print(f"Gold: {case['gold']}")
    print(f"First (wrong): {case['first_pred']}")
    print(f"Reflection: {case['reflection']}")
    print(f"Second (correct): {case['second_pred']}")
    print(f"Tokens used: {case['tokens']}\\n")

if not success_cases:
    print("No successful Reflexion cases found in this sample set.")
'''

# Build notebook
nb = new_notebook()
for md, code in [(md1, code1), (md2, code2), (md3, code3), (md4, code4), (md5, code5), (md6, code6), (md7, code7), (md8, code8), (md9, code9)]:
    nb.cells.append(new_markdown_cell(md))
    nb.cells.append(new_code_cell(code))

with open("/Users/mingrun/reflexion_repro/reflexion_reproduction.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook created: /Users/mingrun/reflexion_repro/reflexion_reproduction.ipynb")
