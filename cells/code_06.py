REFLECT_SYSTEM = '''You are a reflection assistant.
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
    """把轨迹序列转换成文本，用于反思 prompt。"""
    lines = []
    for rec in trajectory:
        lines.append(f"Step {rec['step']}:")
        lines.append(rec["llm_output"])
        if rec.get("action"):
            lines.append(f"Action: {rec['action']}")
        if rec.get("observation"):
            lines.append(f"Observation: {rec['observation']}")
        lines.append("")
    return "\n".join(lines)


def generate_reflection(question: str, gold_answer: str, trajectory: List[Dict]) -> Tuple[str, Dict]:
    """生成语言化反思。"""
    traj_text = format_trajectory(trajectory)
    user_prompt = REFLECT_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        trajectory_text=traj_text,
    )
    reflection, usage = call_llm(REFLECT_SYSTEM, user_prompt)
    return reflection, usage


def run_reflexion(question: str, gold_answer: str) -> Dict:
    """执行 Reflexion 范式：ReAct → 评估 → 反思 → ReAct(带反思)。"""
    # 第一轮 ReAct
    first = run_react(question)
    eval_first = eval_answer(first["answer"], gold_answer)

    # 默认：第一轮已正确，无需反思
    if eval_first["em"]:
        return {
            "first_answer": first["answer"],
            "second_answer": first["answer"],
            "reflection": "",
            "first_eval": eval_first,
            "second_eval": eval_first,
            "first_correct": eval_first["em"],
            "second_correct": eval_first["em"],
            "tokens": first["tokens"],
            "prompt_tokens": first["prompt_tokens"],
            "completion_tokens": first["completion_tokens"],
            # Token 拆分
            "first_tokens": first["tokens"],
            "reflect_tokens": 0,
            "second_tokens": 0,
            "used_reflection": False,
        }

    # 生成反思
    reflection, reflect_usage = generate_reflection(question, gold_answer, first["trajectory"])

    # 第二轮 ReAct，携带反思
    second = run_react(question, reflection=reflection)
    eval_second = eval_answer(second["answer"], gold_answer)

    return {
        "first_answer": first["answer"],
        "second_answer": second["answer"],
        "reflection": reflection,
        "first_eval": eval_first,
        "second_eval": eval_second,
        "first_correct": eval_first["em"],
        "second_correct": eval_second["em"],
        "tokens": first["tokens"] + reflect_usage["total_tokens"] + second["tokens"],
        "prompt_tokens": first["prompt_tokens"] + reflect_usage["prompt_tokens"] + second["prompt_tokens"],
        "completion_tokens": first["completion_tokens"] + reflect_usage["completion_tokens"] + second["completion_tokens"],
        "first_tokens": first["tokens"],
        "reflect_tokens": reflect_usage["total_tokens"],
        "second_tokens": second["tokens"],
        "used_reflection": True,
    }


# # 快速自测
# sample_a = test_data[0]["answer"]
# ref_res = run_reflexion(sample_q, sample_a)
# print("First answer:", ref_res["first_answer"])
# print("Second answer:", ref_res["second_answer"])
# print("Reflection:", ref_res["reflection"])
# print("Second correct:", ref_res["second_correct"])
# print("Total tokens:", ref_res["tokens"])
