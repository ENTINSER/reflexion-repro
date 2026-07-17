results = {
    "cot": [],
    "react": [],
    "reflexion": [],
}

for item in tqdm(test_data, desc="Running experiments"):
    q = item["question"]
    a = item["answer"]
    q_len = len(q.split())
    ans_type = classify_answer_type(a)

    # 记录本样本的 Wikipedia 查询日志
    reset_wiki_query_log()

    # CoT
    cot = run_cot(q)
    cot_eval = eval_answer(cot["answer"], a)
    results["cot"].append({
        "question": q,
        "gold": a,
        "pred": cot["answer"],
        "em": cot_eval["em"],
        "f1": cot_eval["f1"],
        "tokens": cot["tokens"],
        "prompt_tokens": cot["prompt_tokens"],
        "completion_tokens": cot["completion_tokens"],
        "question_len": q_len,
        "answer_type": ans_type,
    })

    # ReAct
    react = run_react(q)
    react_eval = eval_answer(react["answer"], a)
    react_log = get_wiki_query_log()
    react_429 = any("429" in str(e.get("status", "")) for e in react_log)
    react_empty_obs = sum(1 for step in react.get("trajectory", []) if not step.get("observation"))
    results["react"].append({
        "question": q,
        "gold": a,
        "pred": react["answer"],
        "em": react_eval["em"],
        "f1": react_eval["f1"],
        "tokens": react["tokens"],
        "prompt_tokens": react["prompt_tokens"],
        "completion_tokens": react["completion_tokens"],
        "question_len": q_len,
        "answer_type": ans_type,
        "wiki_429_affected": react_429,
        "wiki_empty_obs": react_empty_obs,
    })

    # Reflexion
    reflex = run_reflexion(q, a)
    reflex_log = get_wiki_query_log()
    reflex_429 = any("429" in str(e.get("status", "")) for e in reflex_log)
    reflex_empty_obs = sum(1 for step in reflex.get("trajectory", []) if not step.get("observation"))
    results["reflexion"].append({
        "question": q,
        "gold": a,
        "first_pred": reflex["first_answer"],
        "second_pred": reflex["second_answer"],
        "reflection": reflex["reflection"],
        "first_em": reflex["first_eval"]["em"],
        "first_f1": reflex["first_eval"]["f1"],
        "second_em": reflex["second_eval"]["em"],
        "second_f1": reflex["second_eval"]["f1"],
        "em": reflex["second_eval"]["em"],
        "f1": reflex["second_eval"]["f1"],
        "tokens": reflex["tokens"],
        "prompt_tokens": reflex["prompt_tokens"],
        "completion_tokens": reflex["completion_tokens"],
        # Reflexion 专用 token 拆分
        "first_tokens": reflex["first_tokens"],
        "reflect_tokens": reflex["reflect_tokens"],
        "second_tokens": reflex["second_tokens"],
        "used_reflection": reflex["used_reflection"],
        "question_len": q_len,
        "answer_type": ans_type,
        "wiki_429_affected": reflex_429,
        "wiki_empty_obs": reflex_empty_obs,
    })

    # 简单防限流
    time.sleep(0.5)

print("\nExperiment finished.")
