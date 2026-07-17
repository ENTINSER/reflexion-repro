COT_SYSTEM = '''You are a helpful question-answering assistant.
Answer the following multi-hop question by reasoning step by step.
End your response with a line that starts with "Answer:" followed by the final answer.'''

COT_TEMPLATE = '''Question: {question}

Let's think step by step.'''


def run_cot(question: str) -> Dict:
    """执行 CoT 范式。"""
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


# # 快速自测
# sample_q = test_data[0]["question"]
# print("Sample question:", sample_q)
# cot_res = run_cot(sample_q)
# print("\nCoT raw response (truncated):")
# print(textwrap.shorten(cot_res["raw"], width=500))
# print("\nExtracted answer:", cot_res["answer"])
# print("Tokens:", cot_res["tokens"])