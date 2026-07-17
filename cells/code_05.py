REACT_SYSTEM = '''You are a question-answering agent that solves multi-hop questions by interleaving Thought, Action, and Observation.

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
    """从模型输出解析 Action 类型与参数。"""
    # 找到最后一个 Thought/Action 行
    action_match = re.search(r"Action:\s*(Search|Finish)\[(.*?)\]", text, re.IGNORECASE | re.DOTALL)
    if not action_match:
        return "None", ""
    act_type = action_match.group(1).strip()
    act_arg = action_match.group(2).strip()
    return act_type, act_arg


def run_react(question: str, reflection: Optional[str] = None) -> Dict:
    """执行 ReAct 范式；若提供 reflection，则将其拼入初始 prompt。"""
    scratchpad = ""
    if reflection:
        scratchpad += f"Previous reflection: {reflection}\n\n"

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
            scratchpad += f"Thought: {response}\nAction: Search[{act_arg}]\nObservation: {obs}\n\n"
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


# # 快速自测
# print("Sample question:", sample_q)
# react_res = run_react(sample_q)
# print("\nReAct answer:", react_res["answer"])
# print("Finished:", react_res["finished"], "Steps:", react_res["steps"])
# print("Tokens:", react_res["tokens"])
# print("\nTrajectory (first step):")
# if react_res["trajectory"]:
#     print(json.dumps(react_res["trajectory"][0], ensure_ascii=False, indent=2))