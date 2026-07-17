# 汇总统计：EM + F1 + Token
def summarize(method: str) -> Dict:
    rows = results[method]
    n = len(rows)
    em_acc = sum(r["em"] for r in rows) / n * 100
    avg_f1 = sum(r["f1"] for r in rows) / n * 100
    avg_tokens = sum(r["tokens"] for r in rows) / n
    avg_prompt = sum(r["prompt_tokens"] for r in rows) / n
    avg_completion = sum(r["completion_tokens"] for r in rows) / n
    return {
        "method": method.upper(),
        "EM (%)": round(em_acc, 2),
        "F1 (%)": round(avg_f1, 2),
        "avg_tokens": round(avg_tokens, 1),
        "avg_prompt": round(avg_prompt, 1),
        "avg_completion": round(avg_completion, 1),
    }

summary = [summarize(m) for m in ["cot", "react", "reflexion"]]
import pandas as pd
df_summary = pd.DataFrame(summary)
print("=" * 60)
print("EM + F1 对比表")
print("=" * 60)
print(df_summary.to_string(index=False))

# Reflexion Token 拆分
reflex_rows = results["reflexion"]
print("\n" + "=" * 60)
print("Reflexion Token 拆分")
print("=" * 60)
print(f"avg_first_tokens:    {sum(r['first_tokens'] for r in reflex_rows) / len(reflex_rows):.1f}")
print(f"avg_reflect_tokens:  {sum(r['reflect_tokens'] for r in reflex_rows) / len(reflex_rows):.1f}")
print(f"avg_second_tokens:   {sum(r['second_tokens'] for r in reflex_rows) / len(reflex_rows):.1f}")

# 429 影响统计
print("\n" + "=" * 60)
print("Wikipedia 429 影响统计")
print("=" * 60)
for method in ["react", "reflexion"]:
    rows = results[method]
    affected = [r for r in rows if r.get("wiki_429_affected")]
    unaffected = [r for r in rows if not r.get("wiki_429_affected")]
    aff_em = sum(r["em"] for r in affected) / len(affected) * 100 if affected else 0
    unaff_em = sum(r["em"] for r in unaffected) / len(unaffected) * 100 if unaffected else 0
    print(f"{method.upper():10s} 受影响 {len(affected):2d}/{len(rows):2d}  |  受影响 EM: {aff_em:5.2f}%  |  未受影响 EM: {unaff_em:5.2f}%")

# 答案类型 vs 准确率
print("\n" + "=" * 60)
print("答案类型 vs EM 准确率")
print("=" * 60)
rows_by_type = {}
for r in results["cot"]:
    rows_by_type.setdefault(r["answer_type"], {"cot": [], "react": [], "reflexion": []})
    rows_by_type[r["answer_type"]]["cot"].append(r)
for r in results["react"]:
    rows_by_type.setdefault(r["answer_type"], {"cot": [], "react": [], "reflexion": []})
    rows_by_type[r["answer_type"]]["react"].append(r)
for r in results["reflexion"]:
    rows_by_type.setdefault(r["answer_type"], {"cot": [], "react": [], "reflexion": []})
    rows_by_type[r["answer_type"]]["reflexion"].append(r)

type_rows = []
for typ, m in sorted(rows_by_type.items()):
    type_rows.append({
        "answer_type": typ,
        "count": len(m["cot"]),
        "CoT_EM": f"{sum(r['em'] for r in m['cot'])/len(m['cot'])*100:.1f}%",
        "ReAct_EM": f"{sum(r['em'] for r in m['react'])/len(m['react'])*100:.1f}%",
        "Reflexion_EM": f"{sum(r['em'] for r in m['reflexion'])/len(m['reflexion'])*100:.1f}%",
    })
df_type = pd.DataFrame(type_rows)
print(df_type.to_string(index=False))

# 可视化
methods = [s["method"] for s in summary]
em_accs = [s["EM (%)"] for s in summary]
f1_scores = [s["F1 (%)"] for s in summary]
avg_tokens = [s["avg_tokens"] for s in summary]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# EM 柱状图
axes[0, 0].bar(methods, em_accs, color=["#4C78A8", "#F58518", "#E45756"])
axes[0, 0].set_ylabel("EM (%)")
axes[0, 0].set_title("Exact Match Accuracy")
axes[0, 0].set_ylim(0, 100)
for i, v in enumerate(em_accs):
    axes[0, 0].text(i, v + 2, f"{v:.1f}%", ha="center")

# F1 柱状图
axes[0, 1].bar(methods, f1_scores, color=["#4C78A8", "#F58518", "#E45756"])
axes[0, 1].set_ylabel("F1 (%)")
axes[0, 1].set_title("Average F1 Score")
axes[0, 1].set_ylim(0, 100)
for i, v in enumerate(f1_scores):
    axes[0, 1].text(i, v + 2, f"{v:.1f}%", ha="center")

# Token 消耗
axes[1, 0].bar(methods, avg_tokens, color=["#4C78A8", "#F58518", "#E45756"])
axes[1, 0].set_ylabel("Avg Tokens")
axes[1, 0].set_title("Average Token Consumption")
for i, v in enumerate(avg_tokens):
    axes[1, 0].text(i, v + max(avg_tokens)*0.02, f"{v:.0f}", ha="center")

# Pareto 曲线：EM vs Tokens
axes[1, 1].scatter(avg_tokens, em_accs, s=200, c=["#4C78A8", "#F58518", "#E45756"])
for i, m in enumerate(methods):
    axes[1, 1].annotate(m, (avg_tokens[i], em_accs[i]), textcoords="offset points", xytext=(10, 10), fontsize=11)
axes[1, 1].set_xlabel("Avg Tokens")
axes[1, 1].set_ylabel("EM (%)")
axes[1, 1].set_title("Pareto Frontier: EM vs Tokens")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reflexion_results.png", dpi=150)
plt.show()
