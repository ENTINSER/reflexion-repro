# 加载 HotPotQA
# 优先读取本地 hotpot_dev_fullwiki_v1.json，否则尝试 HuggingFace
import os
from pathlib import Path

local_candidates = [
    os.environ.get("HOTPOT_LOCAL_PATH"),
    "hotpot_dev_fullwiki_v1.json",
    "/Users/mingrun/reflexion_repro/hotpot_dev_fullwiki_v1.json",
]
local_path = None
for c in local_candidates:
    if c and Path(c).exists():
        local_path = c
        break

if local_path:
    print(f"Loading local HotPotQA from {local_path}")
    with open(local_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    dataset = raw_data
else:
    # 从 HuggingFace 下载（需要联网）
    print("No local HotPotQA file found. Trying HuggingFace...")
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
print("\nExample:")
print(json.dumps(test_data[0], ensure_ascii=False, indent=2))
