# 如需安装依赖，取消下面注释并运行一次
# !pip install openai requests wikipedia datasets matplotlib tqdm pandas -q

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
# export KIMI_BASE_URL="https://api.kimi.com/coding/v1"   # Kimi Code 兼容接口
# export MODEL_NAME="kimi-k2p5-coding"                  # 或 "gpt-4o" 等

API_KEY = os.environ.get("KIMI_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("KIMI_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://api.kimi.com/coding/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "kimi-k2p5-coding")

if not API_KEY:
    raise ValueError("请先设置 KIMI_API_KEY 或 OPENAI_API_KEY 环境变量")

from openai import OpenAI
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 实验超参
TEMPERATURE = 1.0          # 尽量确定性，便于复现
MAX_REACT_STEPS = 8        # ReAct 最大步数
TOP_N_SEARCH = 3           # Wiki 搜索返回候选数

print(f"Model: {MODEL_NAME}")
print(f"Base URL: {BASE_URL}")