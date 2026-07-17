# ---------------------------
# Wikipedia 工具（带 429 重试与查询日志）
# ---------------------------
WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "ReflexionReproBot/1.0 (research@example.com)"}

# 全局查询日志：记录每次 Wikipedia API 调用的状态
wiki_query_log: List[Dict] = []


def _wiki_request_with_retry(params: Dict, call_type: str = "search") -> Tuple[Dict, str]:
    """
    带 429 指数退避重试的 Wikipedia API 请求。
    返回 (json_data, status)，status 取值：
    'success', 'retry_success_429', 'failed_429', 'empty', 'error'
    """
    max_retries = 3
    base_wait = 2
    status = "success"

    for attempt in range(max_retries + 1):
        try:
            r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=8)
            r.raise_for_status()
            data = r.json()
            # 成功调用后休息 1 秒，降低触发 429 的概率
            time.sleep(1)
            return data, status
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                status = "retry_success_429" if attempt < max_retries else "failed_429"
                if attempt < max_retries:
                    wait = base_wait * (2 ** attempt)
                    print(f"[Wiki 429] {call_type} got 429, retry in {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                else:
                    print(f"[Wiki 429] {call_type} failed after {max_retries} retries")
                    return {}, status
            else:
                print(f"[Wiki {call_type} HTTP error] {e}")
                return {}, "error"
        except Exception as e:
            print(f"[Wiki {call_type} error] {e}")
            return {}, "error"

    return {}, status


def wiki_search(query: str, top_n: int = TOP_N_SEARCH) -> List[str]:
    """返回与 query 最相关的 Wikipedia 页面标题列表。"""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": top_n,
    }
    data, status = _wiki_request_with_retry(params, call_type="search")
    try:
        results = data.get("query", {}).get("search", [])
        titles = [x["title"] for x in results]
        wiki_query_log.append({"type": "search", "query": query, "status": status, "results": len(titles)})
        return titles
    except Exception as e:
        wiki_query_log.append({"type": "search", "query": query, "status": "error", "results": 0})
        print(f"[Wiki search parse error] {e}")
        return []


def wiki_summary(title: str, sentences: int = 5) -> str:
    """获取指定 Wikipedia 页面的前 sentences 句摘要。"""
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "explaintext": True,
        "exsentences": sentences,
        "format": "json",
    }
    data, status = _wiki_request_with_retry(params, call_type="summary")
    try:
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract:
                wiki_query_log.append({"type": "summary", "title": title, "status": status, "length": len(extract)})
                return extract.strip()
        wiki_query_log.append({"type": "summary", "title": title, "status": "empty", "length": 0})
        return ""
    except Exception as e:
        wiki_query_log.append({"type": "summary", "title": title, "status": "error", "length": 0})
        print(f"[Wiki summary parse error] {e}")
        return ""


def get_wiki_query_log() -> List[Dict]:
    """返回 Wikipedia 查询日志。"""
    return wiki_query_log


def reset_wiki_query_log() -> None:
    """清空查询日志，用于每个样本开始前重置。"""
    global wiki_query_log
    wiki_query_log = []


# ---------------------------
# LLM 调用 + Token 统计
# ---------------------------
def call_llm(system: str, user: str, temperature: float = TEMPERATURE, **kwargs) -> Tuple[str, Dict]:
    """调用 LLM，返回 (content, usage_dict)。"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            timeout=90,
            max_completion_tokens=2048,
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


# ---------------------------
# HotPotQA 官方风格评估：EM + F1
# ---------------------------
def normalize_answer(s: str) -> str:
    """归一化：小写、去冠词、去标点、去多余空格。"""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def f1_score(pred: str, gold: str) -> float:
    """基于归一化后 token 的 F1 分数。"""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = sum((pred_tokens.count(t) for t in set(gold_tokens) if t in pred_tokens))
    # 更准确的 token 交集：按 multiset
    pred_counter = {}
    for t in pred_tokens:
        pred_counter[t] = pred_counter.get(t, 0) + 1
    gold_counter = {}
    for t in gold_tokens:
        gold_counter[t] = gold_counter.get(t, 0) + 1
    common = sum(min(pred_counter.get(t, 0), gold_counter.get(t, 0)) for t in set(gold_tokens))

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def eval_answer(pred: str, gold: str) -> Dict:
    """返回 EM 和 F1。"""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    em = pred_norm == gold_norm
    f1 = f1_score(pred, gold)
    return {"em": em, "f1": f1}


def classify_answer_type(answer: str) -> str:
    """简单启发式判断答案类型：date / number / person / location / other。"""
    s = answer.strip()
    lower = s.lower()

    # 纯数字（允许逗号、小数、负数）
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", s.replace(",", "")):
        return "number"

    # 包含年份或月份
    if re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", s) or \
       re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", lower):
        return "date"

    # 地点线索
    location_keywords = [
        "city", "town", "village", "country", "state", "river", "mountain", "lake",
        "island", "peninsula", "valley", "desert", "forest", "ocean", "sea", "bay",
        "park", "university", "school", "hospital", "airport", "station"
    ]
    if any(kw in lower for kw in location_keywords):
        return "location"

    # 人名线索（简单规则：含空格的首字母大写串，或常见职业/称谓）
    person_keywords = ["mr.", "mrs.", "dr.", "professor", "actor", "director", "author",
                       "musician", "singer", "writer", "politician", "scientist", "player"]
    if any(kw in lower for kw in person_keywords):
        return "person"
    # 两个或以上首字母大写的词
    if len(re.findall(r"\b[A-Z][a-z]+\b", s)) >= 2:
        return "person"

    return "other"


def extract_boxed(text: str) -> str:
    """尝试提取 \boxed{} 或最后的 Answer: 后的内容。"""
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    lines = text.split("\n")
    for line in reversed(lines):
        if "answer:" in line.lower():
            return line.split(":", 1)[1].strip()
    return text.strip()


print("Tool functions ready.")
print("Wiki search test:", wiki_search("Barack Obama", top_n=1)[:1])
