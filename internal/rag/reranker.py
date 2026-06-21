import json
import logging
import re
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# 两个参数，系统提示词，传递过来的chunk 最终返回LLM重排后的chunk
GenerateFn = Callable[[str, str], str]

class LLMReranker:
    """用一次 LLM listwise 调用对候选 chunk 精排，失败时回退原顺序。"""
    def __init__(self, generate_fn: Optional[GenerateFn], preview_len: int = 200):
        self.generate_fn = generate_fn
        self.preview_len = preview_len if preview_len > 0 else 200

    def rerank(self, query: str,results: List, top_k: int) -> List:
        if not results:
            return []

        if self.generate_fn is None or len(results) == 1:
            return _truncate(results, top_k)

        try:
            raw = self.generate_fn(self._system_prompt(), self._user_msg(query, results))
            scores = _parse_scores(raw)
        except Exception as e:
            logger.warning("⚠️  Rerank 失败，回退 RRF 顺序: %s", e)
            return _truncate(results, top_k)

        if not scores:
            return _truncate(results, top_k)

        score_map = {idx: score for idx, score in scores if 0 <= idx < len(results)}
        ordered = []
        for idx, result in enumerate(results):
            llm_score = score_map.get(idx, -1)
            ordered.append((llm_score,getattr(result, "score", 0.0), idx, result))
        ordered.sort(key=lambda item: (item[0], item[1], -item[2]),reverse=True)

        out = []
        for llm_score, score, idx, result in ordered:
            if llm_score >= 0:
                result.score = llm_score / 10.0
            result.source = f"{getattr(result, 'source', '')}+rerank"
            out.append(result)
        return _truncate(out, top_k)


    def _system_prompt(self) -> str:
        return (
            "你是检索系统的精排器。给定用户问题和候选段落，"
            "只输出严格 JSON：{\"scores\": [{\"idx\": 0, \"score\": 9}]}。"
            "score 为 0 到 10。"
        )

    def _user_msg(self, query: str, results: List) -> str:
        lines = [f"用户问题：{query}", "", "候选段落："]
        for idx, result in enumerate(results):
            content = _result_content(result)
            if len(content) > self.preview_len:
                content = content[:self.preview_len] + "..."
            lines.append(f"[{idx}]{content}")
        # 将lines中所有元素用换行符拼接成大的字符串
        return "\n".join(lines)
    """
    用户问题：苹果有什么营养价值？

    候选段落：
    [0] 苹果富含维生素C，每100克含约4毫克...
    [1] 苹果是水果的一种，原产于欧洲及亚...
    [2] 香蕉主要产自热带地区，富含钾元素...
    """

def _result_content(result) -> str:
    if hasattr(result, "content"):
        return str(result.content)
    chunk = getattr(result, "chunk", None)
    if chunk is not None and hasattr(chunk, "content"):
        return str(chunk.content)
    return ""

def _parse_scores(raw: str) -> List[tuple]:
    raw = _strip_json_fence(raw)
    data = json.loads(raw)
    items = data.get("scores", []) if isinstance(data, dict) else []
    scores = []
    for item in items:
        try:
            scores.append((int(item.get("idx")), float(item.get("score"))))
        except Exception:
            continue
    return scores

def _strip_json_fence(raw: str) -> str:
    raw = (raw or "").strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()

def _truncate(results: List, top_k: int) -> List:
    if top_k > 0 and len(results) > top_k:
        return results[:top_k]
    return results


