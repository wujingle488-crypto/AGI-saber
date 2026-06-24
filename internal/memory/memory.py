# memory — 三层记忆系统（短期 / 长期 / 用户偏好）
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from config.config import APIConfig
from internal.infra.infra import Infrastructure

if TYPE_CHECKING:
    from internal.memory.graph_memory import GraphMemory  # noqa: F401

logger = logging.getLogger(__name__)

@dataclass
class Item:
    """长期记忆的基本单元"""
    content: str  # 记忆内容文本
    importance: float = 0.5  # 重要性分数（0~1），用于长期记忆的优先级
    embedding: Optional[List[float]] = None  # 向量表示（可选）
    id: Optional[int] = None  # 全局唯一 ID（未设置时由 GraphMemory 用 content hash 派生）

# 滑动窗口，保留最近N轮对话
class ShortTerm:
    """短期记忆 - 滑动窗口存储最近 N 轮对话。"""
    def __init__(self, max_turns: int = 10):
        self.max_turns = max(1 , max_turns) # 至少 1 轮 2条
        self.messages: List[Dict[str,str]] = [] # role + content 的字典列表

    def add(self, role: str, content: str):
        self.messages.append({"role":role,"content":content})
        while len(self.messages) > self.max_turns * 2:
            self.messages.pop(0) # 删除列表开头最老的消息

    def get(self) -> List[Dict[str,str]]:
        return self.messages.copy() # 返回消息列表的副本，防止外界修改

    def clear(self):
        self.messages = []

    def count(self) -> int:
        return len(self.messages)

# 用于长期记忆返回两条相似的文本后检测只保存一条
def _tokenize_zh(text: str) -> List[str]:
    """中英文混合分词：中文按字、英文/数字按词。"""
    tokens: List[str] = []
    word = ""
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            if word:
                tokens.append(word.lower())
                word = ""
            tokens.append(ch)
        elif ch.isalnum():
            word += ch
        else:
            if word:
                tokens.append(word.lower())
                word = ""
    if word:
        tokens.append(word.lower())
    return tokens

# 长期记忆 通过Embedding模型嵌入为向量，按语义保留
# 重要度 + 衰减
class LongTerm:
    """长期记忆 - 基于 embedding 的语义记忆。"""

    def __init__(self, cfg: APIConfig, inf: Infrastructure):
        self.cfg = cfg
        self.inf = inf
        self.items: List[Item] = []  # 所有长期记忆的列表
        self._embed_fn: Optional[Any] = None  # embedding 函数（外部注入）
        self._last_consolidate_ts = 0.0  # 上次合并的时间戳
        self._items_since_last = 0  # 上次合并后新增的记忆条数
        self.graph_memory: Optional["GraphMemory"] = None  # 图增强（可选）
        self._next_id: int = 0  # 下一个可用的 ID

    def set_embed_fn(self, fn):
        self._embed_fn = fn

    def set_graph_memory(self, graph_memory: Optional["GraphMemory"]) -> None:
        """注入图增强记忆层；可在任意时刻调用，None 表示解除注入。"""
        self.graph_memory = graph_memory

    # 从存储中取出向量
    def load_from_storage(self):
        rows = self.inf.load_long_term_items()
        self.items = [Item(content=r.content, importance=r.importance, embedding=r.embedding) for r in rows]

        for idx, item in enumerate(self.items):
            item.id = idx
        self._next_id = len(self.items)
        logger.info("✅ 从存储恢复了 %d 条长期记忆", len(self.items))
        # 图增强记忆 hook：批量索引（启动期把 LTM 全量同步进图）
        if self.graph_memory is not None:
            try:
                self.graph_memory.bulk_index(self.items)
            except Exception as e:
                logger.warning("⚠️  graph_memory.bulk_index 失败: %s", e)

    def add(self, content: str, importance: float = 0.5):
        embedding = None
        if self._embed_fn:
            try:
                embedding = self._embed_fn(content)
            except Exception as e:
                logger.warning("⚠️  向量化失败: %s", e)

        item = Item(content=content, importance=importance, embedding=embedding, id=self._next_id)
        self._next_id += 1
        # 旧条目快照（add_to_graph 内会扫描这些建立 SIMILAR_TO 边）
        prior = list(self.items)
        self.items.append(item)
        self._items_since_last += 1
        emb_json = json.dumps(embedding) if embedding else "null"
        self.inf.save_long_term_item(content, importance, emb_json)
        # 图增强记忆 hook：新增条目同步进图
        if self.graph_memory is not None:
            try:
                self.graph_memory.add_to_graph(item, neighbors=prior[-50:])
            except Exception as e:
                logger.warning("⚠️  graph_memory.add_to_graph 失败: %s", e)

    def recall(self, query: str, top_k: int = 3) -> List[Item]:
        if not self.items:
            return []
        query_emb = None
        if self._embed_fn:
            try:
                query_emb = self._embed_fn(query)
            except Exception as e:
                logger.warning("⚠️  查询向量化失败: %s", e)
                query_emb = None
        if not query_emb:
            return self.items[:top_k]
        scored: List[tuple] = []
        for item in self.items:
            if item.embedding:
                sim = self._cosine_similarity(query_emb, item.embedding)
                score = sim * 0.7 + item.importance * 0.3
                scored.append((item, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, score in scored[:top_k] if score >= 0.4]

    def filter_by_category(self, categories: List[str], limit: int) -> List[Item]:
        """PromptContext 适配：当前 Item 尚无 category 字段，按重要性返回前 N 条。"""
        if limit <= 0:
            limit = 10
        return sorted(self.items, key=lambda item: item.importance, reverse=True)[:limit]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def need_consolidation(self) -> bool:
        return self._items_since_last >= max(1, self.cfg.memory_consolidation_trigger)

    def consolidate(self):
        """合并/去重 + 衰减 + TTL 淘汰。仅在 need_consolidation 为真时调用。"""
        if not self.items:
            return

        logger.info("🔄 开始记忆合并...")
        now = time.time()
        elapsed_days = (now - self._last_consolidate_ts) / 86400.0 if self._last_consolidate_ts > 0 else 1.0
        self._last_consolidate_ts = now
        self._items_since_last = 0
        # 1) 重复项合并：保留 importance 较高者
        to_remove = set()
        for i in range(len(self.items)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(self.items)):
                if j in to_remove:
                    continue
                sim = self._compute_similarity(self.items[i].content, self.items[j].content)
                # 余弦相似度达到阈值才保留
                if sim >= self.cfg.memory_consolidation_dedup:
                    # 保留重要性更高的
                    drop = i if self.items[i].importance < self.items[j].importance else j
                    to_remove.add(drop)
        # 收集被去重淘汰的 id（用于图同步删除）
        removed_ids: List[int] = [self.items[i].id for i in to_remove if self.items[i].id is not None]
        self.items = [item for i, item in enumerate(self.items) if i not in to_remove]

        # 2) 按时间间隔衰减（避免每轮对话都被衰减一次）
        decay = self.cfg.memory_consolidation_decay_rate ** max(elapsed_days, 0.0)
        for item in self.items:
            item.importance *= decay

        # 3) TTL 淘汰
        ttl_min = self.cfg.memory_consolidation_min_import
        kept: List[Item] = []
        for item in self.items:
            if item.importance >= ttl_min:
                kept.append(item)
            elif item.id is not None:
                removed_ids.append(item.id)
        self.items = kept
        logger.info("✅ 记忆合并完成，剩余 %d 条", len(self.items))

        # 图增强记忆 hook：被淘汰的条目同步从图中删除；保留的条目更新 importance
        if self.graph_memory is not None:
            try:
                for rid in removed_ids:
                    self.graph_memory.delete_from_graph(rid)
                for item in self.items:
                    self.graph_memory.update_node(item)
            except Exception as e:
                logger.warning("⚠️  graph_memory consolidate hook 失败: %s", e)

    # 根据_tokenize_zh分词结果计算相似度
    def _compute_similarity(self, a: str, b: str) -> float:
        """中英文 Jaccard 相似度。"""
        tokens_a = set(_tokenize_zh(a))
        tokens_b = set(_tokenize_zh(b))
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

class Preference:
    """用户偏好管理。"""

    def __init__(self, user_id: str, inf: Infrastructure):
        self.user_id = user_id
        self.inf = inf
        self.preferences: Dict[str, str] = {}
        self.load_from_storage()

    @property
    def data(self) -> Dict[str, str]:
        return self.preferences

    def load_from_storage(self):
        self.preferences = self.inf.load_preferences(self.user_id)
        logger.info("✅ 加载用户 %s 的偏好: %s", self.user_id, self.preferences)

    def set(self, key: str, value: str):
        if not key or value is None:
            return
        self.preferences[key] = value
        self.inf.save_preference(self.user_id, key, value)

    def save_batch(self, kvs: Dict[str, str]):
        for k, v in kvs.items():
            self.set(str(k), str(v))

    def get(self, key: str, default: str = "") -> str:
        return self.preferences.get(key, default)

    def get_all(self) -> Dict[str, str]:
        return self.preferences.copy()

    def snapshot(self) -> Dict[str, str]:
        return self.get_all()

class MemoryManager:
    """三层记忆 + 可选 GraphMemory 的统一容器。

    现有 ShortTerm/LongTerm/Preference 行为完全保留；GraphMemory 作为可选注入点：
      - 存入新长期记忆 → 自动调用 graph_memory.add_to_graph(item)
      - 删除/更新长期记忆（consolidate）→ 自动调用 delete_from_graph / update_node

    所有 hook 实际位于 LongTerm 内部；MemoryManager 负责装配并把 graph_memory
    注入到 LongTerm 上。Neo4j 不可用或未注入时，所有 hook 都是 no-op。
    """
    def __init__(
        self,
        cfg: APIConfig,
        inf: Infrastructure,
        user_id: str = "default_user",
        graph_memory: Optional["GraphMemory"] = None,
    ):
        self.cfg = cfg
        self.inf = inf
        self.short_term = ShortTerm(cfg.short_term_max_turns)
        self.long_term = LongTerm(cfg, inf)
        self.preference = Preference(user_id, inf)
        self.graph_memory: Optional["GraphMemory"] = graph_memory
        if graph_memory is not None:
            self.long_term.set_graph_memory(graph_memory)

    def set_graph_memory(self, graph_memory: Optional["GraphMemory"]) -> None:
        """挂载 / 解除 GraphMemory；同步透传到 LongTerm。"""
        self.graph_memory = graph_memory
        self.long_term.set_graph_memory(graph_memory)

    def add_long_term(self, content: str, importance: float = 0.5) -> None:
        """存入新长期记忆；命中 LongTerm.add hook 自动同步图。"""
        self.long_term.add(content, importance)

    def consolidate(self) -> None:
        """触发长期记忆 consolidate；命中 LongTerm.consolidate hook 自动同步图。"""
        self.long_term.consolidate()

    def recall(self, query: str, top_k: int = 3) -> List[Item]:
        """语义召回；如果挂载了 graph_memory，则在向量召回基础上做一跳图扩展。"""
        seed = self.long_term.recall(query, top_k)
        if not self.graph_memory or not seed:
            return seed
        seed_ids = [it.id for it in seed if it.id is not None]
        if not seed_ids:
            return seed
        try:
            expanded_ids = set()
            for sid in seed_ids:
                expanded_ids.update(self.graph_memory.find_related(sid))
        except Exception as e:
            logger.warning("⚠️  graph_memory.find_related 失败: %s", e)
            return seed
        seed_id_set = set(seed_ids)
        extras: List[Item] = []
        for it in self.long_term.items:
            if it.id in expanded_ids and it.id not in seed_id_set:
                extras.append(it)
        return seed + extras

    def need_consolidation(self) -> bool:
        return self.long_term.need_consolidation()

































