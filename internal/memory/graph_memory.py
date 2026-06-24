# graph_memory — 长期记忆与 Neo4j 知识图谱的双向同步层。
#
# 节点类型：(:Memory {mem_id, content, importance})
# 边类型：
#   FOLLOWS      — 时序相邻（上一条对话记忆 → 当前）
#   SIMILAR_TO   — 语义相似度超阈值（Store 时自动连接）
#   CAUSES       — 因果推断（LLM 提取，可选）
#   BELONGS_TO   — 话题归属
#
# 与 Go 版本 graph_memory.go 对齐；cypher 字符串直接照抄。
# 失败降级：Neo4j 不可用时所有方法都是 no-op，不抛异常。
import logging
import math
from typing import Any, Iterable, List, Optional

from config.config import APIConfig

logger = logging.getLogger(__name__)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class GraphMemory:
    """长期记忆的图增强层；持有 Neo4jClient 直接操作 :Memory 节点和边。

    Neo4jClient 不可用时所有方法静默返回，调用方无需关心降级。
    """

    def __init__(
        self,
        cfg: APIConfig,
        neo,  # internal.platform.neo4j.Neo4jClient
        llm: Optional[Any] = None,
        sim_threshold: float = 0.7,
    ):
        self.cfg = cfg
        self.neo = neo
        self.llm = llm
        self.sim_thresh = sim_threshold if sim_threshold > 0 else 0.7
        self.prev_id: int = -1

    # ─── 可用性 ────────────────────────────────────────────────────────────
    def _available(self) -> bool:
        return self.neo is not None and self.neo.is_real()

    def _mem_id(self, item) -> int:
        """从 Item 提取 mem_id；优先用 item.id，否则用 content hash。"""
        mid = getattr(item, "id", None)
        if mid is None:
            return hash(item.content) & 0x7FFFFFFF
        return int(mid)

    # ─── 原子图操作 ─────────────────────────────────────────────────────────

    def _upsert_memory_node(self, mem_id: int, content: str, importance: float) -> None:
        """插入或更新记忆节点（cypher 照抄 Go 版）。"""
        if not self._available():
            return
        try:
            self.neo.run_cypher(
                """MERGE (m:Memory {mem_id: $id})
                 SET m.content = $content, m.importance = $importance""",
                {"id": int(mem_id), "content": content, "importance": float(importance)},
            )
        except Exception as e:
            logger.warning("⚠️  Neo4j upsertMemoryNode 失败 (id=%s): %s", mem_id, e)

    def _add_memory_edge(self, from_id: int, to_id: int, edge_type: str, weight: float) -> None:
        """在两条记忆之间添加关系边。
        edge_type: FOLLOWS | SIMILAR_TO | CAUSES | BELONGS_TO
        """
        if not self._available():
            return
        if edge_type not in ("FOLLOWS", "SIMILAR_TO", "CAUSES", "BELONGS_TO"):
            logger.warning("⚠️  非法的边类型: %s", edge_type)
            return
        query = (
            "MATCH (a:Memory {mem_id: $from}), (b:Memory {mem_id: $to}) "
            "MERGE (a)-[r:" + edge_type + "]->(b) "
            "SET r.weight = $weight"
        )
        try:
            self.neo.run_cypher(
                query,
                {"from": int(from_id), "to": int(to_id), "weight": float(weight)},
            )
        except Exception as e:
            logger.warning("⚠️  Neo4j addMemoryEdge 失败 (%s→%s): %s", from_id, to_id, e)

    def _expand_memory_neighbors(self, seed_ids: List[int], hops: int) -> List[int]:
        """从种子记忆 ID 出发，按 hops 跳扩展邻居 ID。"""
        if not self._available() or not seed_ids:
            return []
        hop_str = "1" if hops <= 1 else "1.." + str(hops)
        query = (
            "MATCH (m:Memory) WHERE m.mem_id IN $ids "
            "MATCH (m)-[:FOLLOWS|SIMILAR_TO|CAUSES|BELONGS_TO*" + hop_str + "]-(n:Memory) "
            "WHERE NOT n.mem_id IN $ids "
            "RETURN DISTINCT n.mem_id AS id"
        )
        try:
            records = self.neo.run_cypher(query, {"ids": [int(i) for i in seed_ids]})
        except Exception as e:
            logger.warning("⚠️  Neo4j expandMemoryNeighbors 失败: %s", e)
            return []
        result: List[int] = []
        for rec in records:
            v = rec.get("id")
            if v is not None:
                try:
                    result.append(int(v))
                except (TypeError, ValueError):
                    continue
        return result

    def _delete_memory_node(self, mem_id: int) -> None:
        """删除一条记忆节点及其所有边。"""
        if not self._available():
            return
        try:
            self.neo.run_cypher(
                "MATCH (m:Memory {mem_id: $id}) DETACH DELETE m",
                {"id": int(mem_id)},
            )
        except Exception as e:
            logger.warning("⚠️  Neo4j deleteMemoryNode 失败 (id=%s): %s", mem_id, e)

    def _get_high_centrality_ids(self, candidates: List[int], threshold: int) -> List[int]:
        """在候选列表中找出图中入度 ≥ threshold 的节点。"""
        if not self._available() or not candidates:
            return []
        query = (
            "MATCH (m:Memory) WHERE m.mem_id IN $ids "
            "WITH m, size([(m)<-[]-() | 1]) AS indegree "
            "WHERE indegree >= $threshold "
            "RETURN m.mem_id AS id"
        )
        try:
            records = self.neo.run_cypher(
                query,
                {"ids": [int(i) for i in candidates], "threshold": int(threshold)},
            )
        except Exception as e:
            logger.warning("⚠️  Neo4j getHighCentrality 失败: %s", e)
            return []
        result: List[int] = []
        for rec in records:
            v = rec.get("id")
            if v is not None:
                try:
                    result.append(int(v))
                except (TypeError, ValueError):
                    continue
        return result

    # ─── 对外接口（任务约定的核心 4 个方法）────────────────────────────────

    def add_to_graph(self, item, neighbors: Optional[Iterable] = None) -> int:
        """将一条记忆同步进图：
        1) upsert :Memory 节点
        2) 与上一条记忆建立 FOLLOWS 边（时序）
        3) 与传入的 neighbors（同 LTM 中已有条目）按 cosine 相似度建立 SIMILAR_TO 边

        返回写入的 mem_id；不可用时返回 -1。
        """
        if not self._available():
            return -1
        mem_id = self._mem_id(item)
        content = getattr(item, "content", "")
        importance = float(getattr(item, "importance", 0.5) or 0.5)
        self._upsert_memory_node(mem_id, content, importance)

        if self.prev_id >= 0 and self.prev_id != mem_id:
            self._add_memory_edge(self.prev_id, mem_id, "FOLLOWS", 1.0)

        if neighbors:
            new_emb = getattr(item, "embedding", None) or []
            if new_emb:
                for old in neighbors:
                    if old is item:
                        continue
                    old_id = self._mem_id(old)
                    if old_id == mem_id:
                        continue
                    old_emb = getattr(old, "embedding", None) or []
                    if not old_emb:
                        continue
                    sim = _cosine(old_emb, new_emb)
                    if sim >= self.sim_thresh:
                        self._add_memory_edge(old_id, mem_id, "SIMILAR_TO", sim)

        self.prev_id = mem_id
        return mem_id

    def find_related(self, item_id: int, max_hops: Optional[int] = None) -> List[int]:
        """从 item_id 出发，沿图扩展 max_hops 跳，返回关联但不在种子中的 mem_id 列表。

        max_hops 未指定时使用 cfg.kg_max_hops。
        """
        if not self._available():
            return []
        hops = max_hops if (max_hops is not None and max_hops > 0) else self.cfg.kg_max_hops
        return self._expand_memory_neighbors([int(item_id)], int(hops))

    def delete_from_graph(self, item_id: int) -> None:
        """从图中删除一条记忆节点（连同其所有边）。"""
        if not self._available():
            return
        mid = int(item_id)
        if self.prev_id == mid:
            self.prev_id = -1
        self._delete_memory_node(mid)

    def bulk_index(self, items: Iterable) -> int:
        """批量索引：把一批 Item 同步进图（启动期从 LTM 恢复时使用）。

        会按顺序建立 FOLLOWS 链；返回成功 upsert 的节点数。
        """
        if not self._available():
            return 0
        count = 0
        prev_local = self.prev_id
        items_list = list(items)
        for idx, item in enumerate(items_list):
            mem_id = self._mem_id(item)
            self._upsert_memory_node(
                mem_id,
                getattr(item, "content", ""),
                float(getattr(item, "importance", 0.5) or 0.5),
            )
            if prev_local >= 0 and prev_local != mem_id:
                self._add_memory_edge(prev_local, mem_id, "FOLLOWS", 1.0)
            prev_local = mem_id
            count += 1
        self.prev_id = prev_local
        return count

    # ─── 辅助：图中心度保护（可选用于 consolidate）────────────────────────

    def filter_protected(self, candidate_ids: List[int], indegree_threshold: int = 3) -> List[int]:
        """返回 candidate_ids 中入度 ≥ threshold 的节点列表（应豁免删除）。"""
        return self._get_high_centrality_ids(candidate_ids, indegree_threshold)

    def update_node(self, item) -> None:
        """记忆内容/重要性变更后同步更新 Neo4j 节点。"""
        if not self._available():
            return
        self._upsert_memory_node(
            self._mem_id(item),
            getattr(item, "content", ""),
            float(getattr(item, "importance", 0.5) or 0.5),
        )

    def close(self) -> None:
        """语义对齐 Go 版；底层 driver 由 Neo4jClient 拥有，这里仅清状态。"""
        self.prev_id = -1
