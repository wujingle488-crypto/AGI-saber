# hybrid — 企业级混合检索：Milvus 语义 + ES BM25 + Neo4j 知识图谱 + 三路 RRF 融合
#
# 三路 score 来自 reciprocal rank fusion（基于 rank，不依赖原始分尺度）：
#     score(d) = Σ_i  weight_i / (k + rank_i(d))
# 权重：semantic_weight、(1 - semantic_weight - kg_weight)、kg_weight，
# 任一路不可用 / 失败时跳过并把剩余权重重新归一，避免某一路因不可用拉低融合分数。
import logging
from dataclasses import dataclass, field
import threading
from typing import Callable, Dict, List, Optional

from jupyter_server.utils import fetch

from config.config import APIConfig
from internal.graph.kgstore import KGStore
from internal.infra.infra import Infrastructure

logger = logging.getLogger(__name__)

# Embedding 回调签名：text -> List[float]
EmbedFn = Callable[[str], List[float]]

@dataclass
class HybridResult:
    """混合检索的单条结果（与 Go 版 HybridResult 字段对齐）"""
    pg_id: int = 0
    content: str = ""
    score: float = 0.0
    source: str = "" # "hybrid" 三路融合 | "semantic" 仅 Milvus| "keyword" 仅 ES
    parent: str = ""

@dataclass
class _PathHits:
    """单路检索结果（rank 顺序）+ 是否成功"""
    hits: List[dict] = field(default_factory=list)
    ok: bool = False

class HybridStore:
    """
    企业级混合检索：
        - Milvus 语义向量检索
        - Elasticsearch BM25 关键词检索
        - Neo4j 知识图谱实体遍历检索
        - Reciprocal Rank Fusion 三路融合

    根据基础设施可用性自动选择检索模式：
        hybrid / semantic / keyword / unavailable
    """
    def __init__(
            self,
            cfg: APIConfig,
            inf: Infrastructure,
            embed_fn:Optional[EmbedFn] = None,
            kg: Optional[KGStore] = None,
    ):
        self.cfg = cfg
        self.inf = inf
        self._embed_fn = embed_fn
        self._kg = kg
        self._reranker = None
        self.mode = self._resolve_mode()

    # ─── 注入式 setter ─────────────────────────────────
    def set_embed_fn(self, fn: EmbedFn) -> None:
        self._embed_fn = fn

    def set_kg_store(self, kg: Optional[KGStore]) -> None:
        self._kg = kg

    def set_reranker(self, reranker) -> None:
        self._reranker = reranker

    def _milvus_ok(self) -> bool:
        return self.inf.ready.milvus == "connected"

    def _es_ok(self) -> bool:
        return self.inf.ready.elasticsearch == "connected"

    def _kg_ok(self) -> bool:
        return self._kg is not None and self._kg.available()

    def _resolve_mode(self) -> str:
        m, e = self._milvus_ok(), self._es_ok()
        if m and e:
            return "hybrid"
        if m:
            return "semantic"
        if e:
            return "keyword"
        return "unavailable"

    def search(self,query: str,top_k: int) -> List[HybridResult]:
        self.mode = self._resolve_mode()
        if self.mode == "hybrid":
            return self._search_hybrid(query, top_k)
        if self.mode == "semantic":
            return self._search_semantic(query, top_k)
        if self.mode == "keyword":
            return self._search_keyword(query, top_k)
        logger.warning("⚠️  检索基础设施不可用（Milvus 和 ES 均未连接）")
        return []

    def search_multi(self, querys: List[str], top_k: int) -> List[HybridResult]:
        querys = [q for q in (querys or []) if q]
        if not querys:
            return []

        pool = self._rerank_pool(top_k)
        if len(querys) == 1:
            return self._finalize(querys[0], self.search(querys[0], pool), top_k)

        # 代表有几个query就初始化几个List，最终在一个大List中
        results_by_query: List[List[HybridResult]] = [[] for _ in querys]

        threads = []

        def _run(idx: int, q: str) -> None:
            results_by_query[idx] = self.search(q, top_k)

        # 相当于开启三个子进程并发执行任务
        for i, q in enumerate(querys):
            # 这里的结构为[0,"query1"],[1,"query2"]...
            # i代表下边 q代表每一个query
            t = threading.Thread(
                target=_run,
                args=(i, q),
                daemon=True
            )
            threads.append(t)
            t.start()

        # 主进程(当前函数)等待三个子进程执行完成之后再向下执行
        for t in threads:
            t.join()

        k = self.cfg.rrf_constant_k if self.cfg.rrf_constant_k > 0 else 60
        merged: Dict[str,dict] = {}

        for query_result in results_by_query:
            for rank, result in enumerate(query_result):
                # 如果pg_id存在则直接用为Key，如果不存在则取内容的前100个字符
                key = f"id:{result.pg_id}" if result.pg_id else f"c:{result.content[:100]}"
                score = 1 / float(k + rank + 1)
                if key in merged:
                    merged[key]["score"] += score
                    # 保留分数最高的那个 HybridResult 对象
                    if result.score > merged[key]["result"].score:
                        merged[key]["result"] = (result)
                else:
                    merged[key] = {"score":score, "result": result}

        out: List[HybridResult] = []
        for item in merged.values():
            result = item["result"]
            result.score = item["score"]
            out.append(result)
        out.sort(key=lambda r: r.score, reverse=True)

        if len(out) > pool:
            out = out[:pool]

        return self._finalize(querys[0], out, top_k)

    def _rerank_pool(self, top_k: int) -> int:
        # 召回更多让重排模型挑选
        pool = top_k * (4 if self._reranker is not None else 2)
        return max(pool, 10)

    def _finalize(self,query: str,results: List[HybridResult],top_k: int) -> List[HybridResult]:
        # 如果有重排序模型则使用模型进行重排序截取top_k
        if self._reranker is not None and len(results) > 1:
            return self._reranker.rerank(query, results, top_k)

        # 如果没有重排序模型则直接截取top_k
        if top_k > 0 and len(results) > 1:
            return results[:top_k]

        return results

    # ─── 混合检索：三路 RRF 融合 ─────────────────────────────────────────
    def _search_hybird(self, query: str, top_k: int) -> List[HybridResult]:
        # 每路多召回一些，留给融合层挑选。
        fetch_k = max(top_k * 2, 10)
        milvus_path = self._fetch_milvus(query,fetch_k)
        es_path = self._fetch_es(query, fetch_k)
        kg_path = self._fetch_kg(query, fetch_k)

        if not milvus_path.ok and not es_path.ok and not kg_path.ok:
            logger.warning("⚠️  三路检索均失败")
            return []

        if not milvus_path.ok and not es_path.ok:
            return self._materialize_kg_only(kg_path.hits, top_k)

        if not milvus_path.ok:
            logger.warning("⚠️  Milvus 检索失败，降级到关键词检索")
            return self._search_keyword(query, top_k)
        if not es_path.ok:
            logger.warning("⚠️  ES 检索失败，降级到语义检索")
            return self._search_semantic(query, top_k)

        sem_w, kw_w, kg_w = self._normalized_weights(kg_path.ok)
        k = self.cfg.rrf_constant_k if self.cfg.rrf_constant_k > 0 else 60

        # 字典，key=pg_id，value=累计的 RRF 加权分数
        rrf_scores: Dict[int, float] = {}

        # milvus这一路结果累加
        for rank, hit in enumerate(milvus_path.hits):
            pg_id = hit.get("pg_id")
            if pg_id is None:
                continue
            rrf_scores[pg_id] = rrf_scores.get(pg_id, 0.0) + sem_w / (k + rank + 1)

        # es这一路结果累加
        for rank, hit in enumerate(es_path.hits):
            pg_id = hit.get("pg_id")
            if pg_id is None:
                continue
            rrf_scores[pg_id] = rrf_scores.get(pg_id, 0.0) + kw_w / (k + rank + 1)

        # neo4j这一路结果累加
        if kg_path.ok and kg_w > 0:
            for rank, hit in enumerate(kg_path.hits):
                pg_id = hit.pg_id if hasattr(hit, "pg_id") else hit.get("pg_id", 0)
                if not pg_id:
                    continue
                rrf_scores[pg_id] = rrf_scores.get(pg_id, 0.0) + kg_w / (k + rank + 1)

        # 排序 + 截断
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_ids) > top_k:
            sorted_ids = sorted_ids[:top_k]
        if not sorted_ids:
            return []

        # 从 PG 批量取回 chunk 内容
        ids = [pid for pid,_ in sorted_ids]
        rows = self.inf.load_rag_chunk_by_ids(ids)
        row_map: Dict[int, dict] = {r["id"] : r for r in rows}

        results: List[HybridResult] = []
        for pid, score in sorted_ids:
            row = row_map.get(pid)
            if row is None:
                continue
            results.append(
                HybridResult(
                    pg_id = pid,
                    content = row.get("content",""),
                    score = score,
                    source="hybrid",
                    parent=row.get("parent_content", "") or row.get("parent", ""),
                )
            )

        return results

    def _search_semantic(self, query: str, top_k: int) -> List[HybridResult]:
        path = self._fetch_milvus(query, top_k)
        if not path.ok:
            return []

        ids = [h["pg_id"] for h in path.hits if h.get("pg_id") is not None]
        rows = self.inf.load_rag_chunk_by_ids(ids) if ids else []
        row_map: Dict[int, dict] = {r["id"]: r for r in rows}
        results: List[HybridResult] = []

        for h in path.hits:
            pid = h.get("pg_id")
            if pid is None:
                continue
            row = row_map.get(pid,{})
            content = row.get("content") or h.get("content") or ""
            if not content:
                continue
            results.append(
                HybridResult(
                    pg_id=pid,
                    content = content,
                    score=float(h.get("score", 0.0)),
                    source="semantic",
                    parent=row.get("parent_content", "") or row.get("parent", ""),
                )
            )
        return results

    def _search_keyword(self, query: str, top_k: int) -> List[HybridResult]:
        path = self._fetch_es(query, top_k)
        if not path.ok:
            return []
        ids = [h["pg_id"] for h in path.hits if h.get("pg_id") is not None]
        rows = self.inf.load_rag_chunks_by_ids(ids) if ids else []
        row_map: Dict[int, dict] = {r["id"]: r for r in rows}
        results: List[HybridResult] = []
        for h in path.hits:
            pid = h.get("pg_id")
            if pid is None:
                continue
            row = row_map.get(pid, {})
            content = row.get("content") or h.get("content") or ""
            if not content:
                continue
            results.append(
                HybridResult(
                    pg_id=pid,
                    content=content,
                    score=float(h.get("score", 0.0)),
                    source="keyword",
                    parent=row.get("parent_content", "") or row.get("parent", ""),
                )
            )
        return results

    def _fetch_milvus(self,query: str, fetch_k: int) -> _PathHits:
        if not self._milvus_ok():
            return _PathHits(ok=False)
        if self._embed_fn is None:
            logger.warning("⚠️  embed_fn 未注入，跳过 Milvus 语义路")
            return _PathHits(ok=False)

        try:
            query_emb = self._embed_fn(query)
        except Exception as e:
            logger.warning("⚠️  查询向量化失败: %s", e)
            return _PathHits(ok=False)

        if not query_emb:
            return _PathHits(ok=False)
        if self.cfg.rag_milvus_dim and len(query_emb) != self.cfg.rag_milvus_dim:
            logger.warning(
                "⚠️  embedding 维度 %d 与 rag_milvus_dim=%d 不匹配，跳过语义路",
                len(query_emb), self.cfg.rag_milvus_dim,
            )
            return _PathHits(ok=False)

        try:
            hits = self.inf.milvus_search_with_scores("rag_embeddings", query_emb, fetch_k)
            return _PathHits(ok=True, hits=hits)
        except Exception as e:
            logger.warning("⚠️  Milvus 检索失败: %s", e)
            return _PathHits(ok=False)

    def _fetch_es(self, query: str, fetch_k: int) -> _PathHits:
        if not self._es_ok():
            return _PathHits(ok=False)

        try:
            hits = self.inf.search_rag_chunks(query, fetch_k) or []
            return _PathHits(ok=True, hits=hits)
        except Exception as e:
            logger.warning("⚠️  ES 检索失败: %s", e)
            return _PathHits(ok=False)

    def _fetch_kg(self, query: str, fetch_k: int) -> _PathHits:
        if not self._kg_ok():
            return _PathHits(ok=False)
        try:
            hits = self._kg.search(query, fetch_k) or []
            return _PathHits(hits=hits, ok=True)
        except Exception as e:
            logger.warning("⚠️  KG 检索失败: %s", e)
            return _PathHits(ok=False)

    def _normalized_weights(self, kg_available: bool) -> tuple:
        """
        计算三路权重（semantic, keyword, kg），KG 不可用时把权重归一到剩余两路。

            约定：
                sem_w = cfg.semantic_weight
                kw_w  = 1 - sem_w - kg_w
                kg_w  = cfg.kg_weight  （cfg.kg_enabled=False 或 KG 不可用时强制 0）
        """
        sem_w = max(0.0, float(self.cfg.semantic_weight))
        kg_w = max(0.0, float(self.cfg.kg_weight))
        if not getattr(self.cfg, "kg_enabled", False) or not kg_available:
            kg_w = 0.0

        # clamp，避免配置错误导致权重溢出
        if sem_w > 1.0:
            sem_w = 1.0
        if sem_w + kg_w > 1.0:
            kg_w = max(0.0, 1.0 - sem_w)

        kw_w = 1.0 - sem_w - kg_w
        if kw_w < 0:
            kw_w = 0.0

        total = sem_w + kw_w + kg_w
        if total <= 0:
            # 极端兜底：全 0 配置 → 平分两路
            return 0.5, 0.5, 0.0
        # 归一（已确保 total≈1 时不变）
        return sem_w / total, kw_w / total, kg_w / total

    def _materialize_kg_only(self, kg_hits: List, top_k: int) -> List[HybridResult]:
        if not kg_hits:
            return []
        ids: List[int] = []
        for h in kg_hits:
            pid = h.pg_id if hasattr(h, "pg_id") else h.get("pg_id", 0)
            if pid:
                ids.append(pid)
        if not ids:
            return []
        ids = ids[:top_k]
        rows = self.inf.load_rag_chunks_by_ids(ids)
        row_map: Dict[int, dict] = {r["id"]: r for r in rows}
        results: List[HybridResult] = []
        for h in kg_hits:
            pid = h.pg_id if hasattr(h, "pg_id") else h.get("pg_id", 0)
            row = row_map.get(pid, {})
            content = row.get("content")
            if not pid or content is None:
                continue
            score = float(getattr(h, "score", 0.0))
            results.append(HybridResult(
                pg_id=pid, content=content, score=score, source="hybrid",
                parent=row.get("parent_content", "") or row.get("parent", ""),
            ))
            if len(results) >= top_k:
                break
        return results




