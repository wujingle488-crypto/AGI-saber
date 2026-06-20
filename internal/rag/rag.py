# rag — 检索增强生成（RAG）：Milvus 语义 + ES BM25 + Neo4j 图 + 三路 RRF 融合
import json
import logging
import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from config.config import APIConfig
from internal.graph.kgstore import KGStore
from internal.infra.infra import Infrastructure
from internal.llm.llm import Client as LLMClient
from internal.rag.hybrid import HybridStore
from internal.rag.rewriter import HistoryMessage
from internal.rag.splitter import Chunk, RecursiveSplitter

logger = logging.getLogger(__name__)


class Engine:
    """RAG 引擎：切分 → 入库（PG/Milvus/ES） → 检索（RRF 融合） → LLM 合成。"""

    def __init__(self, cfg: APIConfig, inf: Infrastructure, llm: Optional[LLMClient] = None):
        self.cfg = cfg
        self.inf = inf
        parent_size = max(cfg.chunk_size * 4, 600)
        parent_overlap = cfg.chunk_overlap * 2
        self.parent_splitter = RecursiveSplitter(parent_size, parent_overlap)
        self.child_splitter = RecursiveSplitter(cfg.chunk_size, cfg.chunk_overlap)
        self.loaded = False
        self._generate_fn: Optional[Callable[[str, str], str]] = None
        self._rewriter = None
        self._reranker = None
        # 复用 agent 注入的 LLM 客户端，避免重复实例
        self._llm = llm if llm is not None else LLMClient(cfg)
        # 三路混合检索（Milvus + ES + KG），由 cfg.enable_hybrid_search 控制是否启用
        self._hybrid: Optional[HybridStore] = None
        if getattr(cfg, "enable_hybrid_search", False):
            self._hybrid = HybridStore(cfg, inf, embed_fn=self._llm.embed)
        self._check_existing_chunks()

    def set_generate_fn(self, fn: Callable[[str, str], str]):
        self._generate_fn = fn

    def set_rewriter(self, rewriter) -> None:
        self._rewriter = rewriter

    def set_reranker(self, reranker) -> None:
        self._reranker = reranker
        if self._hybrid is not None:
            self._hybrid.set_reranker(reranker)

    def set_kg_store(self, kg: Optional[KGStore]) -> None:
        """注入知识图谱存储（图路检索的第三路）。

        必须在 cfg.enable_hybrid_search=True 时才生效；否则仅记录但不影响单路检索。
        """
        if self._hybrid is not None:
            self._hybrid.set_kg_store(kg)

    def _check_existing_chunks(self):
        try:
            if self.inf.ready.postgresql == "connected" and self.inf.count_rag_chunks() > 0:
                self.loaded = True
                logger.info("✅ 检测到知识库中已有文档")
        except Exception as e:
            logger.error("检查知识库文档失败: %s", e)

    # ── 入库 ────────────────────────────────────────────────────────────────

    def ingest(self, doc: str) -> int:
        parents = self.parent_splitter.split(doc)
        chunks: List[Chunk] = []
        child_parents: List[str] = []
        for parent in parents:
            for child in self.child_splitter.split(parent.content):
                child.id = len(chunks)
                chunks.append(child)
                child_parents.append(parent.content)
        if not chunks:
            return 0

        contents = [c.content for c in chunks]
        embeddings = [self._llm.embed(content) for content in contents]

        pg_ids: List[int] = []
        valid_contents: List[str] = []
        valid_embeddings: List[List[float]] = []
        doc_hash = hashlib.sha256(doc.encode("utf-8")).hexdigest()[:16]
        for i, chunk in enumerate(chunks):
            parent_content = child_parents[i] if i < len(child_parents) else ""
            if hasattr(self.inf, "save_rag_chunk_with_parent"):
                pg_id = self.inf.save_rag_chunk_with_parent(
                    doc_hash, i, chunk.content, parent_content, json.dumps(embeddings[i])
                )
            else:
                pg_id = self.inf.save_rag_chunk(doc_hash, i, chunk.content, json.dumps(embeddings[i]))
            if pg_id > 0:
                pg_ids.append(pg_id)
                valid_contents.append(chunk.content)
                valid_embeddings.append(embeddings[i])

        # 仅在 Milvus 真实连接 + embedding 维度匹配时才插入
        if (
            self.inf.ready.milvus == "connected"
            and pg_ids
            and self._llm.cfg.is_real_embedding()
            and valid_embeddings
            and len(valid_embeddings[0]) == self.cfg.rag_milvus_dim
        ):
            self.inf.insert_rag_chunks(pg_ids, valid_contents, valid_embeddings)

        for i, pg_id in enumerate(pg_ids):
            self.inf.index_rag_chunk(pg_id, valid_contents[i], doc_hash, i)

        self.loaded = True
        self.inf.publish_event("rag.ingest", json.dumps({
            "chunk_count": len(chunks),
            "parent_count": len(parents),
            "doc_hash": doc_hash,
        }))
        return len(chunks)

    # ── 检索 ────────────────────────────────────────────────────────────────

    def query(self, question: str) -> Tuple[str, List[dict]]:
        return self.query_with_history(question, [])

    def query_with_history(self, question: str, history: Optional[List[HistoryMessage]] = None) -> Tuple[str, List[dict]]:
        if not self.loaded:
            return "知识库为空，请先上传文档。", []
        if self.inf.ready.postgresql != "connected":
            return "PostgreSQL 未连接，无法查询知识库。", []

        top_k = max(1, self.cfg.top_k)
        queries = [question]
        if self._rewriter is not None:
            rewritten = self._rewriter.rewrite(question, history or [])
            if rewritten:
                queries = rewritten

        # 优先走三路混合检索（开关：cfg.enable_hybrid_search）。
        # 仅在 ES + Milvus 都连接时启用，否则降级到单路语义/关键词或老 RRF 路径。
        if (
            self._hybrid is not None
            and self.inf.ready.elasticsearch == "connected"
            and self.inf.ready.milvus == "connected"
            and self._llm.cfg.is_real_embedding()
        ):
            hybrid_hits = self._hybrid.search_multi(queries, top_k)
            fused = [
                {
                    "pg_id": h.pg_id,
                    "content": h.parent or h.content,
                    "score": h.score,
                    "source": h.source,
                }
                for h in hybrid_hits
            ]
            ask_query = queries[0] if queries else question
            return self._compose_answer(ask_query, fused)

        # 仅当 Milvus 真实连接且 embedding 真实可用时使用语义路
        milvus_results: List[dict] = []
        if self.inf.ready.milvus == "connected" and self._llm.cfg.is_real_embedding():
            try:
                query_emb = self._llm.embed(question)
                if query_emb and len(query_emb) == self.cfg.rag_milvus_dim:
                    milvus_results = self.inf.milvus_search_with_scores(
                        "rag_embeddings", query_emb, top_k * 2
                    )
            except Exception as e:
                logger.warning("⚠️  Milvus 语义检索失败: %s", e)

        es_results: List[dict] = []
        if self.inf.ready.elasticsearch == "connected":
            es_results = self.inf.search_rag_chunks(question, top_k * 2)

        fused = self._rrf_fuse(milvus_results, es_results, top_k)
        return self._compose_answer(question, fused)

    def _compose_answer(self, question: str, fused: List[dict]) -> Tuple[str, List[dict]]:
        if not fused:
            return "知识库中未找到相关内容。", []

        context = "\n\n".join(r["content"] for r in fused if r.get("content"))
        if not context:
            return "知识库中未找到相关内容。", []

        if self._generate_fn:
            system_prompt = (
                "你是一个基于知识库回答问题的助手。请仅根据提供的上下文内容回答问题，"
                "不要编造信息。如果上下文不足以回答，请说明。"
            )
            user_msg = f"上下文：\n{context}\n\n问题：{question}"
            return self._generate_fn(system_prompt, user_msg), fused

        return f"【知识库检索结果】\n{context}", fused

    def _rrf_fuse(
        self,
        milvus_results: List[dict],
        es_results: List[dict],
        top_k: int,
    ) -> List[dict]:
        """Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank_i(d))。"""
        k = self.cfg.rrf_constant_k if self.cfg.rrf_constant_k > 0 else 60

        # key 用 pg_id 优先，回退到 content 前缀
        merged: Dict[str, dict] = {}
        scores: Dict[str, float] = {}

        def _key(item: dict) -> str:
            pg_id = item.get("pg_id")
            if pg_id is not None:
                return f"id:{pg_id}"
            return f"c:{(item.get('content') or '')[:100]}"

        for rank, item in enumerate(milvus_results):
            key = _key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in merged:
                merged[key] = {
                    "pg_id": item.get("pg_id"),
                    "content": item.get("content", ""),
                    "score": 0.0,
                    "source": "milvus",
                }

        for rank, item in enumerate(es_results):
            key = _key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in merged:
                merged[key] = {
                    "pg_id": item.get("pg_id"),
                    "content": item.get("content", ""),
                    "score": 0.0,
                    "source": "es",
                }
            else:
                merged[key]["source"] = "hybrid"

        for key, item in merged.items():
            item["score"] = scores[key]

        sorted_items = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return sorted_items[:top_k]