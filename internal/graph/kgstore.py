# kgstore — Neo4j 知识图谱存储 + 多跳遍历检索
import logging
from typing import Callable, List, Optional

from config.config import APIConfig
from internal.platform.neo4j import Neo4jClient

from .extractor import Extractor
from .types import (
    ChunkRef,
    Entity,
    GraphSearchResult,
    Relation,
)

logger = logging.getLogger(__name__)


# LLM 回调签名：(system_prompt, user_msg) -> str
LLMFn = Callable[[str, str], str]

class KGStore:
    """
    在 Neo4jClient 之上封装 RAG 专用的图操作：
    - index_document：文档摄入时写入实体节点和关系边
    - delete_document：删除文档及其关联的孤立节点
    - search：。根据查询实体做 1~2 跳子图扩展，返回关联的 ChunkID 列表
    所有操作在 Neo4j 不可用时均优雅降级（返回空结果，不阻塞主流程）
    """
    def __init__(
        self,
        cfg: APIConfig,
        neo4j_client: Neo4jClient,
        llm_fn: Optional[LLMFn] = None,
    ):
        self.neo4j = neo4j_client
        self.max_hops = cfg.kg_max_hops
        self.kg_weight = cfg.kg_weight
        self.extractor = Extractor(llm_fn)

    def available(self) -> bool:
        """图存储是否可用"""
        return self.neo4j is not None and self.neo4j.is_real()

    def close(self) -> None:
        """关闭底层连接"""
        if self.neo4j is not None:
            self.neo4j.close()

    def client(self) -> Neo4jClient:
        """暴露底层 Neo4j 客户端，供 memory 包共享同一连接驱动记忆图"""
        return self.neo4j

    def inedx_document(self, doc_hash: str, chunks: List[ChunkRef]) -> None:
        """为一批 chunks 抽取实体关系并写入图（不阻塞主 Ingest 流程）。"""
        if not self.available():
            return
        for c in chunks:
            result = self.extractor.extract(c.content)
            if not result.entities:
                continue

        # 写入实体节点
        for ent in result.entities:
            ent.doc_hash = doc_hash
            ent.chunk_id = c.id
            ent.pg_id = c.pg_id
            self._upsert_entity(ent)

        # 写入关系边
        for rel in result.relations:
            rel.doc_hash = doc_hash
            rel.chunk_id = c.id
            rel.pg_id = c.pg_id
            self._upsert_relation(rel)

        logger.info("🕸️  知识图谱索引完成：docHash=%s，chunks=%d", doc_hash, len(chunks))

    def _upsert_entity(self, ent: Entity) -> None:
        """MERGE 实体节点（幂等）"""
        query = (
            "MERGE (e:Entity {name: $name}) "
            "SET e.type = $type, e.doc_hash = $doc_hash, e.chunk_id = $chunk_id, e.pg_id = $pg_id"
        )
        try:
            self.neo4j.run_cypher(query,{
                "name": ent.name,
                "type": str(ent.type),
                "doc_hash": ent.doc_hash,
                "chunk_id": ent.chunk_id,
                "pg_id": ent.pg_id,
            })
        except Exception as e:
            logger.warning("⚠️  Neo4j upsertEntity 失败 (%s): %s", ent.name, e)

    def _upsert_relation(self, rel: Relation) -> None:
        """
        MERGE 关系边（幂等）。
            动态关系类型无法用参数传递，必须拼入查询字符串；安全性由 extractor 已过滤非法类型保证。
        """
        query = (
            "MERGE (a:Entity {name: $from}) "
            "MERGE (b:Entity {name: $to}) "
            f"MERGE (a)-[r:{rel.rel_type} {{doc_hash: $doc_hash}}]->(b) "
            "SET r.chunk_id = $chunk_id, r.pg_id = $pg_id"
        )
        try:
            self.neo4j.run_cypher(query, {
                "from": rel.from_name,
                "to": rel.to_name,
                "doc_hash": rel.doc_hash,
                "chunk_id": rel.chunk_id,
                "pg_id": rel.pg_id,
            })
        except Exception as e:
            logger.warning("⚠️  Neo4j upsertRelation 失败 (%s→%s): %s", rel.from_name, rel.to_name, e)

    def delete_document(self, doc_hash: str) -> None:
        """删除与 doc_hash 关联的所有关系，并清理孤立节点"""
        if not self.available():
            return
        try:
            self.neo4j.run_cypher(
                "MATCH ()-[r {doc_hash: $doc_hash}]-() DELETE r",
                {"doc_hash": doc_hash},
            )
        except Exception as e:
            logger.warning("⚠️  Neo4j 删除文档关系失败: %s", e)
        try:
            self.neo4j.run_cypher(
                "MATCH (e:Entity) WHERE NOT (e)--() AND e.doc_hash = $doc_hash DELETE e",
                {"doc_hash": doc_hash},
            )
        except Exception as e:
            logger.warning("⚠️  Neo4j 清理孤立节点失败: %s", e)
