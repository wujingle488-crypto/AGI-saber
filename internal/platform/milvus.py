# milvus — Milvus 向量数据库平台层薄封装：连接、集合初始化、insert/search/delete。
# 失败时降级到内存模式（self._client 为 None），不阻塞应用启动。
import logging
from typing import Any, Dict, List, Optional

from config.config import APIConfig

logger = logging.getLogger(__name__)

try:
    from pymilvus import MilvusClient
    _HAS_MILVUS = True
except ImportError:
    MilvusClient = None  # type: ignore
    _HAS_MILVUS = False


# 默认 RAG 集合名（与 infra 旧实现保持一致）
DEFAULT_RAG_COLLECTION = "rag_embeddings"

class MilvusClientWrapper:
    """
    Milvus 平台客户端：连接、集合管理、关键向量操作。

    保持平台层无业务语义；业务集合的 schema 与查询逻辑由调用方决定。
    """
    def __init__(self,cfg: APIConfig):
        self.cfg = cfg
        self._client = None
        self.status: str = "disconnected"
        self._connect()
        if self._client is not None:
            self._init_default_collections()

    # ─── 连接 ───
    def _connect(self) -> None:
        if not _HAS_MILVUS:
            logger.warning("⚠️  pymilvus 未安装，Milvus 不可用 (将使用内存向量库)")
            self.status = "memory-mode"
            return
        if not self.cfg.milvus_host or not self.cfg.milvus_port:
            logger.warning("⚠️  Milvus 未配置")
            self.status = "memory-mode"
            return

        try:
            uri = f"http://{self.cfg.milvus_addr()}"
            self._client = MilvusClient(uri=uri)
            self.status = "connected"
            logger.info("✅ Milvus 已连接: %s", self.cfg.milvus_addr())

        except Exception as e:
            logger.warning("⚠️  Milvus 连接失败: %s (将使用内存向量库)", e)
            self._client = None
            self.status = "memory-mode"

    # ─── 状态判断 ───
    def is_real(self) -> bool:
        return self._client is not None

    @property
    def client(self):
        return self._client

    # ─── 集合初始化 ───
    def _init_default_collections(self) -> None:
        """启动期幂等创建默认 RAG 集合。"""
        if self._client is None:
            return

        # 向量嵌入的维度，指定或者默认1024
        dim = int(self.cfg.rag_milvus_dim or 1024)

        try:
            if not self._client.has_collection(DEFAULT_RAG_COLLECTION):
                self._client.create_collection(
                    collection_name = DEFAULT_RAG_COLLECTION,
                    dimension = dim,
                    auto_id=True,
                    enable_dynamic_field=True,
                )
            logger.info("✅ Milvus 集合 %s 已创建 (dim=%d)", DEFAULT_RAG_COLLECTION, dim)
        except Exception as e:
            logger.warning("⚠️  Milvus 创建集合失败: %s", e)

    def ensure_collection(self, collection_name: str, dimension: int,
                          auto_id: bool = True, enable_dynamic_field: bool = True) -> bool:
        """幂等创建任意业务集合。"""
        if self._client is None:
            return False
        try:
            if not self._client.has_collection(collection_name):
                self._client.create_collection(
                    collection_name=collection_name,
                    dimension=dimension,
                    auto_id=auto_id,
                    enable_dynamic_field=enable_dynamic_field,
                )
                logger.info("✅ Milvus 集合 %s 已创建 (dim=%d)", collection_name, dimension)
            return True
        except Exception as e:
            logger.warning("⚠️  Milvus 创建集合失败 (%s): %s", collection_name, e)
            return False

    # ─── 关键操作 ───
    def insert(self,collection_name: str,data: List[Dict[str,Any]]) -> bool:
        """向 Milvus 插入实体记录；失败返回 False。"""
        if not self._client or not data:
            return False

        try:
            self._client.insert(collection_name = collection_name,data = data)
            return True
        except Exception as e:
            logger.warning("⚠️  Milvus 插入失败: %s", e)
            return False

    def search(self, collection_name: str, query_emb: List[float], top_k: int,
               output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """单 query 向量检索；返回 hit 列表（pg_id/content/score）。失败返回空列表。"""
        if not self._client:
            return []

        try:
            results = self._client.search(
                collection_name=collection_name,
                data=query_emb,
                limit=top_k,
                output_fields=output_fields or ["pg_id","content"]
            )
            hits: List[Dict[str, Any]] = []
            if not results:
                return hits
            for result in results:
                entity = result.get("entity",{})  if isinstance(result, dict) else {} # 如果是一个字典，就获取entity的值，如果不是，返回一个空字典
                hits.append(
                    {
                        "pg_id":entity.get("pg_id"),
                        "content":entity.get("content"),
                        "score":entity.get("score")
                    }
                )
                return hits
        except Exception as e:
            logger.warning("⚠️  Milvus 检索失败: %s", e)
            return []

    def delete(self, collection_name: str, filter_expr: str) -> bool:
        """按布尔表达式删除（如 'pg_id == 123'）。"""
        if self._client is None:
            return False
        try:
            self._client.delete(collection_name=collection_name, filter=filter_expr)
            return True
        except Exception as e:
            logger.warning("⚠️  Milvus 删除失败: %s", e)
            return False

    def delete_by_pg_ids(self, collection_name: str, pg_ids: List[int]) -> None:
        if self._client is None or not pg_ids:
            return
        for pid in pg_ids:
            try:
                self._client.delete(collection_name=collection_name, filter=f"pg_id == {pid}")
            except Exception as e:
                logger.warning("⚠️  Milvus 删除失败 (pg_id=%s): %s", pid, e)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.warning("⚠️  Milvus 关闭失败: %s", e)
            finally:
                self._client = None
                self.status = "disconnected"



