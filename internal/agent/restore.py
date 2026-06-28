# restore — 启动时从持久化层恢复 agent 运行时状态
#
# 对应 Go 版 internal/agent/restore.go：
#   - restore_from_db：恢复偏好/长期记忆/聊天记录
#   - restore_rag_from_db：恢复 RAG chunks
#   - init_knowledge_graph：构造 KGStore + 把图层接入 RAG / GraphMemory
#
# 所有恢复动作都做异常吞没（best-effort），任意一项失败不阻塞 agent 启动。
import logging

logger = logging.getLogger(__name__)


def restore_from_db(agent):
    """启动时从 PostgreSQL 恢复跨会话的偏好、长期记忆和聊天记录。"""
    try:
        # 偏好 + 长期记忆由 Preference / LongTerm 在 __init__/load_from_storage 内部完成。
        # 这里只补做聊天记录恢复（短期记忆）：从 chat_history 取最近 N 条。
        chat_repo = getattr(agent, "chat_repo", None)
        if chat_repo is not None:
            chat_limit = agent.cfg.short_term_max_turns * 2  # 每轮 = user + assistant
            try:
                history = chat_repo.load(chat_limit)
            except Exception:
                history = []
            for h in history:
                role = getattr(h, "role", None)
                content = getattr(h, "content", None)
                if role and content:
                    agent.stm.add(role, content)
            if history:
                logger.info("✅ 聊天记录恢复：%d 条", len(history))
    except Exception as e:
        logger.warning("⚠️  restore_from_db 失败: %s", e)


def restore_rag_from_db(agent):
    """从 PostgreSQL 加载持久化的 RAG chunks 到内存索引。"""
    try:
        # Engine 内部已通过 _check_existing_chunks 检测 chunks 数量。
        # 这里只在显式提供 ragchunk_repo 时主动 reload。
        rag = getattr(agent, "rag", None)
        if rag is not None and hasattr(rag, "_check_existing_chunks"):
            rag._check_existing_chunks()
    except Exception as e:
        logger.warning("⚠️  restore_rag_from_db 失败: %s", e)


def init_knowledge_graph(agent):
    """初始化 Neo4j 知识图谱并接入 RAG / GraphMemory。失败时降级 None。"""
    try:
        from internal.graph.kgstore import KGStore
        from internal.platform.neo4j import Neo4jClient

        client = Neo4jClient(agent.cfg)

        def llm_fn(system_prompt: str, user_msg: str) -> str:
            from internal.llm.llm import Message
            return agent.llm.chat([Message(role="user", content=user_msg)], system_prompt=system_prompt)

        kg = KGStore(agent.cfg, client, llm_fn=llm_fn)
        agent.kg = kg
        # 注入到 RAG 引擎（可选）
        if hasattr(agent.rag, "set_kg_store"):
            try:
                agent.rag.set_kg_store(kg)
            except Exception:
                pass
        logger.info("🕸️  知识图谱已就绪")
    except Exception as e:
        logger.info("ℹ️  Neo4j 不可用 (%s)，记忆/RAG 退化为非图模式", e)
        agent.kg = None
