# handler — HTTP API 路由处理（FastAPI + Pydantic + CORS）
import logging
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.config import APIConfig
from internal.agent.agent import ChatOptions, Response, UnifiedAgent
from internal.infra.infra import Infrastructure

logger = logging.getLogger(__name__)

try:
    from PyPDF2 import PdfReader
    _HAS_PDF = True
except ImportError:
    _HAS_PDF = False
    logger.warning("⚠️  PyPDF2 未安装，PDF 解析不可用")


# ─── 请求 / 响应 模型 ──────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户输入")
    use_rag: bool = False
    selected_tools: Optional[List[str]] = None
    explicit: bool = False


class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class MCPParam(BaseModel):
    name: str
    description: str = ""
    required: bool = False


class MCPRegisterRequest(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""
    endpoint: str = Field(..., min_length=1)
    params: List[Dict[str, Any]] = Field(default_factory=list)


class DocsDeleteRequest(BaseModel):
    doc_hash: str = Field(..., min_length=1)


class UploadJSONRequest(BaseModel):
    content: str = Field(..., min_length=1)


# ─── 工具函数 ───────────────────────────────────────────────────────────────

def extract_text_from_file(file: UploadFile, content: bytes) -> str:
    filename = file.filename or ""
    if filename.lower().endswith(".pdf"):
        if not _HAS_PDF:
            raise HTTPException(status_code=500, detail="PyPDF2 未安装，无法解析 PDF")
        try:
            reader = PdfReader(BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            logger.error("PDF 解析失败: %s", e)
            raise HTTPException(status_code=500, detail=f"PDF 解析失败: {str(e)}")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("gbk", errors="ignore")


def _response_to_dict(resp: Response) -> Dict[str, Any]:
    return {
        "query": resp.query,
        "answer": resp.answer,
        "mode": resp.mode,
        "steps": [
            {
                "type": s.type,
                "content": s.content,
                "tool": s.tool,
                "params": s.params,
            }
            for s in resp.steps
        ],
        "tool_call": resp.tool_call,
        "search_results": resp.search_results,
        "task": resp.task,
        "extracted_info": resp.extracted_info,
        "short_term_count": resp.short_term_count,
        "long_term_count": resp.long_term_count,
        "preferences": resp.preferences,
        "interrupted": resp.interrupted,
        "success": True,
    }


# ─── 路由组装 ───────────────────────────────────────────────────────────────

def setup_routes(agent: UnifiedAgent, inf: Infrastructure, cfg: APIConfig) -> FastAPI:
    app = FastAPI(title="AGI Assistant", version="1.0")

    # CORS：开发期允许全部，生产可由 cfg.cors_origins 收紧
    origins = getattr(cfg, "cors_origins", None) or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "milvus": inf.ready.milvus,
            "postgresql": inf.ready.postgresql,
            "elasticsearch": inf.ready.elasticsearch,
            "kafka": inf.ready.kafka,
        }

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        try:
            opts = ChatOptions(
                use_rag=req.use_rag,
                selected_tools=req.selected_tools,
                explicit=req.explicit,
            )
            resp = agent.process_with_options(req.message, opts)
            return _response_to_dict(resp)
        except HTTPException:
            raise
        except Exception as e:
            logger.error("聊天接口错误: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/chat/cancel")
    async def chat_cancel():
        try:
            agent.cancel()
            return {"ok": True, "message": "已发送取消信号"}
        except Exception as e:
            logger.error("取消失败: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/docs/delete")
    async def docs_delete(req: DocsDeleteRequest):
        try:
            rag = getattr(agent, "rag", None)
            if rag and hasattr(rag, "delete"):
                rag.delete(req.doc_hash)
            return {"ok": True, "doc_hash": req.doc_hash}
        except HTTPException:
            raise
        except Exception as e:
            logger.error("删除文档失败: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        try:
            content = await file.read()
            text = extract_text_from_file(file, content)
            if not text.strip():
                return {"chunk_count": 0, "success": False, "message": "文件内容为空"}
            chunk_count = agent.rag_ingest(text)
            return {"chunk_count": chunk_count, "success": True}
        except HTTPException:
            raise
        except Exception as e:
            logger.error("上传接口错误: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/rag/query")
    async def rag_query(req: RAGQueryRequest):
        try:
            answer, results = agent.rag_query(req.question)
            return {
                "answer": answer,
                "results": [
                    {
                        "content": r["content"],
                        "score": r["score"],
                        "source": r.get("source", "unknown"),
                    }
                    for r in results
                ],
                "success": True,
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error("RAG 查询接口错误: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/tools/mcp")
    async def register_mcp_tool(req: MCPRegisterRequest):
        try:
            name = req.name.strip()
            description = req.description.strip()
            endpoint = req.endpoint.strip()
            if not name or not endpoint:
                raise HTTPException(status_code=400, detail="缺少 name 或 endpoint 参数")

            def _mcp_func(args: Dict[str, str]) -> str:
                return f"MCP 工具 {name} 已注册，端点: {endpoint}，参数: {args}"

            agent.register_mcp_tool(name, description, req.params, _mcp_func)
            return {"success": True, "ok": True}
        except HTTPException:
            raise
        except Exception as e:
            logger.error("注册 MCP 工具失败: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/status")
    async def status():
        return {
            "status": "running",
            "milvus": inf.ready.milvus,
            "postgresql": inf.ready.postgresql,
            "elasticsearch": inf.ready.elasticsearch,
            "kafka": inf.ready.kafka,
            "llm": cfg.llm_model,
            "embedding": cfg.embedding_model,
        }

    @app.get("/api/tools")
    async def tools():
        return {"tools": agent.get_tools()}

    @app.get("/api/memory")
    async def memory():
        return {
            "short_term_turns": agent.stm.count(),
            "long_term_count": len(agent.ltm.items),
            "preferences": agent.preference.get_all(),
        }

    @app.get("/api/snapshots")
    async def snapshots(limit: int = 50):
        try:
            return {"snapshots": inf.list_snapshots(limit=limit), "success": True}
        except Exception as e:
            logger.error("加载快照失败: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    # 静态前端：仅在目录存在时挂载，避免容器内缺失目录直接崩
    frontend_dir = os.environ.get("FRONTEND_DIR", "frontend")
    if os.path.isdir(frontend_dir):
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    else:
        logger.warning("⚠️  frontend 目录不存在: %s（跳过静态挂载）", frontend_dir)

    return app
