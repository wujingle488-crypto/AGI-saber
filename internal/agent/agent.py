# agent — Python 版统一智能体（重写后版本）
#
# 主分支 Go 版 internal/agent/agent.go 拆出的诸多职责被分散到同目录下的：
#   - router.py          — chat / tool / react / rag 模式路由
#   - planner.py         — ReAct 模式下的 Planner LLM
#   - restore.py         — 启动期从 PG 恢复偏好/长期记忆/聊天记录 + KG 初始化
#   - cancel.py          — 取消令牌注册表 + go_safe
#   - init_sandbox.py    — 沙箱 + exec_command 工具初始化
#   - memory_writer.py   — 异步记忆写入 + 回复事实抽取
#   - status.py          — 系统状态视图聚合
#
# 本文件只负责：构造 + 路由分派 + ReAct 推理循环（保留原 _generate_thought /
# _parse_action / _call_tool_with_retry / _async_update_memory 实现作为 react
# 分支的内部细节）。
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.config import APIConfig
from internal.infra.infra import Infrastructure
from internal.llm.llm import Client as LLMClient, Message
from internal.memory.memory import LongTerm, Preference, ShortTerm
from internal.promptctx import (
    ConstraintsSource,
    ContextAssembler,
    PlannerSnapshot,
    PlannerSource,
    Policy,
    ProfileSource,
    Query,
    RecallSource,
    SourceRegistry,
    StepObservation,
    TaskMemBuffer,
    TaskMemSource,
    ToolCallTrace,
    ToolStateSource,
    ToolStateTracker,
    default_schemas,
)
from internal.rag.rag import Engine as RAGEngine
from internal.rag.reranker import LLMReranker
from internal.rag.rewriter import HistoryMessage, LLMRewriter
from internal.tools.tools import Tool, ToolExecutor, default_tools, new_mcp_tool

from .cancel import CancelRegistry, go_safe
from .graph_runtime import GraphConfig, GraphRuntime
from .init_sandbox import init_sandbox
from .memory_writer import (
    AsyncMemoryWriter,
    async_update_memory,
    extract_memory_from_reply,
    maybe_consolidate_memory,
)
from .restore import init_knowledge_graph, restore_from_db, restore_rag_from_db
from .router import detect_tool, need_rag, need_react, need_react_from_tools, need_tool
from .planner import llm_plan_graph
from .status import infra_status, status as build_status

logger = logging.getLogger(__name__)


class StepType:
    THOUGHT = "Thought"
    ACTION = "Action"
    OBSERVATION = "Observation"
    FINAL_ANSWER = "Final Answer"


@dataclass
class ReActStep:
    type: str
    content: str
    tool: str = ""
    params: Optional[Dict[str, str]] = None


@dataclass
class ChatOptions:
    use_rag: bool = False
    selected_tools: Optional[List[str]] = None
    explicit: bool = False


@dataclass
class Response:
    query: str
    answer: str = ""
    mode: str = "chat"
    steps: List[ReActStep] = field(default_factory=list)
    tool_call: Optional[Dict[str, Any]] = None
    search_results: List[dict] = field(default_factory=list)
    task: Optional[dict] = None
    extracted_info: str = ""
    short_term_count: int = 0
    long_term_count: int = 0
    preferences: Dict[str, str] = field(default_factory=dict)
    interrupted: bool = False


class UnifiedAgent:
    """统一智能体入口。负责装配各子模块、路由分派与 ReAct 推理循环。"""

    def __init__(self, cfg: APIConfig, inf: Infrastructure):
        self.cfg = cfg
        self.inf = inf
        self.llm = LLMClient(cfg)
        self.stm = ShortTerm(cfg.short_term_max_turns)
        self.ltm = LongTerm(cfg, inf)
        self.preference = Preference("default_user", inf)

        # RAG 引擎构造失败不致命：降级为禁用知识库
        try:
            self.rag = RAGEngine(cfg, inf, self.llm)
        except Exception as e:
            logger.warning("⚠️  RAG 引擎初始化失败: %s（已禁用知识库）", e)
            self.rag = None

        # 默认工具集；planner / sandbox 可后续追加
        self.tool_executor = ToolExecutor(default_tools(cfg=cfg, llm=self.llm))

        self.max_iterations = cfg.max_iterations
        self.max_retries = cfg.max_retries

        # 取消令牌注册表 + 兼容旧接口的 process-level cancel event
        self._cancel_registry = CancelRegistry()
        self._memory_lock = threading.Lock()

        # 异步记忆写入器（单 worker 线程串行化）
        self.memory_writer = AsyncMemoryWriter()

        # 接通 LLM embed / RAG generate
        try:
            self.ltm.set_embed_fn(self.llm.embed)
        except Exception as e:
            logger.warning("⚠️  LTM embed 函数注入失败: %s", e)
        if self.rag is not None:
            try:
                self.rag.set_generate_fn(self._llm_generate)
                if getattr(cfg, "rag_rewrite_enabled", False):
                    self.rag.set_rewriter(LLMRewriter(self._llm_generate, cfg.rag_rewrite_num_queries))
                if getattr(cfg, "rag_rerank_enabled", False):
                    self.rag.set_reranker(LLMReranker(self._llm_generate, cfg.rag_rerank_preview_len))
            except Exception as e:
                logger.warning("⚠️  RAG generate 函数注入失败: %s", e)

        # 加载持久化的长期记忆 + chat_history（best-effort）
        try:
            self.ltm.load_from_storage()
        except Exception as e:
            logger.warning("⚠️  长期记忆加载失败: %s", e)

        # 沙箱：失败降级 None
        self.sandbox = None
        try:
            init_sandbox(self)
        except Exception as e:
            logger.warning("⚠️  init_sandbox 失败: %s", e)
            self.sandbox = None

        # 知识图谱：失败降级 None
        self.kg = None
        try:
            init_knowledge_graph(self)
        except Exception as e:
            logger.warning("⚠️  init_knowledge_graph 失败: %s", e)
            self.kg = None

        # 启动期恢复偏好/长期记忆/聊天记录 + RAG chunks（best-effort）
        try:
            restore_from_db(self)
            restore_rag_from_db(self)
        except Exception as e:
            logger.warning("⚠️  启动期恢复失败: %s", e)

        # 快照计数器（每 N 轮序列化 agent_state 到 PG）
        self._turn_count = 0
        self._snapshot_every = max(1, getattr(cfg, "snapshot_every_turns", 5) or 5)
        self._build_prompt_context()

        logger.info("✅ UnifiedAgent 初始化完成")

    # ── 对外 API ────────────────────────────────────────────────────────────

    def cancel(self):
        """触发所有 in-flight 请求的取消。"""
        self._cancel_registry.cancel_all()

    def process(self, query: str) -> Response:
        return self.process_with_options(query, ChatOptions(explicit=False))

    def process_with_options(self, query: str, opts: ChatOptions) -> Response:
        token, unregister = self._cancel_registry.register()
        try:
            return self._dispatch(query, opts, token)
        finally:
            unregister()

    def route(self, user_input: str, use_rag: bool = False) -> str:
        return self.process_with_options(user_input, ChatOptions(use_rag=use_rag, explicit=False)).answer

    def get_tools(self) -> List[Dict[str, Any]]:
        return self.tool_executor.get_tool_descriptions()

    def add_tool(self, tool: Tool):
        self.tool_executor.add_tool(tool)

    def register_mcp_tool(self, name: str, description: str, params: List[Dict[str, str]], func):
        self.add_tool(new_mcp_tool(name, description, params, func))

    def rag_ingest(self, document: str) -> int:
        if self.rag is None:
            return 0
        return self.rag.ingest(document)

    def rag_query(self, question: str) -> tuple:
        if self.rag is None:
            return ("RAG 不可用", [])
        return self.rag.query(question)

    def status(self) -> Dict[str, Any]:
        return build_status(self)

    def infra_status(self) -> Dict[str, str]:
        return infra_status(self)

    # ── 调度主循环 ─────────────────────────────────────────────────────────

    def _dispatch(self, query: str, opts: ChatOptions, token) -> Response:
        resp = Response(query=query)
        self.stm.add("user", query)
        self._save_chat_history("user", query)

        # 偏好/记忆抽取（同步规则 + 异步 LLM）
        async_update_memory(self, query, resp)

        mem_prefix = self._build_memory_system_prefix(query)
        hist_msgs = self._build_history_messages(query)

        if token.is_cancelled():
            resp.interrupted = True
            resp.answer = "[已中断] 请求在开始前被取消"
            return resp

        rag_loaded = bool(self.rag and getattr(self.rag, "loaded", False))

        # ── 路由 ───────────────────────────────────────────────────────────
        if opts.explicit:
            if opts.selected_tools:
                filtered = self._filter_tools(opts.selected_tools)
                if need_react_from_tools(query, filtered):
                    resp.mode = "react"
                    resp.answer, resp.steps, resp.task = self._run_react_with_tools(
                        query, filtered, mem_prefix, hist_msgs, token
                    )
                else:
                    resp.mode = "tool"
                    resp.answer, resp.tool_call = self._run_tool_from_set(
                        query, filtered, mem_prefix, hist_msgs
                    )
            elif opts.use_rag and rag_loaded:
                resp.mode = "rag"
                resp.answer, resp.search_results = self._run_rag_query(query)
            else:
                resp.mode = "chat"
                resp.answer = self._chat_response(mem_prefix, hist_msgs)
        else:
            if need_react(query):
                resp.mode = "react"
                resp.answer, resp.steps, resp.task = self._run_react_with_tools(
                    query, self.tool_executor._tool_map, mem_prefix, hist_msgs, token
                )
            elif need_tool(query):
                resp.mode = "tool"
                resp.answer, resp.tool_call = self._run_tool_from_set(
                    query, self.tool_executor._tool_map, mem_prefix, hist_msgs
                )
            elif need_rag(query, rag_loaded):
                resp.mode = "rag"
                resp.answer, resp.search_results = self._run_rag_query(query)
            else:
                resp.mode = "chat"
                resp.answer = self._chat_response(mem_prefix, hist_msgs)

        if token.is_cancelled():
            resp.interrupted = True

        # 写回短期记忆与持久化
        self.stm.add("assistant", resp.answer)
        self._save_chat_history("assistant", resp.answer)

        # 异步：从回复中提取事实 → 长期记忆
        self.memory_writer.submit(lambda: extract_memory_from_reply(self, resp.answer))
        # 异步：长期记忆合并/淘汰
        self.memory_writer.submit(lambda: maybe_consolidate_memory(self))

        # 每 N 轮快照一次 agent 状态到 PG
        self._turn_count += 1
        if self._turn_count % self._snapshot_every == 0:
            go_safe("snapshot", lambda: self._save_agent_snapshot(query, resp))

        try:
            self.inf.publish_event(
                "agent.chat",
                json.dumps({"query": query, "mode": resp.mode}, ensure_ascii=False),
            )
        except Exception:
            pass

        resp.short_term_count = self.stm.count()
        resp.long_term_count = len(self.ltm.items)
        resp.preferences = self.preference.get_all()
        return resp

    # ── Memory / Prompt 拼装 ───────────────────────────────────────────────

    def _llm_generate(self, system_prompt: str, user_msg: str) -> str:
        return self.llm.chat([Message(role="user", content=user_msg)], system_prompt=system_prompt)

    def _build_prompt_context(self) -> None:
        self.task_mem = TaskMemBuffer(20)
        self.tool_tracker = ToolStateTracker(10)
        registry = SourceRegistry()
        registry.register(ProfileSource(self.preference, self.ltm))
        registry.register(PlannerSource(self._planner_snapshot))
        registry.register(TaskMemSource(self.task_mem))
        registry.register(ToolStateSource(lambda: self.tool_executor._tool_map, self.tool_tracker))
        registry.register(ConstraintsSource([
            Policy(pattern="rm -rf", reason="禁止破坏性删除命令", level="block"),
            Policy(pattern="sudo", reason="禁止提权命令", level="block"),
        ]))
        registry.register(RecallSource(self.ltm))
        self.prompt_assembler = ContextAssembler(default_schemas(), registry)

    def _planner_snapshot(self):
        task = self._cancel_registry.current_task() if hasattr(self, "_cancel_registry") else None
        if not task:
            return None
        steps = task.get("steps") or []
        return PlannerSnapshot(
            task_id=task.get("task_id", ""),
            query=task.get("query", ""),
            status=task.get("status", ""),
            phase=task.get("phase", ""),
            total_steps=len(steps),
            current_step=task.get("current_step", 0),
            interrupted_at=task.get("interrupted_at", ""),
        )

    def _build_context_prefix(self, query: str, mode: str = "chat") -> str:
        if not hasattr(self, "prompt_assembler"):
            self._build_prompt_context()
        try:
            return self.prompt_assembler.assemble(Query(text=query, mode=mode)).render()
        except Exception as e:
            logger.warning("⚠️  promptctx 装配失败，降级到旧记忆前缀: %s", e)
            return self._build_memory_system_prefix(query)

    def push_task_mem(self, obs: StepObservation) -> None:
        if hasattr(self, "task_mem"):
            self.task_mem.push(obs)

    def record_tool_call(self, trace: ToolCallTrace) -> None:
        if hasattr(self, "tool_tracker"):
            self.tool_tracker.record(trace)

    def save_snapshot(self, task: dict) -> None:
        task_id = task.get("task_id", f"task_{int(time.time())}") if isinstance(task, dict) else f"task_{int(time.time())}"
        try:
            self.inf.save_snapshot(task_id, json.dumps(task, ensure_ascii=False))
        except Exception as e:
            logger.warning("⚠️  快照写入失败: %s", e)

    def _save_chat_history(self, role: str, content: str) -> None:
        """best-effort 写聊天记录。优先 chat_repo，其次 inf.save_chat_history。"""
        chat_repo = getattr(self, "chat_repo", None)
        if chat_repo is not None and hasattr(chat_repo, "save"):
            try:
                chat_repo.save(role, content)
                return
            except Exception:
                pass
        save_fn = getattr(self.inf, "save_chat_history", None)
        if callable(save_fn):
            try:
                save_fn(role, content)
            except Exception:
                pass

    def _build_memory_system_prefix(self, query: str = "") -> str:
        parts: List[str] = []
        prefs = self.preference.get_all()
        if prefs:
            parts.append(f"用户偏好: {json.dumps(prefs, ensure_ascii=False)}")
        memories = self.ltm.recall(query, self.cfg.long_term_top_k) if query else []
        if memories:
            parts.append("相关记忆:\n" + "\n".join(f"- {m.content}" for m in memories))
        return "\n".join(parts)

    def _recent_history_for_rag(self) -> List[HistoryMessage]:
        return [HistoryMessage(role=m["role"], content=m["content"]) for m in self.stm.get()]

    def _run_rag_query(self, query: str):
        if self.rag is None:
            return "RAG 不可用", []
        if hasattr(self.rag, "query_with_history"):
            return self.rag.query_with_history(query, self._recent_history_for_rag())
        return self.rag.query(query)

    def _build_history_messages(self, query: str) -> List[Message]:
        msgs = [Message(role=m["role"], content=m["content"]) for m in self.stm.get()]
        if not msgs or msgs[-1].content != query:
            msgs.append(Message(role="user", content=query))
        return msgs

    def _chat_response(self, mem_prefix: str, hist_msgs: List[Message]) -> str:
        system_prompt = "你是一个简洁的AI助手。结合你掌握的用户信息，使回答更个性化。"
        if mem_prefix:
            system_prompt = mem_prefix + "\n\n" + system_prompt
        return self.llm.chat(hist_msgs, system_prompt=system_prompt)

    # ── 工具调用（tool 模式） ──────────────────────────────────────────────

    def _filter_tools(self, names: List[str]) -> Dict[str, Tool]:
        return {n: self.tool_executor._tool_map[n] for n in names if n in self.tool_executor._tool_map}

    def _parse_tool_params(self, tool_name: str, user_input: str) -> Dict[str, str]:
        params: Dict[str, str] = {}
        if tool_name == "get_weather":
            match = re.search(r"(天气|温度)\s*([^\s,，。？?!！]+)", user_input)
            if match:
                params["city"] = match.group(2)
        elif tool_name in {"search_web", "rag_search"}:
            match = re.search(r"(搜索|查找|知识|文档)\s*(.*)", user_input)
            if match and match.group(2).strip():
                params["query"] = match.group(2).strip()
            else:
                params["query"] = user_input
        return params

    def _run_tool_from_set(self, query: str, tools_map: Dict[str, Tool], mem_prefix: str, hist_msgs: List[Message]):
        tool_name = detect_tool(query, tools_map)
        if not tool_name:
            return self._chat_response(mem_prefix, hist_msgs), None
        params = self._parse_tool_params(tool_name, query)
        result = self.tool_executor.call(tool_name, params)
        answer = result.content if result.success else f"工具调用失败: {result.error}"
        tool_call = {
            "tool_name": tool_name,
            "params": params,
            "tool_result": result.content,
            "success": result.success,
            "error": result.error,
        }
        return answer, tool_call

    # ── ReAct 推理循环（保留原 agent.py 的实现） ───────────────────────────

    def _run_react_with_tools(self, query: str, tools_map: Dict[str, Tool], mem_prefix: str, hist_msgs: List[Message], token):
        task = {"task_id": f"task_{int(time.time())}", "query": query, "status": "running", "steps": []}
        self._cancel_registry.set_task(task)
        plan_nodes = llm_plan_graph(self, query, tools_map, mem_prefix)
        if plan_nodes:
            from internal.graph.task_graph import TaskGraph

            graph = TaskGraph(plan_nodes)
            try:
                graph.validate()
            except Exception:
                for node in plan_nodes:
                    node.depends_on = []
                graph = TaskGraph(plan_nodes)
            cfg = GraphConfig(
                max_parallel=getattr(self.cfg, "graph_max_parallel", 2),
                race_timeout_ms=getattr(self.cfg, "graph_race_timeout_ms", 30000),
                enable_racing=getattr(self.cfg, "graph_enable_racing", True),
            )
            result = GraphRuntime(graph, self, cfg, tools_map, task).execute(token)
            steps = [
                ReActStep(
                    type=StepType.OBSERVATION,
                    content=node.result or node.error,
                    tool=node.tool_name,
                    params=node.params,
                )
                for node in graph.nodes.values()
            ]
            final_answer = self._generate_final_answer(query, steps, mem_prefix)
            steps.append(ReActStep(type=StepType.FINAL_ANSWER, content=final_answer))
            task["status"] = "interrupted" if result.interrupted else "completed"
            task["graph"] = {
                "nodes": [
                    {
                        "id": node.id,
                        "tool": node.tool_name,
                        "status": str(node.status),
                        "result": node.result,
                        "error": node.error,
                    }
                    for node in graph.nodes.values()
                ]
            }
            self._cancel_registry.set_task(None)
            return self._format_react_response(steps), steps, task

        steps: List[ReActStep] = []

        for _ in range(self.max_iterations):
            if token.is_cancelled():
                task["status"] = "interrupted"
                return "[已中断]", steps, task
            thought = self._generate_thought(query, steps, mem_prefix, tools_map)
            steps.append(ReActStep(type=StepType.THOUGHT, content=thought))
            if self._is_complete(thought):
                final_answer = self._generate_final_answer(query, steps, mem_prefix)
                steps.append(ReActStep(type=StepType.FINAL_ANSWER, content=final_answer))
                break
            action, tool_name, params = self._parse_action(thought, tools_map)
            if not tool_name:
                final_answer = self._generate_final_answer(query, steps, mem_prefix)
                steps.append(ReActStep(type=StepType.FINAL_ANSWER, content=final_answer))
                break
            if tool_name not in tools_map:
                steps.append(ReActStep(type=StepType.OBSERVATION, content=f"工具 {tool_name} 不可用"))
                break
            result = self._call_tool_with_retry(tool_name, params)
            steps.append(ReActStep(type=StepType.ACTION, content=action, tool=tool_name, params=params))
            steps.append(
                ReActStep(
                    type=StepType.OBSERVATION,
                    content=result.content if result.success else f"失败: {result.error}",
                )
            )
            self._save_snapshot(task["task_id"], steps)
        else:
            # 达到最大迭代后强制总结
            final_answer = self._generate_final_answer(query, steps, mem_prefix)
            steps.append(ReActStep(type=StepType.FINAL_ANSWER, content=final_answer))

        answer = self._format_react_response(steps)
        task["status"] = "completed"
        task["steps"] = [
            {"type": s.type, "content": s.content, "tool": s.tool, "params": s.params} for s in steps
        ]
        # 任务结束再做一次快照
        self._save_snapshot(task["task_id"], steps)
        self._cancel_registry.set_task(None)
        return answer, steps, task

    def _call_tool_with_retry(self, tool_name: str, params: Dict[str, str]):
        last = None
        for _ in range(max(1, self.max_retries)):
            last = self.tool_executor.call(tool_name, params)
            if last.success:
                return last
            time.sleep(self.cfg.retry_delay_ms / 1000.0)
        return last

    def _generate_thought(self, query: str, steps: List[ReActStep], mem_prefix: str, tools_map: Dict[str, Tool]) -> str:
        steps_str = "\n".join(f"{s.type}: {s.content}" for s in steps) or "（暂无）"
        tools_desc = "\n".join(
            f"- {name}: {tool.description}; 参数: {[p['name'] for p in tool.params]}"
            for name, tool in tools_map.items()
        )
        prompt = f"""你是一个 ReAct 推理助手。请基于任务和已有步骤决定下一步。

可用工具：
{tools_desc}

任务: {query}

记忆上下文:
{mem_prefix or '（无）'}

历史步骤:
{steps_str}

请输出一条决策，必须严格满足以下两种格式之一：
1) 需要调用工具：    Action: tool_name(key1="value1", key2="value2")
2) 已可给出答案：    Final: <最终结论>

不要输出额外内容。
"""
        messages = [Message(role="user", content=prompt)]
        return self.llm.chat(messages, system_prompt="你是一个擅长推理的助手，按要求格式输出决策。")

    def _is_complete(self, thought: str) -> bool:
        return bool(re.search(r"^\s*Final\s*[:：]", thought, re.MULTILINE))

    def _parse_action(self, thought: str, tools_map: Dict[str, Tool]):
        m = re.search(r"Action\s*[:：]\s*([a-zA-Z_][\w]*)\s*\((.*?)\)", thought, re.DOTALL)
        if not m:
            return "", "", {}
        tool_name = m.group(1)
        if tool_name not in tools_map:
            return "", "", {}
        params: Dict[str, str] = {}
        raw = m.group(2).strip()
        if raw:
            for pair in re.split(r",(?![^\"]*\")", raw):
                if "=" not in pair:
                    continue
                k, v = pair.split("=", 1)
                v = v.strip().strip('"').strip("'")
                params[k.strip()] = v
        return f"调用工具 {tool_name}", tool_name, params

    def _generate_final_answer(self, query: str, steps: List[ReActStep], mem_prefix: str) -> str:
        steps_str = "\n".join(f"{s.type}: {s.content}" for s in steps)
        prompt = f"""基于以下推理过程，给出最终答案。

任务: {query}

记忆上下文:
{mem_prefix or '（无）'}

推理过程:
{steps_str}

请用自然语言总结最终答案，不要包含 Action/Final 等关键字。
"""
        messages = [Message(role="user", content=prompt)]
        return self.llm.chat(messages, system_prompt="你是一个总结助手，能够基于推理过程给出简洁的最终答案。")

    def _format_react_response(self, steps: List[ReActStep]) -> str:
        lines = []
        for s in steps:
            if s.type == StepType.THOUGHT:
                lines.append(f"💭 {s.content}")
            elif s.type == StepType.ACTION:
                lines.append(f"⚡ {s.tool}({s.params})")
            elif s.type == StepType.OBSERVATION:
                lines.append(f"👁 {s.content}")
            elif s.type == StepType.FINAL_ANSWER:
                lines.append(f"\n📝 最终答案:\n{s.content}")
        return "\n".join(lines)

    def _save_snapshot(self, task_id: str, steps: List[ReActStep]):
        snapshot = {
            "task_id": task_id,
            "steps": [{"type": s.type, "content": s.content, "tool": s.tool, "params": s.params} for s in steps],
            "timestamp": time.time(),
        }
        try:
            self.inf.save_snapshot(task_id, json.dumps(snapshot, ensure_ascii=False))
        except Exception as e:
            logger.warning("⚠️  快照写入失败: %s", e)

    def _save_agent_snapshot(self, query: str, resp: Response):
        """每 N 轮把 agent 整体状态序列化到 PG（含路由 mode/计数/偏好）。"""
        snapshot = {
            "task_id": f"agent_{int(time.time())}",
            "query": query,
            "mode": resp.mode,
            "short_term_count": resp.short_term_count,
            "long_term_count": resp.long_term_count,
            "preferences": resp.preferences,
            "timestamp": time.time(),
        }
        try:
            self.inf.save_snapshot(snapshot["task_id"], json.dumps(snapshot, ensure_ascii=False))
        except Exception as e:
            logger.warning("⚠️  agent 快照写入失败: %s", e)

    # ── 生命周期 ────────────────────────────────────────────────────────────

    def close(self):
        try:
            self.memory_writer.stop()
        except Exception:
            pass
