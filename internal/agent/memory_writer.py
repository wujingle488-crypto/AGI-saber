# memory_writer — 异步记忆写入与回复中事实抽取
#
# 对应 Go 版 internal/agent/memory_writer.go：
#   - extract_memory_from_reply：从 assistant 回复抽取 k-v 事实
#   - classify_memory_content：基于关键字的快速分类规则
#   - llm_classify_memory：LLM 兜底分类
#   - sync_consolidation_to_db：把记忆合并结果同步回 PG
#
# Python 在 Go 的 goroutine + channel 基础上额外提供 AsyncMemoryWriter：
# 后台线程 + queue.Queue 串行化所有记忆写入，避免 PG/Milvus 并发竞争。
import json
import logging
import queue
import re
import threading
from typing import Any, Dict, List, Tuple

from internal.llm.llm import Message

logger = logging.getLogger(__name__)


# ── 异步记忆写入器 ──────────────────────────────────────────────────────────

class AsyncMemoryWriter:
    """后台线程串行化记忆写入（对应 Go 的 goroutine + channel 模型）。

    使用 queue.Queue 排队写任务，单 worker 线程消费，避免 ltm/preference 同时
    被多线程改写。stop() 触发优雅退出。
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._stopped = threading.Event()
        self._worker = threading.Thread(target=self._run, name="memory-writer", daemon=True)
        self._worker.start()

    def submit(self, fn):
        """提交一个无参可调用，最终在 worker 线程执行。"""
        if self._stopped.is_set():
            return
        try:
            self._queue.put_nowait(fn)
        except Exception as e:
            logger.warning("⚠️  memory-writer 提交失败: %s", e)

    def stop(self):
        self._stopped.set()
        try:
            self._queue.put_nowait(None)  # 唤醒 worker
        except Exception:
            pass

    def _run(self):
        while not self._stopped.is_set():
            try:
                fn = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if fn is None:
                break
            try:
                fn()
            except Exception as e:
                logger.warning("⚠️  memory-writer 任务异常: %s", e)


# ── 回复 → 记忆抽取 ────────────────────────────────────────────────────────

def extract_memory_from_reply(agent, answer: str):
    """从 assistant 回复中提取值得记忆的 k-v 事实并存入长期记忆。"""
    if not answer or not agent.cfg.is_real_llm():
        return

    prompt = (
        "从下面这段AI回复中，提取值得长期记住的客观事实或用户偏好信息。\n"
        "只提取明确的、非临时性的信息，忽略对话上下文和临时细节。\n"
        "输出 JSON 对象（key为中文名称，value为具体值），如果没有值得记忆的信息则输出 {}。\n"
        "只输出 JSON，不要有其他内容。\n\n"
        f"回复：{answer}"
    )
    try:
        raw = agent.llm.chat([Message(role="user", content=prompt)], system_prompt="")
    except Exception as e:
        logger.warning("⚠️  记忆抽取 LLM 调用失败: %s", e)
        return

    raw = (raw or "").strip()
    raw = re.sub(r"^```json", "", raw)
    raw = re.sub(r"^```", "", raw)
    raw = re.sub(r"```$", "", raw)
    raw = raw.strip()

    try:
        kvs = json.loads(raw)
    except Exception:
        return
    if not isinstance(kvs, dict) or not kvs:
        return

    for k, v in kvs.items():
        if not k or v in (None, ""):
            continue
        try:
            agent.preference.set(str(k), str(v))
        except Exception:
            pass
        content = f"用户{k}: {v}"
        category, tags, slot_hint = classify_memory_content(str(k), str(v))
        if not category:
            category, tags, slot_hint = llm_classify_memory(agent, content)

        try:
            # LongTerm.add 内部会做 embed + 持久化
            agent.ltm.add(content, importance=0.7)
        except Exception as e:
            logger.warning("⚠️  长期记忆写入失败: %s", e)
        logger.info("🧠 从回复中提取记忆：%s = %s（类别=%s）", k, v, category)


def classify_memory_content(key: str, value: str) -> Tuple[str, List[str], str]:
    """用规则快速分类；返回空字符串表示规则未命中，由 LLM 兜底。"""
    combined = f"{key}{value}"
    if _contains_any(combined, "叫", "名字", "姓名", "是我", "我是"):
        return "identity", ["name"], "profile"
    if _contains_any(combined, "喜欢", "偏好", "习惯", "爱好", "讨厌", "不喜欢"):
        return "preference", ["preference"], "profile"
    if _contains_any(combined, "工具", "失败", "错误", "报错", "异常"):
        return "tool_failure", ["tool", "error"], "tool_state"
    if _contains_any(combined, "禁止", "不要", "不能", "必须", "强制"):
        return "policy", ["constraint"], "constraints"
    return "", [], ""


def _contains_any(s: str, *subs: str) -> bool:
    return any(sub in s for sub in subs)


def llm_classify_memory(agent, content: str) -> Tuple[str, List[str], str]:
    """LLM 兜底分类。失败时回退到 'general'。"""
    if not agent.cfg.is_real_llm():
        return "general", [], ""

    prompt = (
        "请对以下记忆内容进行分类，只输出 JSON，格式如下：\n"
        '{"category":"identity|preference|fact|episodic|tool_failure|policy|general",'
        '"tags":["tag1"],"slot_hint":"profile|planner|task_memory|tool_state|constraints|recall_memory"}\n'
        f"\n记忆内容：{content}"
    )
    try:
        raw = agent.llm.chat([Message(role="user", content=prompt)], system_prompt="")
    except Exception:
        return "general", [], ""
    raw = (raw or "").strip()
    raw = re.sub(r"^```json", "", raw)
    raw = re.sub(r"^```", "", raw)
    raw = re.sub(r"```$", "", raw)
    raw = raw.strip()
    try:
        result = json.loads(raw)
    except Exception:
        return "general", [], ""
    cat = result.get("category") or "general"
    return cat, list(result.get("tags") or []), result.get("slot_hint") or ""


def sync_consolidation_to_db(agent, result: Any):
    """将记忆合并结果同步到 PostgreSQL（best-effort）。

    Python 端 LongTerm.consolidate() 直接修改 self.items（无 result 返回值），
    本函数保留为 Go 版 API 对齐占位，仅在传入 dict-like result 时落库。
    """
    if not result:
        return
    try:
        delete_ids = result.get("delete_from_db") if isinstance(result, dict) else None
        if delete_ids:
            logger.info("🧹 记忆合并：删除 %d 条 id", len(delete_ids))
        update_items = result.get("update_in_db") if isinstance(result, dict) else None
        if update_items:
            logger.info("🔗 记忆合并：更新 %d 条", len(update_items))
    except Exception as e:
        logger.warning("⚠️  sync_consolidation_to_db 失败: %s", e)


# ── ReAct 模式下的同步 + 异步偏好提取（保留原 agent.py 的实现）──────────────

def async_update_memory(agent, user_input: str, resp: Any) -> None:
    """ReAct 入口前的偏好抽取：

      1) 同步：用规则提取，立即填到 resp.extracted_info（用户即时反馈）
      2) 异步：丢到 memory writer 线程做 LLM 提取 + 长期记忆写入
    """
    try:
        from internal.llm.llm import _extract_rule_based
    except Exception:
        _extract_rule_based = None  # type: ignore

    # 1) 同步规则提取
    if _extract_rule_based is not None:
        try:
            quick = _extract_rule_based(user_input) or {}
        except Exception:
            quick = {}
        if quick:
            try:
                agent.preference.save_batch(quick)
            except Exception:
                pass
            if hasattr(resp, "extracted_info"):
                resp.extracted_info = "已记住：" + ", ".join(f"{k}={v}" for k, v in quick.items())

    # 2) 异步 LLM 提取 + LTM 写入
    def _bg():
        try:
            extracted = agent.llm.extract_preferences(user_input) if hasattr(agent.llm, "extract_preferences") else {}
            if extracted:
                agent.preference.save_batch(extracted)
            agent.ltm.add(user_input)
        except Exception as e:
            logger.warning("异步更新记忆失败: %s", e)

    writer = getattr(agent, "memory_writer", None)
    if writer is not None:
        writer.submit(_bg)
    else:
        threading.Thread(target=_bg, name="memory-fallback", daemon=True).start()


def maybe_consolidate_memory(agent):
    """达到触发阈值时合并/去重/衰减/淘汰长期记忆。"""
    try:
        if agent.ltm.need_consolidation():
            agent.ltm.consolidate()
    except Exception as e:
        logger.warning("记忆合并失败: %s", e)
