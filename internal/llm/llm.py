import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

from config.config import APIConfig

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """对话消息"""
    role: str      # "user","system"
    content: str   # 文本内容

# Mock 向量维度（无 API Key 时使用）；务必与 cfg.rag_milvus_dim / config.yaml 中真实模型维度区分
MOCK_EMBED_DIM = 768

class client:
    """LLM 客户端封装：OpenAI 兼容 Chat + Embedding"""
    def __init__(self,cfg: APIConfig):
        self.cfg = cfg
        self._timeout = 60 # 超时时间
        self._mock_response = {
            "你是谁": "我是一个全能 AI 助手，具备知识库、工具调用、推理、记忆和稳定执行能力。",
            "后端工程师": "后端工程师负责服务器端逻辑开发：API 设计、数据库、业务逻辑、系统架构、性能优化。",
        }

    def chat(self, messages: List[Message], system_prompt: str = "") -> str:
        """
        发送对话请求

        参数：
            - messages: 对话历史
            - system_prompt: 系统提示词
        返回：
            - LLM 的回复文本
        """
        if not self.cfg.is_real_llm():
            return self._mock_response

        try:
            return self._call_chat(messages, system_prompt)
        except Exception as e:
            logging.error("LLM API调用失败：%s",e)
            return self._mock(messages)

    def chat_context(self, ctx, system_prompt: str, messages: List[Message]) -> str:
        """兼容主分支 Go 版 ChatContext。"""
        if getattr(ctx, "cancelled", False):
            return "[已中断]"
        return self.chat(messages, system_prompt=system_prompt)

    def _call_chat(self, messages: List[Message], system_prompt: str = "") -> str:
        msgs = List[Dict[str,str]]
        if system_prompt:
            msgs.append({"role":"system","content":system_prompt})
        msgs.extend({"role" : m.role,"content" : m.content} for m in messages)

        payload = {
            "model": self.cfg.llm_model,
            "messages": msgs,
            "temperature": self.cfg.temperature,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.llm_api_key}",
        }

        resp = requests.post(self.cfg.llm_api_url, headers=headers, json=payload, timeout=self._timeout)

        if resp.status_code != 200:
            raise RuntimeError(f"API 返回错误状态 {resp.status_code}, body: {resp.text}")
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"API 错误: {data['error'].get('message')}")
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"API 返回空结果, body: {resp.text}")

        return choices[0].get("message", {}).get("content", "")

# ── Embedding ───────────────────────────────────────────────────────────
    def embed(self, text: str) -> List[float]:
        """文本向量化；未配置或调用失败时返回 Mock 向量。"""
        if not self.cfg.is_real_embedding():
            return self._mock_embed(text)
        try:
            return self._call_embed(text)
        except Exception as e:
            logger.error("Embedding API 调用失败: %s，回退到 Mock", e)
            return self._mock_embed(text)

    def _call_embed(self, text: str) -> List[float]:
        api_url = self.cfg.embedding_api_url
        is_multimodal = "/embeddings/multimodal" in api_url

        # 火山方舟多模态 embedding 端点：input 为结构化数组，data 为单对象
        if is_multimodal:
            input_payload = [{"type": "text", "text": text}]
        else:
            input_payload = text

        payload = {"model": self.cfg.embedding_model, "input": input_payload}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.embedding_api_key}",
        }
        resp = requests.post(api_url, headers=headers, json=payload, timeout=self._timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"embedding API 返回错误状态 {resp.status_code}, body: {resp.text}")
        result = resp.json()
        if result.get("error"):
            raise RuntimeError(f"embedding API 错误: {result['error'].get('message')}")

        if is_multimodal:
            embedding = (result.get("data") or {}).get("embedding") or []
        else:
            data_list = result.get("data") or []
            if not data_list:
                raise RuntimeError("embedding 返回空结果")
            embedding = data_list[0].get("embedding") or []

        if not embedding:
            raise RuntimeError("embedding 返回空向量")
        return embedding

    def _mock_embed(self, text: str) -> List[float]:
        h = hash(text)
        return [((h * (i + 1)) % 1000) / 1000.0 for i in range(MOCK_EMBED_DIM)]

    def extract_preferences(self, msg: str) -> Dict[str, str]:
        """对齐 main 分支：优先用 LLM 抽取偏好 JSON，失败时规则兜底。"""
        if not msg:
            return {}
        if not self.cfg.is_real_llm():
            return _extract_rule_based(msg)

        prompt = (
            "从下面这句用户消息中，提取所有用户的个人信息和偏好，"
            "输出 JSON 对象（key 为中文名称，value 为具体值）。"
            "如果没有任何偏好信息，输出 {}。只输出 JSON，不要有其他内容。\n\n"
            f"消息：{msg}"
        )
        try:
            raw = self._call_chat("", [Message(role="user", content=prompt)])
        except Exception:
            return _extract_rule_based(msg)

        raw = raw.strip()
        for prefix in ("```json", "```"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except Exception:
            pass
        return _extract_rule_based(msg)


    def _mock(self, messages: List[Message]) -> str:
        user_query = ""
        for m in messages:
            if m.role == "user":
                user_query = m.content
        q = user_query.lower()
        for key, response in self._mock_responses.items():
            if key in q:
                return response
        return f"收到：「{user_query}」——这是模拟 LLM 回复，接入真实 API 后会更智能。"

def _extract_rule_based(msg: str) -> Dict[str, str]:
    """规则兜底，与 Go 版 extractRuleBased 一致。"""
    result: Dict[str, str] = {}
    if "我喜欢" in msg:
        parts = msg.split("喜欢", 1)
        if len(parts) == 2 and parts[1].strip():
            result["喜好"] = parts[1].strip()
    elif "我爱" in msg:
        parts = msg.split("爱", 1)
        if len(parts) == 2 and parts[1].strip():
            result["喜好"] = parts[1].strip()
    if "我叫" in msg:
        parts = msg.split("叫", 1)
        if len(parts) == 2 and parts[1].strip():
            result["姓名"] = parts[1].strip()
    return result

