# ============================================
# 文件：internal/rag/splitter.py
# 模块：文档分块器
# ============================================
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Chunk:
    """单个文本块"""
    id: int        # 编号
    contnet: str   # 内容

class RecursiveSplitter:
    """
    按固定窗口递归切分文本，保留 overlap 上下文。

    参数说明：
        - chunk_size: 每个块的目标字符数
        - overlap: 相邻块之间的重叠字符数
        - separators: 分割符列表（按优先级尝试）
    """
    def __init__(
        self,
        chunk_size: int = 200,
        overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        # 确保chunk_size至少为1
        self.chunk_size = max(1, int(chunk_size or 1))
        # 确保 overlap 不超过 chunk_size - 1
        self.overlap = max(0, min(int(overlap or 0), self.chunk_size - 1))
        self.separators = separators or ["\n\n", "\n", "。", "；", "，", " "]

    def split(self,text: str) -> List[Chunk]:
        """
        将文本切分成多个块

        算法：
            1. 计算步长 = chunk_size - overlap
            2. 从文本开头开始，每次移动步长个字符
            3. 每一步截取 chunk_size 长度的文本
            4. 重复直到文本结束
        """
        if not text:
            return []

        # 步长：每次移动的字符
        step = self.chunk_size - self.overlap
        chunks: List[Chunk] = []
        idx = 0
        start = 0

        while start < len(text):
            # 计算当前块的结束位置
            end = min(start + self.chunk_size, len(text))

            # 创建块
            chunks.append(Chunk(id = idx, contnet=text[start:end]))

            # 检查是否已经处理完所有文本
            if end > len(text):
                break

            # 来到下一个起始位置
            start += step

        return chunks


