from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

class NodeType(str, Enum):
    TOOL = "tool"
    LLM = "llm"

class NodeStatus(str, Enum):
    PENDING = "pending"      # 等待执行
    RUNNING = "running"     # 执行中
    DONE = "done"           # 执行成功
    FAILED = "failed"       # 执行失败
    CANCELLED = "cancelled" # 被取消
    SKIPPED = "skipped"     # 被跳过（race 赢家）

@dataclass
class Node:
    id: str                           # 节点唯一 ID，如 "n1", "n2"
    type: NodeType = NodeType.TOOL   # 节点类型，默认 TOOL
    name: str = ""                   # 节点名称（用于显示）
    tool_name: str = ""              # 工具名称（实际要调用的工具名）
    params: Dict[str, str] = field(default_factory=dict)  # 工具参数字典
    depends_on: List[str] = field(default_factory=list)   # 依赖的节点 ID 列表
    race_group: str = ""             # 竞速组名（空=不参与竞速）
    status: NodeStatus = NodeStatus.PENDING  # 当前状态
    result: str = ""                 # 执行结果
    error: str = ""                  # 错误信息
    retry_count: int = 0            # 当前重试次数


class TaskGraph:
    """有向无环任务图，通过拓扑层级决定并行调度顺序。"""

    def __init__(self, nodes: List[Node]):
        self.nodes: Dict[str, Node] = {node.id: node for node in nodes}
        self.adj: Dict[str, List[str]] = {node.id: [] for node in nodes} # 该节点指向的子节点 （出边）
        self.indegree: Dict[str, int] = {node.id: 0 for node in nodes}  # 有多少条边指向该节点（入度）
        for node in nodes:
            for dep in node.depends_on or []: # 遍历该节点依赖的每个前置节点
                if dep in self.adj:
                    self.adj[dep].append(node.id) # dep → node.id 的一条边
                    self.indegree[node.id] = self.indegree.get(node.id, 0) + 1  # node.id 的入度 +1

    def validate(self) -> None:
        for node in self.nodes.values():
            for dep in node.depends_on or []:
                if dep not in self.nodes:
                    raise ValueError(f"missing dependency {dep} for node {node.id}")
        self.topological_levels()

    def topological_levels(self) -> List[List[str]]:
        self._check_missing_dependencies()
        indegree = dict(self.indegree)  # 复制入度表（不修改原数据）
        ready = sorted([node_id for node_id, degree in indegree.items() if degree == 0])
        levels: List[List[str]] = []
        visited = 0

        while ready:
            level = ready
            levels.append(level)
            visited += len(ready)

        next_ready: List[str] = []
        for node_id in level:
            for child in sorted(self.adj.get(node_id, [])):
                indegree[child] -= 1 # 子节点的入度减 1（因为它的一个前置节点已处理完）
                if indegree[child] == 0: # 入度变为 0 → 这个子节点可以被执行了，加入 next_ready
                    next_ready.append(child)
        ready = sorted(next_ready)
        if visited != len(self.nodes):
            raise ValueError("cycle detected in task graph")
        return levels

    def ready_nodes(self) -> List[str]:
        ready: List[str] = []
        for node_id, node in self.nodes.items():
            if node.status != NodeStatus.PENDING:  # 非等待状态跳过
                continue
            deps_done = all(self.nodes[dep].status == NodeStatus.DONE for dep in node.depends_on)
            if deps_done:
                ready.append(node_id)
        return sorted(ready)

    def mark_done(self, node_id: str) -> List[str]:
        self.set_node_status(node_id, NodeStatus.DONE)
        return self.ready_nodes()

    def race_groups(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for node in self.nodes.values():
            if not node.race_group:
                continue
            groups.setdefault(node.race_group, []).append(node.id)
        return groups

    def set_node_status(self, node_id: str, status: NodeStatus) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].status = status

    def set_node_result(self, node_id: str, result: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].result = result
            self.nodes[node_id].status = NodeStatus.DONE

    def set_node_error(self, node_id: str, error: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].error = error
            self.nodes[node_id].status = NodeStatus.FAILED

    def set_node_retry_count(self, node_id: str, count: int) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].retry_count = count

    def successful_results(self) -> List[str]:
        return [node.result for node in self.nodes.values() if node.status == NodeStatus.DONE and node.result]

    def summary(self) -> str:
        parts = []
        for node in self.nodes.values():
            parts.append(f"{node.id}:{node.tool_name}:{node.status}")
        return "\n".join(parts)

    def _check_missing_dependencies(self) -> None:
        for node in self.nodes.values():
            for dep in node.depends_on or []:
                if dep not in self.nodes:
                    raise ValueError(f"missing dependency {dep} for node {node.id}")





