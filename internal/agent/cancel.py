# cancel — 并发执行 helper：取消令牌注册表 + 当前任务持锁访问 + go_safe
#
# 对应 Go 版 internal/agent/cancel.go：
#   - cancelFns map：每个 in-flight 请求一个 token，Cancel() 触发全部
#   - currentTask / setTask：持锁访问当前 TaskState
#   - goSafe：带 panic recover 的后台 goroutine 启动器
#
# Python 用 threading.Event 取代 context.CancelFunc。每次请求注册一个独立
# 的 Event，agent.cancel() 会触发所有 in-flight 请求的 Event。
import logging
import threading
import traceback
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class CancelToken:
    """单次请求的取消令牌。基于 threading.Event，is_cancelled()/cancel() 幂等。"""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


class CancelRegistry:
    """所有 in-flight 请求的 CancelToken 注册表（对应 Go 的 cancelFns map）。

    每个请求开始时 register() 拿到一个 token，结束时调用返回的 unregister。
    cancel_all() 会触发所有 in-flight 的 token，对应 Go 的 Cancel()。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._tokens: Dict[int, CancelToken] = {}
        self._next_id = 0
        # 当前正在执行的 task 状态（仅保留一个引用，用于 PlannerSource 读取）
        self._current_task: Optional[dict] = None

    def register(self) -> tuple:
        """注册一个新的 CancelToken，返回 (token, unregister)。"""
        token = CancelToken()
        with self._lock:
            self._next_id += 1
            tid = self._next_id
            self._tokens[tid] = token

        def _unregister():
            with self._lock:
                self._tokens.pop(tid, None)
            token.cancel()  # 幂等

        return token, _unregister

    def cancel_all(self):
        """触发所有 in-flight 请求的 cancel（对应 Go Cancel()）。"""
        with self._lock:
            tokens = list(self._tokens.values())
        for t in tokens:
            t.cancel()

    def current_task(self) -> Optional[dict]:
        with self._lock:
            return self._current_task

    def set_task(self, task: Optional[dict]):
        """设置当前 task，对应 Go setTask。"""
        with self._lock:
            self._current_task = task


def go_safe(name: str, fn: Callable[[], None]) -> threading.Thread:
    """启动一个带 panic recover 的后台线程（对应 Go 的 goSafe）。

    所有异步任务（偏好提取/记忆挖掘/记忆合并/KG 异步写）走这里，
    任意线程崩溃都不会影响主流程。
    """
    def _wrapped():
        try:
            fn()
        except Exception as e:
            logger.warning("⚠️  background thread panic [%s]: %s\n%s", name, e, traceback.format_exc())

    t = threading.Thread(target=_wrapped, name=f"go-safe:{name}", daemon=True)
    t.start()
    return t
