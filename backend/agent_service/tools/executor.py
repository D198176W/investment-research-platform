"""工具执行器 —— 结果理解（校验/截断/摘要）+ 异常分类与失败恢复

异常分类策略（对齐 JD「失败恢复效果」）：
- ToolFatalError     : 参数非法 → 不重试，错误信息直接回喂模型让其换参
- ToolRetryableError : 网络/超时/空数据 → 指数退避重试，仍失败则回喂模型换工具
- 其他未知异常        : 保守视为可重试一次后放弃
"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Dict, Optional

from .registry import Tool, registry, validate_arguments
from .market_tools import ToolFatalError, ToolRetryableError  # noqa: F401  (re-export)

logger = logging.getLogger(__name__)

MAX_TOOL_RETRIES = 2  # 工具级重试次数（不含首次）


class ToolExecutionResult:
    """工具执行结果（给模型的回喂内容统一为字符串）"""

    def __init__(self, ok: bool, content: str, tool_name: str,
                 latency_ms: int, attempts: int, error_type: Optional[str] = None,
                 truncated: bool = False):
        self.ok = ok
        self.content = content
        self.tool_name = tool_name
        self.latency_ms = latency_ms
        self.attempts = attempts
        self.error_type = error_type
        self.truncated = truncated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "tool_name": self.tool_name,
            "content": self.content,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "error_type": self.error_type,
            "truncated": self.truncated,
        }


def _summarize_result(data: Any, max_chars: int) -> tuple[str, bool]:
    """结果理解：JSON 序列化 → 超长截断（优先压缩列表长度，保留头部信息）"""
    try:
        text = json.dumps(data, ensure_ascii=False, default=str)
    except Exception:
        text = str(data)

    if len(text) <= max_chars:
        return text, False

    # 截断策略：若 data 含长列表，先裁剪列表再序列化
    def _trim(obj: Any, budget: int) -> Any:
        if isinstance(obj, list):
            # 逐步减半直到序列化后能放入预算的一半
            trimmed = obj
            while trimmed and len(json.dumps(trimmed, ensure_ascii=False, default=str)) > budget:
                trimmed = trimmed[: max(1, len(trimmed) // 2)]
            return trimmed + ([{"_truncated": f"剩余 {len(obj) - len(trimmed)} 条已省略"}] if len(trimmed) < len(obj) else [])
        if isinstance(obj, dict):
            return {k: _trim(v, budget // max(1, len(obj))) for k, v in obj.items()}
        return obj

    trimmed_data = _trim(data, int(max_chars * 0.8))
    text = json.dumps(trimmed_data, ensure_ascii=False, default=str)
    if len(text) > max_chars:
        text = text[:max_chars] + "...(已截断)"
    return text, True


def execute_tool(tool_name: str, arguments: Dict[str, Any],
                 tool: Optional[Tool] = None) -> ToolExecutionResult:
    """执行单个工具调用（含参数校验、超时控制、异常分类重试、结果截断）

    Args:
        tool_name: 工具名
        arguments: 模型生成的参数 dict
        tool: 可选，直接传入 Tool 对象（MCP 模式下由 client 解析后传入）

    Returns:
        ToolExecutionResult
    """
    started = time.time()
    tool = tool or registry.get(tool_name)
    if tool is None:
        return ToolExecutionResult(
            ok=False, tool_name=tool_name, latency_ms=0, attempts=0,
            error_type="tool_not_found",
            content=f"工具不存在: {tool_name}。可用工具: {', '.join(registry.tool_names())}",
        )

    # 参数校验（护栏：非法参数直接回喂，不进入执行）
    arg_errors = validate_arguments(tool, arguments or {})
    if arg_errors:
        return ToolExecutionResult(
            ok=False, tool_name=tool_name, latency_ms=int((time.time() - started) * 1000),
            attempts=0, error_type="invalid_arguments",
            content=f"参数校验失败: {'; '.join(arg_errors)}。请修正参数后重试。",
        )

    attempts = 0
    last_error: Optional[Exception] = None
    for attempt in range(MAX_TOOL_RETRIES + 1):
        attempts = attempt + 1
        if attempt > 0:
            delay = 1.0 * (2 ** (attempt - 1))
            logger.info(f"工具 {tool_name} 第 {attempt} 次重试，退避 {delay}s")
            time.sleep(delay)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(tool.handler, **(arguments or {}))
                raw = future.result(timeout=tool.timeout)
            content, truncated = _summarize_result(raw, tool.max_result_chars)
            return ToolExecutionResult(
                ok=True, tool_name=tool_name, content=content,
                latency_ms=int((time.time() - started) * 1000),
                attempts=attempts, truncated=truncated,
            )
        except FutureTimeoutError:
            last_error = ToolRetryableError(f"执行超时({tool.timeout}s)")
            logger.warning(f"工具 {tool_name} 超时（第 {attempts} 次）")
        except ToolFatalError as e:
            # 不可重试：立即返回给模型
            return ToolExecutionResult(
                ok=False, tool_name=tool_name, content=f"参数或数据不可用: {e}",
                latency_ms=int((time.time() - started) * 1000),
                attempts=attempts, error_type="fatal",
            )
        except ToolRetryableError as e:
            last_error = e
            logger.warning(f"工具 {tool_name} 可重试错误（第 {attempts} 次）: {e}")
        except Exception as e:  # 未知异常：按可重试处理一次
            last_error = e
            logger.warning(f"工具 {tool_name} 未知异常（第 {attempts} 次）: {e}")

    return ToolExecutionResult(
        ok=False, tool_name=tool_name,
        content=f"工具执行失败（已重试 {MAX_TOOL_RETRIES} 次）: {last_error}。建议更换工具或参数。",
        latency_ms=int((time.time() - started) * 1000),
        attempts=attempts, error_type="retry_exhausted",
    )
