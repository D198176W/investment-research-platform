"""工具执行器单测：异常分类、重试、结果截断、护栏"""
import time

from agent_service.tools import executor
from agent_service.tools.executor import execute_tool
from agent_service.tools.market_tools import ToolFatalError, ToolRetryableError
from agent_service.tools.registry import Tool


def _tool(name, handler, timeout=5, max_result_chars=4000):
    return Tool(
        name=name, description="测试工具",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
        handler=handler, timeout=timeout, max_result_chars=max_result_chars,
    )


def test_execute_success():
    tool = _tool("t_ok", lambda x: {"echo": x})
    result = execute_tool("t_ok", {"x": "hello"}, tool=tool)
    assert result.ok
    assert "hello" in result.content
    assert result.attempts == 1


def test_tool_not_found():
    result = execute_tool("nonexistent_tool_xyz", {})
    assert not result.ok
    assert result.error_type == "tool_not_found"


def test_invalid_arguments_blocked_by_guard():
    called = []

    def handler(x):
        called.append(x)
        return x

    tool = _tool("t_guard", handler)
    result = execute_tool("t_guard", {}, tool=tool)  # 缺必填参数
    assert not result.ok
    assert result.error_type == "invalid_arguments"
    assert called == [], "参数非法时 handler 不应被执行（护栏）"


def test_fatal_error_no_retry():
    calls = []

    def handler(x):
        calls.append(1)
        raise ToolFatalError("数据不存在")

    tool = _tool("t_fatal", handler)
    result = execute_tool("t_fatal", {"x": "1"}, tool=tool)
    assert not result.ok
    assert result.error_type == "fatal"
    assert len(calls) == 1, "Fatal 错误不应重试"


def test_retryable_error_exhausted(monkeypatch):
    monkeypatch.setattr(executor, "MAX_TOOL_RETRIES", 1)
    monkeypatch.setattr(time, "sleep", lambda s: None)  # 跳过退避等待
    calls = []

    def handler(x):
        calls.append(1)
        raise ToolRetryableError("网络超时")

    tool = _tool("t_retry", handler)
    result = execute_tool("t_retry", {"x": "1"}, tool=tool)
    assert not result.ok
    assert result.error_type == "retry_exhausted"
    assert len(calls) == 2, "可重试错误应重试 MAX_TOOL_RETRIES 次"


def test_retryable_then_success(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = []

    def handler(x):
        calls.append(1)
        if len(calls) < 2:
            raise ToolRetryableError("偶发超时")
        return {"ok": True}

    tool = _tool("t_flaky", handler)
    result = execute_tool("t_flaky", {"x": "1"}, tool=tool)
    assert result.ok
    assert result.attempts == 2


def test_timeout_is_retryable(monkeypatch):
    monkeypatch.setattr(executor, "MAX_TOOL_RETRIES", 0)

    def handler(x):
        time.sleep(2)

    tool = _tool("t_slow", handler, timeout=1)
    result = execute_tool("t_slow", {"x": "1"}, tool=tool)
    assert not result.ok
    assert result.error_type == "retry_exhausted"
    assert "超时" in result.content


def test_result_truncation():
    big = [{"i": i, "text": "长文本" * 100} for i in range(100)]
    tool = _tool("t_big", lambda x: big, max_result_chars=500)
    result = execute_tool("t_big", {"x": "1"}, tool=tool)
    assert result.ok
    assert result.truncated
    assert len(result.content) <= 600
