"""MCP Client —— 动态发现与调用 MCP Server 上的工具

设计要点（对齐 JD「MCP 工具与 Skills 体系」）：
- 启动时通过 stdio 拉起 mcp_server 子进程，initialize 握手
- tools/list 动态拉取工具清单（新增工具无需改 Agent 代码）
- tools/call 远程调用，结果转为 ToolExecutionResult
- 不可用时自动降级为进程内注册中心直接执行（保证健壮性）
"""
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from .executor import ToolExecutionResult

logger = logging.getLogger(__name__)

_SERVER_MODULE = "agent_service.tools.mcp_server"


class MCPClient:
    """MCP stdio 客户端（同步包装，供同步工作流调用）"""

    def __init__(self):
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._available: Optional[bool] = None

    # ---------- 内部：异步原语 ----------

    async def _with_session(self, fn):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env = dict(os.environ)
        env["PYTHONPATH"] = backend_dir + os.pathsep + env.get("PYTHONPATH", "")

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", _SERVER_MODULE],
            env=env,
            cwd=backend_dir,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await fn(session)

    @staticmethod
    def _run(coro):
        """在独立线程中跑事件循环，避免与调用方已有 loop 冲突"""
        import threading
        box: Dict[str, Any] = {}

        def _runner():
            try:
                box["result"] = asyncio.run(coro)
            except Exception as e:
                box["error"] = e

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join(timeout=60)
        if t.is_alive():
            raise TimeoutError("MCP 调用超时(60s)")
        if "error" in box:
            raise box["error"]
        return box.get("result")

    # ---------- 对外 API ----------

    def list_tools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """动态拉取工具清单（带缓存）"""
        if self._tools_cache is not None and not force_refresh:
            return self._tools_cache

        async def _list(session):
            resp = await session.list_tools()
            return [
                {"name": t.name, "description": t.description or "",
                 "inputSchema": getattr(t, "inputSchema", None) or getattr(t, "input_schema", {}) or {}}
                for t in resp.tools
            ]

        self._tools_cache = self._run(_with_session_safe(self._with_session, _list))
        return self._tools_cache

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
        """通过 MCP 调用工具"""
        async def _call(session):
            resp = await session.call_tool(name, arguments or {})
            # 服务端返回 TextContent(JSON 字符串)
            for content in resp.content:
                if getattr(content, "type", None) == "text":
                    payload = json.loads(content.text)
                    return ToolExecutionResult(
                        ok=payload.get("ok", False),
                        tool_name=payload.get("tool_name", name),
                        content=payload.get("content", ""),
                        latency_ms=payload.get("latency_ms", 0),
                        attempts=payload.get("attempts", 1),
                        error_type=payload.get("error_type"),
                        truncated=payload.get("truncated", False),
                    )
            return ToolExecutionResult(ok=False, tool_name=name, content="MCP 返回为空",
                                       latency_ms=0, attempts=1, error_type="empty_response")

        return self._run(_with_session_safe(self._with_session, _call))

    def ping(self) -> bool:
        """探活：能否完成 initialize + tools/list"""
        try:
            tools = self.list_tools(force_refresh=True)
            self._available = len(tools) > 0
        except Exception as e:
            logger.warning(f"MCP Server 不可用，降级为进程内执行: {e}")
            self._available = False
        return self._available

    @property
    def available(self) -> bool:
        if self._available is None:
            return self.ping()
        return self._available


def _with_session_safe(with_session_fn, inner_fn):
    """把 _with_session(session_fn) 包装为一个协程（便于 _run 调用）"""
    async def _wrapper():
        return await with_session_fn(inner_fn)
    return _wrapper()


# 全局客户端单例
mcp_client = MCPClient()
