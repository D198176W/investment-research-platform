"""MCP Server —— 将 Tool 注册中心中的工具通过 MCP 协议（stdio）暴露

用法：
    python -m agent_service.tools.mcp_server

MCP Python SDK 2.x API：Server 构造函数注册 on_list_tools / on_call_tool handler。
客户端（Claude Desktop / 本项目 mcp_client）通过 stdio 连接，
调用 initialize → tools/list → tools/call 完成动态工具发现与执行。
"""
import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_mcp_server():
    """创建 MCP Server 实例，将所有注册工具暴露为 MCP Tools"""
    from mcp.server import Server
    import mcp.types as types

    from .registry import registry
    from . import market_tools  # noqa: F401  触发内置工具注册

    async def on_list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[
            types.Tool(
                name=t.name,
                description=t.description,
                inputSchema=t.parameters,
            )
            for t in registry.list_tools()
        ])

    async def on_call_tool(ctx: Any, params: Any) -> types.CallToolResult:
        from .executor import execute_tool
        # 在线程中执行同步 handler，避免阻塞事件循环
        result = await asyncio.to_thread(execute_tool, params.name, params.arguments or {})
        return types.CallToolResult(
            content=[types.TextContent(
                type="text",
                text=json.dumps(result.to_dict(), ensure_ascii=False),
            )],
            isError=not result.ok,
        )

    server = Server(
        "investment-research-tools",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )
    return server


async def _amain():
    from mcp.server.stdio import stdio_server
    server = create_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
