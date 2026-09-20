"""Tool 注册中心 —— 以 JSON Schema 描述工具，支持模型动态选择与 MCP 暴露

设计要点：
- 每个 Tool 含 name / description / parameters(JSON Schema) / handler
- 新增工具只需 @register_tool 装饰一个函数，Agent 与 MCP Server 自动可见
- 只读白名单：所有数据工具标记 readonly=True，构成安全护栏
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """工具定义（对齐 OpenAI Function Calling / MCP Tool 规范）"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema object
    handler: Callable[..., Any]
    readonly: bool = True       # 工具白名单护栏：数据工具均为只读
    timeout: int = 15           # 单次执行超时（秒）
    max_result_chars: int = 4000  # 结果理解：注入上下文前的最大字符数

    def to_openai_schema(self) -> Dict[str, Any]:
        """转为 OpenAI Function Calling 的 tools 参数格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_mcp_schema(self) -> Dict[str, Any]:
        """转为 MCP tools/list 响应格式"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
        }


class ToolRegistry:
    """工具注册中心（进程内单例）"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        if tool.name in self._tools:
            logger.warning(f"工具重复注册，覆盖: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug(f"工具已注册: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]


# 全局注册中心
registry = ToolRegistry()


def register_tool(
    name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None,
    readonly: bool = True,
    timeout: int = 15,
    max_result_chars: int = 4000,
):
    """工具注册装饰器

    用法:
        @register_tool(
            name="get_stock_quote",
            description="获取A股个股实时行情",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "6位股票代码，如 600519"}
                },
                "required": ["symbol"],
            },
        )
        def get_stock_quote(symbol: str):
            ...
    """
    if parameters is None:
        parameters = {"type": "object", "properties": {}}

    def decorator(func: Callable[..., Any]):
        registry.register(Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=func,
            readonly=readonly,
            timeout=timeout,
            max_result_chars=max_result_chars,
        ))
        return func

    return decorator


def validate_arguments(tool: Tool, arguments: Dict[str, Any]) -> List[str]:
    """轻量 JSON Schema 校验（required + type），返回错误信息列表（空列表=通过）"""
    errors = []
    schema = tool.parameters or {}
    props = schema.get("properties", {})

    for req in schema.get("required", []):
        if req not in arguments or arguments[req] is None:
            errors.append(f"缺少必填参数: {req}")

    type_map = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
    for key, value in arguments.items():
        if key in props:
            expected = props[key].get("type")
            py_type = type_map.get(expected)
            if py_type:
                # JSON Schema 语义：bool 不是 integer/number（Python 中 bool 是 int 子类，需特判）
                if isinstance(value, bool) and expected in ("integer", "number"):
                    errors.append(f"参数 {key} 类型错误: 期望 {expected}, 实际 bool")
                elif not isinstance(value, py_type):
                    errors.append(f"参数 {key} 类型错误: 期望 {expected}, 实际 {type(value).__name__}")

    return errors
