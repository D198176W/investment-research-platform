"""Tool 注册中心与参数校验单测"""
from agent_service.tools.registry import (
    Tool, ToolRegistry, validate_arguments,
)


def _make_tool():
    return Tool(
        name="demo_tool",
        description="演示工具",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码"},
                "limit": {"type": "integer", "description": "条数"},
            },
            "required": ["symbol"],
        },
        handler=lambda symbol, limit=10: {"symbol": symbol, "limit": limit},
    )


def test_register_and_get():
    reg = ToolRegistry()
    tool = _make_tool()
    reg.register(tool)
    assert reg.get("demo_tool") is tool
    assert "demo_tool" in reg.tool_names()
    assert len(reg.list_tools()) == 1


def test_to_openai_schema():
    schema = _make_tool().to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "demo_tool"
    assert "parameters" in schema["function"]


def test_to_mcp_schema():
    schema = _make_tool().to_mcp_schema()
    assert schema["name"] == "demo_tool"
    assert "inputSchema" in schema


def test_validate_arguments_ok():
    errors = validate_arguments(_make_tool(), {"symbol": "600519", "limit": 5})
    assert errors == []


def test_validate_arguments_missing_required():
    errors = validate_arguments(_make_tool(), {"limit": 5})
    assert any("symbol" in e for e in errors)


def test_validate_arguments_wrong_type():
    errors = validate_arguments(_make_tool(), {"symbol": "600519", "limit": "五"})
    assert any("limit" in e and "类型错误" in e for e in errors)


def test_validate_arguments_bool_is_not_integer():
    """bool 是 int 子类，但不应通过 integer 校验"""
    errors = validate_arguments(_make_tool(), {"symbol": "600519", "limit": True})
    assert any("limit" in e for e in errors)
