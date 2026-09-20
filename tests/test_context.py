"""上下文窗口管理单测：token 计数、协议安全裁剪、优先级组装"""
from agent_service.memory.context import (
    ContextSection, ContextWindow, count_tokens, count_messages_tokens,
    trim_messages_to_budget,
    PRIORITY_SYSTEM, PRIORITY_HISTORY,
)


def test_count_tokens_basic():
    assert count_tokens("") == 0
    assert count_tokens("hello") > 0
    assert count_tokens("中文文本") > 0
    assert count_tokens("abcd" * 10) > count_tokens("abcd")


def test_count_messages_tokens_includes_overhead():
    msgs = [{"role": "user", "content": "你好"}]
    assert count_messages_tokens(msgs) > count_tokens("你好")


def test_trim_under_budget_unchanged():
    msgs = [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "任务"},
        {"role": "assistant", "content": "回复"},
    ]
    trimmed = trim_messages_to_budget(msgs, max_tokens=10000)
    assert trimmed == msgs


def test_trim_keeps_system_and_first_user():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    # 填充大量历史消息使其超预算
    for i in range(200):
        msgs.append({"role": "assistant", "content": "长回复" * 200})
        msgs.append({"role": "user", "content": "追问" * 200})
    trimmed = trim_messages_to_budget(msgs, max_tokens=3000, reserve_for_output=500)
    assert trimmed[0]["role"] == "system"
    assert trimmed[1]["content"] == "U"
    assert len(trimmed) < len(msgs)
    assert count_messages_tokens(trimmed) <= 3000


def test_trim_no_orphan_tool_message():
    """裁剪边界绝不能落在 tool 消息上（OpenAI 协议要求 tool 响应必须跟在 tool_calls 后）"""
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    for i in range(100):
        msgs.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "t", "arguments": "{}"}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "结果" * 500})
    trimmed = trim_messages_to_budget(msgs, max_tokens=3000, reserve_for_output=500)
    assert trimmed[0]["role"] == "system"
    # 裁剪后第一条非头部消息不能是孤立 tool 消息
    assert trimmed[2]["role"] != "tool" or len(trimmed) == 2


def test_trim_long_tool_content_truncated():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "tool", "tool_call_id": "c1", "content": "数据" * 5000},
    ]
    trimmed = trim_messages_to_budget(msgs, max_tokens=4000, reserve_for_output=500)
    tool_msg = trimmed[-1]
    assert "已按 token 预算截断" in tool_msg["content"] or count_tokens(tool_msg["content"]) <= 1600


def test_context_window_priority_drop():
    """超预算时低优先级段（历史对话）先被丢弃"""
    window = ContextWindow(max_tokens=1000, reserve_for_output=100)
    sections = [
        ContextSection(role="system", content="系统提示", priority=PRIORITY_SYSTEM, name="sys"),
        ContextSection(role="user", content="历史" * 2000, priority=PRIORITY_HISTORY, name="hist"),
        ContextSection(role="user", content="当前任务", priority=10, name="task"),
    ]
    assembled = window.assemble(sections)
    contents = [m["content"] for m in assembled]
    assert "系统提示" in contents
    assert "当前任务" in contents
    assert not any("历史" in c for c in contents), "低优先级段应被丢弃"


def test_context_window_section_cap():
    window = ContextWindow(max_tokens=100000, reserve_for_output=100)
    sections = [
        ContextSection(role="user", content="证据" * 5000, priority=30, name="evi",
                       max_tokens=100),
    ]
    assembled = window.assemble(sections)
    assert count_tokens(assembled[0]["content"]) <= 110
    assert "已按 token 预算截断" in assembled[0]["content"]
