"""Agentic 工具调用循环 —— 模型自主规划 → Tool 选择 → 执行 → 结果理解 → 继续/结束

对齐 JD「任务理解与拆解、规划与推理、Tool 选择与执行、失败恢复、Agentic Search」：
- 模型基于工具描述（JSON Schema）自主决定是否/调用哪个工具
- 每轮调用记录进 Scratchpad（工作记忆），供最终结构化输出引用
- 工具失败将错误回喂模型，由模型换参/换工具恢复
- 达到 max_rounds 或模型不再发起 tool_calls 时收敛
"""
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .llm_client import LLMClient, LLMFatalError
from .memory.context import trim_messages_to_budget
from .tools.registry import registry
from .tools.executor import execute_tool, ToolExecutionResult

logger = logging.getLogger(__name__)


class ToolCallRecord:
    """一次工具调用的完整记录（进入 trace 与 scratchpad）"""

    def __init__(self, tool_name: str, arguments: Dict[str, Any], result: ToolExecutionResult):
        self.tool_name = tool_name
        self.arguments = arguments
        self.ok = result.ok
        self.content = result.content
        self.latency_ms = result.latency_ms
        self.attempts = result.attempts
        self.truncated = result.truncated
        self.error_type = result.error_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "arguments": self.arguments,
            "ok": self.ok,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "truncated": self.truncated,
            "error_type": self.error_type,
            "result_preview": self.content[:500],
        }


class AgentLoopResult:
    """工具调用循环的最终产出"""

    def __init__(self):
        self.records: List[ToolCallRecord] = []   # 所有工具调用记录
        self.final_text: str = ""                  # 模型最终文本输出
        self.rounds: int = 0
        self.total_tokens: int = 0
        self.total_prompt_tokens: int = 0          # API 返回的真实 prompt tokens 累计
        self.total_completion_tokens: int = 0      # API 返回的真实 completion tokens 累计
        self.total_latency_ms: int = 0
        self.stopped_reason: str = ""              # no_more_tool_calls / max_rounds / error

    def succeeded_tools(self) -> List[str]:
        return [r.tool_name for r in self.records if r.ok]

    def evidence(self) -> str:
        """汇总所有成功工具的返回，作为下游阶段的证据上下文"""
        parts = []
        for r in self.records:
            if r.ok:
                parts.append(f"【工具 {r.tool_name}({json.dumps(r.arguments, ensure_ascii=False)})】\n{r.content}")
        return "\n\n".join(parts) if parts else "（未获取到工具数据）"


def run_tool_calling_loop(
    llm: LLMClient,
    system_prompt: str,
    user_prompt: str,
    max_rounds: int = 8,
    temperature: float = 0.3,
    on_tool_call: Optional[Callable[[ToolCallRecord], None]] = None,
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
    executor: Optional[Callable[[str, Dict[str, Any]], ToolExecutionResult]] = None,
) -> AgentLoopResult:
    """运行 ReAct 风格的工具调用循环

    Args:
        llm: LLM 客户端
        system_prompt: 系统提示（含任务背景与 Skill）
        user_prompt: 用户任务
        max_rounds: 最大工具调用轮数（安全阀）
        on_tool_call: 每次工具调用后的回调（用于进度展示/Trace）
        tool_schemas: 可选，覆盖默认工具清单（默认用注册中心全部工具）
        executor: 可选，覆盖默认执行器（默认进程内执行；MCP 模式传 mcp_client.call_tool）

    Returns:
        AgentLoopResult
    """
    tools = tool_schemas or registry.to_openai_tools()
    exec_fn = executor or execute_tool

    # 上下文窗口预算（记忆窗口截断 / Token 优化）
    try:
        from .config import get_agent_settings
        _settings = get_agent_settings()
        _max_tokens = _settings.CONTEXT_MAX_TOKENS
        _reserve = _settings.CONTEXT_RESERVE_OUTPUT
    except Exception:
        _max_tokens, _reserve = 30000, 2000

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    result = AgentLoopResult()

    for round_idx in range(1, max_rounds + 1):
        result.rounds = round_idx
        logger.info(f"工具调用循环 第 {round_idx}/{max_rounds} 轮")

        # 记忆窗口截断：超预算时按"系统提示 > 当前任务 > 近期证据"优先级裁剪
        messages = trim_messages_to_budget(messages, max_tokens=_max_tokens,
                                           reserve_for_output=_reserve)

        try:
            chat = llm.chat(messages, tools=tools, temperature=temperature)
        except LLMFatalError:
            raise
        except Exception as e:
            result.stopped_reason = f"error: {e}"
            logger.error(f"工具调用循环 LLM 失败: {e}")
            break

        result.total_tokens += chat.total_tokens
        result.total_prompt_tokens += chat.prompt_tokens
        result.total_completion_tokens += chat.completion_tokens
        result.total_latency_ms += chat.latency_ms

        # 收敛条件：模型不再调用工具
        if not chat.tool_calls:
            result.final_text = chat.content
            result.stopped_reason = "no_more_tool_calls"
            # 把最终回答也记录进消息（便于调用方续接）
            messages.append({"role": "assistant", "content": chat.content})
            break

        # 记录 assistant 的 tool_calls 消息（OpenAI 协议要求原样回传）
        messages.append({
            "role": "assistant",
            "content": chat.content or "",
            "tool_calls": chat.tool_calls,
        })

        # 逐个执行工具调用
        for tc in chat.tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            try:
                arguments = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}

            exec_result = exec_fn(tool_name, arguments)
            record = ToolCallRecord(tool_name, arguments, exec_result)
            result.records.append(record)

            if on_tool_call:
                try:
                    on_tool_call(record)
                except Exception:
                    pass

            # 回喂工具结果（OpenAI tool 消息）
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": exec_result.content,
            })

    else:
        result.stopped_reason = "max_rounds"
        # 最后一轮强制模型总结（不再给工具）
        messages.append({
            "role": "user",
            "content": "已达到最大工具调用轮数，请基于已获取的数据直接给出分析结论。",
        })
        try:
            messages = trim_messages_to_budget(messages, max_tokens=_max_tokens,
                                               reserve_for_output=_reserve)
            chat = llm.chat(messages, temperature=temperature)
            result.final_text = chat.content
            result.total_tokens += chat.total_tokens
            result.total_prompt_tokens += chat.prompt_tokens
            result.total_completion_tokens += chat.completion_tokens
            result.total_latency_ms += chat.latency_ms
        except Exception as e:
            logger.error(f"工具调用循环收尾总结失败: {e}")

    logger.info(
        f"工具调用循环结束: {result.rounds} 轮, {len(result.records)} 次工具调用, "
        f"成功工具 {result.succeeded_tools()}, 原因={result.stopped_reason}"
    )
    return result
