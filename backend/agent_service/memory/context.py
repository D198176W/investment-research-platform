"""上下文窗口管理 —— Token 计数、优先级截断、摘要压缩

对齐 JD「上下文管理、记忆窗口截断、Token 优化」：
- 用 tiktoken 精确计数（而非按字符数估算）
- 超预算时按优先级截断：系统提示 > 当前任务 > 检索证据 > 历史对话
- 证据类内容超长时按条截断并保留头部（重要信息置顶）
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# tiktoken 对中文的近似：cl100k_base 对中文约 1 token/字（偏保守）
_ENCODING = None


def _get_encoding():
    global _ENCODING
    if _ENCODING is None:
        try:
            import tiktoken
            _ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"tiktoken 不可用，使用字符近似计数: {e}")
            _ENCODING = False
    return _ENCODING


def count_tokens(text: str) -> int:
    """精确 token 计数；tiktoken 不可用时按字符近似（中文≈1 token/字）"""
    enc = _get_encoding()
    if enc:
        try:
            return len(enc.encode(text or ""))
        except Exception:
            pass
    return len(text or "")


def count_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """估算一组消息的总 token（含每条消息的角色开销）"""
    total = 0
    for m in messages:
        total += 4  # role 等元数据开销
        total += count_tokens(str(m.get("content", "")))
        if m.get("tool_calls"):
            total += count_tokens(str(m["tool_calls"]))
    return total


@dataclass
class ContextSection:
    """一段上下文，带优先级（数值越小优先级越高，截断时后删）"""
    role: str                    # system / user / tool / assistant
    content: str
    priority: int = 50           # 0=最高（必保留），越大越优先被截断
    name: str = ""               # 段名（用于日志）
    max_tokens: Optional[int] = None  # 该段自身的 token 上限


# 优先级约定
PRIORITY_SYSTEM = 0        # 系统提示（必保留）
PRIORITY_TASK = 10         # 当前任务描述（必保留）
PRIORITY_EVIDENCE = 30     # 检索证据/工具数据（可截断）
PRIORITY_MEMORY = 40       # 长期记忆注入（可截断）
PRIORITY_HISTORY = 60      # 历史对话（最先被截断）


class ContextWindow:
    """上下文窗口管理器：按 token 预算组装消息列表"""

    def __init__(self, max_tokens: int = 30000, reserve_for_output: int = 2000):
        self.max_tokens = max_tokens
        self.reserve_for_output = reserve_for_output

    @property
    def budget(self) -> int:
        return self.max_tokens - self.reserve_for_output

    def assemble(self, sections: List[ContextSection]) -> List[Dict[str, str]]:
        """按优先级与预算组装消息列表

        策略：
        1. 高优先级段（priority 小）先放入
        2. 低优先级段按 max_tokens 截断后放入
        3. 总预算超限时，从最低优先级开始整体丢弃
        """
        # 按优先级排序（稳定排序保持同级相对顺序）
        ordered = sorted(enumerate(sections), key=lambda x: (x[1].priority, x[0]))

        placed: List[tuple[int, ContextSection]] = []
        used = 0
        for orig_idx, sec in ordered:
            content = sec.content
            tokens = count_tokens(content)

            # 段内截断
            if sec.max_tokens is not None and tokens > sec.max_tokens:
                content = self._truncate_head(content, sec.max_tokens)
                tokens = count_tokens(content)

            # 预算检查
            if used + tokens > self.budget:
                if sec.priority <= PRIORITY_TASK:
                    # 高优段超预算：强制截断而非丢弃
                    content = self._truncate_head(content, max(100, self.budget - used))
                    tokens = count_tokens(content)
                    logger.warning(f"高优先级上下文段 {sec.name} 超预算，强制截断至 {tokens} tokens")
                else:
                    logger.info(f"上下文段 {sec.name}(priority={sec.priority}) 因预算不足被丢弃 "
                                f"(需 {tokens} tokens, 剩余 {self.budget - used})")
                    continue

            placed.append((orig_idx, ContextSection(**{**sec.__dict__, "content": content})))
            used += tokens

        # 恢复原始顺序输出
        placed.sort(key=lambda x: x[0])
        total = sum(count_tokens(s.content) for _, s in placed)
        logger.info(f"上下文组装完成: {len(placed)}/{len(sections)} 段, 共 {total} tokens (预算 {self.budget})")
        return [{"role": s.role, "content": s.content} for _, s in placed]

    @staticmethod
    def _truncate_head(text: str, max_tokens: int) -> str:
        """保留头部截断（重要信息置顶）"""
        if count_tokens(text) <= max_tokens:
            return text
        # 二分查找最大可保留字符数
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count_tokens(text[:mid]) <= max_tokens - 10:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo] + "\n...(已按 token 预算截断)"

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """对外暴露的文本截断工具"""
        return self._truncate_head(text, max_tokens)


def trim_messages_to_budget(messages: List[Dict[str, Any]],
                            max_tokens: int = 30000,
                            reserve_for_output: int = 2000) -> List[Dict[str, Any]]:
    """协议安全的消息列表裁剪（OpenAI tool 协议感知）

    用于工具调用循环中防止上下文超窗：
    1. 始终保留 system 消息与第一条 user 消息（任务描述）
    2. 从最新向前保留后续消息，直到预算耗尽
    3. 裁剪边界绝不落在 tool 消息上（避免孤立的 tool 响应导致 API 报错）
    4. 单条超长 tool 结果内容按 token 截断（重要信息置顶）

    Args:
        messages: OpenAI 格式消息列表
        max_tokens: 总预算（含输出预留）
        reserve_for_output: 为模型输出预留的 token

    Returns:
        裁剪后的消息列表（未超预算时原样返回）
    """
    budget = max_tokens - reserve_for_output
    if count_messages_tokens(messages) <= budget:
        return messages

    # 头部：所有 system + 第一条 user（任务描述，必保留）
    head: List[Dict[str, Any]] = []
    rest_start = 0
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            head.append(m)
            rest_start = i + 1
        elif m.get("role") == "user" and not any(x.get("role") == "user" for x in head):
            head.append(m)
            rest_start = i + 1
        else:
            break
    rest = messages[rest_start:]

    # 单条超长 tool 结果先截断（保留头部重要信息）
    window = ContextWindow(max_tokens=max_tokens, reserve_for_output=reserve_for_output)
    per_tool_cap = 1500
    processed_rest: List[Dict[str, Any]] = []
    for m in rest:
        if m.get("role") == "tool" and count_tokens(str(m.get("content", ""))) > per_tool_cap:
            m = {**m, "content": window.truncate_text(str(m.get("content", "")), per_tool_cap)}
        processed_rest.append(m)
    rest = processed_rest

    # 从最新向前累计，直到预算耗尽
    remaining = budget - count_messages_tokens(head)
    kept: List[Dict[str, Any]] = []
    used = 0
    for m in reversed(rest):
        t = 4 + count_tokens(str(m.get("content", "")))
        if m.get("tool_calls"):
            t += count_tokens(str(m["tool_calls"]))
        if used + t > remaining:
            break
        kept.insert(0, m)
        used += t

    # 协议安全：裁剪后的首条不能是 tool 消息（其对应的 assistant tool_calls 已被裁掉）
    while kept and kept[0].get("role") == "tool":
        kept.pop(0)

    if len(kept) < len(rest):
        logger.info(
            f"上下文窗口裁剪: 历史消息 {len(rest)}→{len(kept)} 条 "
            f"(预算 {budget} tokens，头部保留 {len(head)} 条)"
        )
    return head + kept
