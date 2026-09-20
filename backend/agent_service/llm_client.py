"""LLM 客户端 —— OpenAI 兼容协议统一接入，支持 Function Calling 与多模型路由/降级

设计要点（对齐 JD「多模型路由与降级」「Function Calling」）：
- 统一走 OpenAI 兼容协议（DashScope compatible-mode / DeepSeek / 本地 vLLM 均兼容）
- chat_with_tools: 原生 tools 参数，模型返回结构化 tool_calls
- ModelRouter: 按任务级别选择模型（heavy 推理 / light 摘要），失败自动降级
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ChatResult:
    """一次对话的结果"""
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """OpenAI 兼容 LLM 客户端"""

    def __init__(self, api_key: str, base_url: str, model: str,
                 timeout: int = 120, max_retries: int = 2):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=self.timeout, max_retries=0)  # 重试由本层控制
        return self._client

    def chat(self, messages: List[Dict[str, Any]],
             tools: Optional[List[Dict[str, Any]]] = None,
             temperature: float = 0.3,
             response_format: Optional[Dict[str, Any]] = None,
             tool_choice: Optional[Any] = None) -> ChatResult:
        """发送对话请求，支持 tools / response_format

        Raises:
            LLMRetryableError: 网络/限流/超时，可重试
            LLMFatalError: 认证失败等，不重试
        """
        client = self._get_client()
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            started = time.time()
            try:
                resp = client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                msg = choice.message
                tool_calls = []
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        })
                usage = resp.usage
                return ChatResult(
                    content=msg.content or "",
                    tool_calls=tool_calls,
                    model=self.model,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    latency_ms=int((time.time() - started) * 1000),
                )
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                # 认证/参数错误不重试
                if any(k in err_str for k in ("api key", "unauthorized", "401", "invalid_api_key", "incorrect api key")):
                    raise LLMFatalError(f"LLM 认证失败: {e}") from e
                # 配额耗尽（403）不重试——重试不会恢复，快速失败并给出明确指引
                if "403" in err_str or "quota" in err_str or "free quota" in err_str:
                    raise LLMFatalError(
                        f"模型 {self.model} 配额耗尽或无权限(403)，请在模型下拉框更换可用模型: {e}"
                    ) from e
                logger.warning(f"LLM 调用失败（第 {attempt + 1} 次）: {e}")
                if attempt < self.max_retries:
                    time.sleep(1.5 * (2 ** attempt))

        raise LLMRetryableError(f"LLM 调用重试 {self.max_retries} 次后仍失败: {last_err}") from last_err


class LLMRetryableError(Exception):
    """可重试 LLM 错误"""


class LLMFatalError(Exception):
    """不可重试 LLM 错误（认证、参数）"""


# ==================== 多模型路由 ====================

@dataclass
class ModelRoute:
    """一条模型路由：主模型 + 降级链"""
    name: str                     # 路由名，如 "heavy" / "light"
    models: List[str]             # 按优先级排序的模型列表


class ModelRouter:
    """多模型路由器：按任务级别选模型，失败沿降级链切换

    用法:
        router = ModelRouter(api_key, api_url, routes={
            "heavy": ModelRoute("heavy", ["qwen-max", "qwen-plus"]),
            "light": ModelRoute("light", ["qwen-turbo", "qwen-plus"]),
        })
        result = router.chat(messages, route="heavy")
    """

    def __init__(self, api_key: str, base_url: str,
                 routes: Optional[Dict[str, ModelRoute]] = None,
                 default_route: str = "heavy", timeout: int = 120):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.default_route = default_route
        self.routes = routes or {
            "heavy": ModelRoute("heavy", ["qwen-max", "qwen-plus"]),
            "light": ModelRoute("light", ["qwen-turbo", "qwen-plus"]),
        }
        self._clients: Dict[str, LLMClient] = {}
        # 记录实际生效的模型（降级链切换后可观测）
        self.last_used_model: str = ""

    def _client_for(self, model: str) -> LLMClient:
        if model not in self._clients:
            self._clients[model] = LLMClient(self.api_key, self.base_url, model,
                                             timeout=self.timeout)
        return self._clients[model]

    def chat(self, messages: List[Dict[str, Any]], route: Optional[str] = None,
             **kwargs) -> ChatResult:
        """沿降级链调用：当前模型失败（含 403 配额耗尽）则切换下一个

        与 LLMClient 直接调用的区别：LLMFatalError（配额/权限）不再立即抛出，
        而是沿降级链继续尝试——单模型配额耗尽不应导致整个任务失败。
        """
        route_name = route or self.default_route
        r = self.routes.get(route_name) or self.routes.get("heavy")
        assert r is not None, "ModelRouter 至少需要一个路由"

        last_err: Optional[Exception] = None
        for model in r.models:
            try:
                result = self._client_for(model).chat(messages, **kwargs)
                self.last_used_model = model
                if model != r.models[0]:
                    logger.info(f"路由 {route_name} 降级到模型 {model}")
                return result
            except Exception as e:
                logger.warning(f"模型 {model} 失败（路由 {route_name}），尝试降级链下一个: {e}")
                last_err = e
        raise LLMRetryableError(
            f"路由 {route_name} 全部模型失败（{r.models}）: {last_err}"
        ) from last_err
