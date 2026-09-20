"""多模型路由单测：降级链切换、配额耗尽容错"""
import pytest

from agent_service.llm_client import (
    ChatResult, LLMFatalError, LLMRetryableError, ModelRoute, ModelRouter,
)


class _FakeClient:
    """模拟 LLMClient：按预设行为抛错或返回"""

    def __init__(self, behavior):
        self.behavior = behavior  # "ok" / "fatal" / "retryable"

    def chat(self, messages, **kwargs):
        if self.behavior == "fatal":
            raise LLMFatalError("模型配额耗尽(403)")
        if self.behavior == "retryable":
            raise LLMRetryableError("网络超时")
        return ChatResult(content="ok", model="fake", prompt_tokens=10,
                          completion_tokens=5, latency_ms=100)


def _router(behaviors):
    router = ModelRouter("key", "url", routes={
        "heavy": ModelRoute("heavy", ["m1", "m2", "m3"]),
    })
    for model, behavior in behaviors.items():
        router._clients[model] = _FakeClient(behavior)
    # 让 _client_for 返回预设的假 client
    router._client_for = lambda model: router._clients[model]
    return router


def test_primary_model_used_first():
    router = _router({"m1": "ok"})
    result = router.chat([{"role": "user", "content": "hi"}])
    assert result.content == "ok"
    assert router.last_used_model == "m1"


def test_fallback_on_fatal_quota_error():
    """主模型 403 配额耗尽时应沿降级链切换，而非直接失败"""
    router = _router({"m1": "fatal", "m2": "ok"})
    result = router.chat([{"role": "user", "content": "hi"}])
    assert result.content == "ok"
    assert router.last_used_model == "m2"


def test_fallback_on_retryable_error():
    router = _router({"m1": "retryable", "m2": "ok"})
    result = router.chat([{"role": "user", "content": "hi"}])
    assert router.last_used_model == "m2"


def test_all_models_failed_raises():
    router = _router({"m1": "fatal", "m2": "retryable", "m3": "fatal"})
    with pytest.raises(LLMRetryableError):
        router.chat([{"role": "user", "content": "hi"}])
