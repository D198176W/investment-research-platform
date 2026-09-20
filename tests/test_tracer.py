"""Trace 与工具指标单测：挂起缓冲、步骤归属、指标聚合"""
import os

from agent_service.tracer import ToolMetrics, Tracer


def test_pending_buffer_flushes_into_step(tmp_path):
    """图驱动执行时，节点内的记录发生在 start_step 之前，应通过挂起缓冲正确归属"""
    tracer = Tracer("t1", trace_dir=str(tmp_path))
    # 模拟节点执行期间的记录（此时无当前步骤）
    tracer.record_llm(purpose="reasoning", model="qwen-plus",
                      prompt_tokens=100, completion_tokens=50, latency_ms=200)
    tracer.record_tool({"tool": "get_major_indices", "ok": True})
    # 节点完成后才建立步骤
    tracer.start_step("reasoning")
    tracer.end_step(status="ok")

    assert len(tracer.steps) == 1
    step = tracer.steps[0]
    assert len(step.llm_calls) == 1
    assert step.llm_calls[0].total_tokens == 150
    assert len(step.tool_calls) == 1


def test_save_and_load_roundtrip(tmp_path):
    tracer = Tracer("t2", trace_dir=str(tmp_path))
    tracer.start_step("perception")
    tracer.record_llm(purpose="perception", model="m", prompt_tokens=10,
                      completion_tokens=5, latency_ms=100)
    tracer.end_step(status="ok")
    path = tracer.save()
    assert os.path.exists(path)

    loaded = Tracer.load("t2", trace_dir=str(tmp_path))
    assert loaded["summary"]["total_llm_tokens"] == 15
    assert loaded["steps"][0]["step"] == "perception"


def test_summary_tool_success_rate(tmp_path):
    tracer = Tracer("t3", trace_dir=str(tmp_path))
    tracer.start_step("p")
    tracer.record_tool({"tool": "a", "ok": True})
    tracer.record_tool({"tool": "b", "ok": False})
    tracer.end_step()
    summary = tracer.summary()
    assert summary["total_tool_calls"] == 2
    assert summary["failed_tool_calls"] == 1
    assert summary["tool_success_rate"] == 0.5


def test_tool_metrics_aggregation():
    metrics = ToolMetrics()
    for i in range(10):
        metrics.record("tool_a", ok=(i != 9), latency_ms=100 + i * 10,
                       error_type="timeout" if i == 9 else None)
    report = metrics.report()["tool_a"]
    assert report["calls"] == 10
    assert report["success_rate"] == 0.9
    assert report["error_distribution"] == {"timeout": 1}
    assert report["avg_latency_ms"] > 0
    assert report["p95_latency_ms"] >= report["avg_latency_ms"]
