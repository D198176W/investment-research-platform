"""执行轨迹（Trace）—— 记录每一步的输入/输出/工具调用/token/耗时，支持回放与瓶颈定位

对齐 JD「推理轨迹构建」「通过执行轨迹定位模型、工程与 Tool 的能力边界」「评测驱动」：
- 每个阶段产生一个 TraceStep（含 LLM 调用与工具调用明细）
- 任务结束落盘 traces/{task_id}.json
- ToolMetrics 聚合工具级指标（成功率/时延/异常分布）
"""
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 项目根/data/traces（与 memory_store/checkpoints 同级，便于容器化挂载）
# tracer.py 位于 backend/agent_service/，向上 3 级到项目根
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRACE_DIR = os.path.join(_PROJECT_ROOT, "data", "traces")


@dataclass
class LLMCall:
    """一次 LLM 调用记录"""
    purpose: str                  # 调用目的（如 perception_plan / reflect）
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    input_preview: str = ""       # 输入截断预览
    output_preview: str = ""      # 输出截断预览

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "purpose": self.purpose,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "input_preview": self.input_preview[:500],
            "output_preview": self.output_preview[:500],
        }


@dataclass
class TraceStep:
    """一个执行步骤（一个阶段或一次工具调用循环）"""
    step: str
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    status: str = "running"       # running / ok / error
    error: Optional[str] = None
    llm_calls: List[LLMCall] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    output_summary: str = ""

    def finish(self, status: str = "ok", error: Optional[str] = None,
               output_summary: str = ""):
        self.ended_at = time.time()
        self.status = status
        self.error = error
        self.output_summary = output_summary[:800]

    @property
    def latency_ms(self) -> int:
        if not self.ended_at:
            return 0
        return int((self.ended_at - self.started_at) * 1000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "llm_calls": [c.to_dict() for c in self.llm_calls],
            "tool_calls": self.tool_calls,
            "output_summary": self.output_summary,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


class Tracer:
    """单次任务的轨迹记录器（线程安全）"""

    def __init__(self, task_id: str, trace_dir: str = TRACE_DIR):
        self.task_id = task_id
        self.trace_dir = trace_dir
        self.steps: List[TraceStep] = []
        self._current: Optional[TraceStep] = None
        # 挂起缓冲：图驱动（graph.stream）执行时，节点内部的 record_llm/record_tool
        # 发生在步骤边界确定之前，先进入缓冲，start_step 时冲刷进对应步骤
        self._pending_llm: List[LLMCall] = []
        self._pending_tools: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.started_at = time.time()

    # ---------- 步骤生命周期 ----------

    def start_step(self, step: str) -> TraceStep:
        with self._lock:
            ts = TraceStep(step=step)
            if self._pending_llm:
                ts.llm_calls.extend(self._pending_llm)
                self._pending_llm.clear()
            if self._pending_tools:
                ts.tool_calls.extend(self._pending_tools)
                self._pending_tools.clear()
            self.steps.append(ts)
            self._current = ts
            return ts

    def end_step(self, status: str = "ok", error: Optional[str] = None,
                 output_summary: str = ""):
        with self._lock:
            if self._current:
                self._current.finish(status=status, error=error, output_summary=output_summary)
                self._current = None

    # ---------- 记录 ----------

    def record_llm(self, purpose: str, model: str, prompt_tokens: int = 0,
                   completion_tokens: int = 0, latency_ms: int = 0,
                   input_preview: str = "", output_preview: str = ""):
        call = LLMCall(
            purpose=purpose, model=model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            input_preview=input_preview, output_preview=output_preview,
        )
        with self._lock:
            if self._current:
                self._current.llm_calls.append(call)
            else:
                self._pending_llm.append(call)

    def record_tool(self, tool_call: Dict[str, Any]):
        with self._lock:
            if self._current:
                self._current.tool_calls.append(tool_call)
            else:
                self._pending_tools.append(tool_call)

    # ---------- 汇总与落盘 ----------

    def summary(self) -> Dict[str, Any]:
        """任务级汇总指标"""
        total_llm_tokens = sum(c.total_tokens for s in self.steps for c in s.llm_calls)
        total_tool_calls = sum(len(s.tool_calls) for s in self.steps)
        failed_tools = sum(1 for s in self.steps for t in s.tool_calls if not t.get("ok", True))
        return {
            "task_id": self.task_id,
            "total_steps": len(self.steps),
            "total_latency_ms": int((time.time() - self.started_at) * 1000),
            "total_llm_tokens": total_llm_tokens,
            "total_tool_calls": total_tool_calls,
            "failed_tool_calls": failed_tools,
            "tool_success_rate": round(1 - failed_tools / total_tool_calls, 4) if total_tool_calls else None,
            "step_breakdown": [
                {"step": s.step, "status": s.status, "latency_ms": s.latency_ms}
                for s in self.steps
            ],
        }

    def save(self) -> str:
        """落盘 traces/{task_id}.json"""
        os.makedirs(self.trace_dir, exist_ok=True)
        path = os.path.join(self.trace_dir, f"{self.task_id}.json")
        payload = {
            "task_id": self.task_id,
            "started_at": self.started_at,
            "saved_at": time.time(),
            "summary": self.summary(),
            "steps": [s.to_dict() for s in self.steps],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Trace 已保存: {path} | {payload['summary']}")
        return path

    @staticmethod
    def load(task_id: str, trace_dir: str = TRACE_DIR) -> Optional[Dict[str, Any]]:
        path = os.path.join(trace_dir, f"{task_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class ToolMetrics:
    """工具级指标聚合（跨任务，进程内累计）—— 用于定位 Tool 能力边界"""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats: Dict[str, Dict[str, Any]] = {}

    def record(self, tool_name: str, ok: bool, latency_ms: int, error_type: Optional[str] = None):
        with self._lock:
            s = self._stats.setdefault(tool_name, {
                "calls": 0, "success": 0, "fail": 0,
                "latencies": [], "errors": {},
            })
            s["calls"] += 1
            if ok:
                s["success"] += 1
            else:
                s["fail"] += 1
                if error_type:
                    s["errors"][error_type] = s["errors"].get(error_type, 0) + 1
            s["latencies"].append(latency_ms)
            if len(s["latencies"]) > 500:
                s["latencies"] = s["latencies"][-500:]

    def report(self) -> Dict[str, Any]:
        with self._lock:
            out = {}
            for name, s in self._stats.items():
                lats = sorted(s["latencies"])
                p95 = lats[int(len(lats) * 0.95)] if lats else 0
                out[name] = {
                    "calls": s["calls"],
                    "success_rate": round(s["success"] / s["calls"], 4) if s["calls"] else 0,
                    "avg_latency_ms": int(sum(lats) / len(lats)) if lats else 0,
                    "p95_latency_ms": p95,
                    "error_distribution": s["errors"],
                }
            return out


# 全局工具指标单例
tool_metrics = ToolMetrics()
