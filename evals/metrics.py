"""评测指标计算 —— 规则校验（不依赖 LLM，确定性、可复现）"""
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class RuleCheckResult:
    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def to_dict(self):
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def check_task_completed(final_state: Dict[str, Any]) -> RuleCheckResult:
    """任务完成率：是否到达 completed 阶段"""
    ok = final_state.get("current_phase") == "completed"
    return RuleCheckResult("task_completed", ok,
                           f"final_phase={final_state.get('current_phase')}")


def check_min_plans(final_state: Dict[str, Any], min_plans: int) -> RuleCheckResult:
    """方案数量：推理阶段是否生成足够候选方案"""
    plans = final_state.get("reasoning_plans") or []
    ok = isinstance(plans, list) and len(plans) >= min_plans
    return RuleCheckResult("min_plans", ok, f"plans={len(plans)}, required={min_plans}")


def check_must_mention(final_state: Dict[str, Any], keywords: List[str]) -> RuleCheckResult:
    """关键内容覆盖：报告是否包含必备要点"""
    report = final_state.get("final_report") or ""
    missing = [k for k in keywords if k not in report]
    ok = len(missing) == 0
    return RuleCheckResult("must_mention", ok,
                           f"missing={missing}" if missing else "all covered")


def check_has_reflection(final_state: Dict[str, Any], required: bool) -> RuleCheckResult:
    """反思阶段是否执行（v2 核心能力）"""
    reflection = final_state.get("reflection")
    ok = (reflection is not None) if required else True
    return RuleCheckResult("has_reflection", ok,
                           f"reflection={'present' if reflection else 'missing'}")


def check_has_risk(final_state: Dict[str, Any], required: bool) -> RuleCheckResult:
    """风险评估覆盖：决策是否包含风险评估"""
    selected = final_state.get("selected_plan") or {}
    has_risk = bool(selected.get("risk_assessment"))
    ok = has_risk if required else True
    return RuleCheckResult("has_risk_assessment", ok,
                           f"risk_assessment={'present' if has_risk else 'missing'}")


def check_tools_used(final_state: Dict[str, Any], min_tools: int) -> RuleCheckResult:
    """工具使用：Agentic 感知是否真正调用了工具"""
    perception = final_state.get("perception_data") or {}
    agentic = perception.get("_agentic") or {}
    tools_used = len(agentic.get("succeeded_tools", []))
    # 若非 Agentic 模式，检查固定数据源
    if tools_used == 0:
        tools_used = len(perception.get("_realtime_data_sources", []))
    ok = tools_used >= min_tools
    return RuleCheckResult("min_tools_used", ok,
                           f"tools_used={tools_used}, required={min_tools}")


def check_output_format(final_state: Dict[str, Any]) -> RuleCheckResult:
    """格式合规率：关键字段是否齐全"""
    required_keys = ["perception_data", "world_model", "reasoning_plans", "selected_plan", "final_report"]
    missing = [k for k in required_keys if not final_state.get(k)]
    ok = len(missing) == 0
    return RuleCheckResult("output_format", ok,
                           f"missing={missing}" if missing else "all present")


def check_direction_keywords(final_state: Dict[str, Any], keywords: List[str]) -> RuleCheckResult:
    """结论方向覆盖：报告是否体现了期望的方向性判断（至少命中 1 个方向关键词）"""
    if not keywords:
        return RuleCheckResult("direction_keywords", True, "not required")
    report = final_state.get("final_report") or ""
    hits = [k for k in keywords if k in report]
    ok = len(hits) >= 1
    return RuleCheckResult("direction_keywords", ok,
                           f"hits={hits}" if hits else f"none of {keywords} found")


def check_data_validation_applied(final_state: Dict[str, Any]) -> RuleCheckResult:
    """真实数据校验是否生效：方案应带 data_validation 字段（结论-数据一致性机制）"""
    plans = final_state.get("reasoning_plans") or []
    if not isinstance(plans, list) or not plans:
        return RuleCheckResult("data_validation", False, "no plans")
    with_dv = sum(1 for p in plans if p.get("data_validation"))
    ok = with_dv > 0
    return RuleCheckResult("data_validation", ok,
                           f"plans_with_data_validation={with_dv}/{len(plans)}")


def run_rule_checks(final_state: Dict[str, Any], expected: Dict[str, Any]) -> List[RuleCheckResult]:
    """执行全部规则校验"""
    return [
        check_task_completed(final_state),
        check_min_plans(final_state, expected.get("min_plans", 2)),
        check_must_mention(final_state, expected.get("must_mention", [])),
        check_direction_keywords(final_state, expected.get("direction_keywords", [])),
        check_has_reflection(final_state, expected.get("must_have_reflection", False)),
        check_has_risk(final_state, expected.get("must_have_risk", False)),
        check_tools_used(final_state, expected.get("min_tools_used", 1)),
        check_output_format(final_state),
        check_data_validation_applied(final_state),
    ]
