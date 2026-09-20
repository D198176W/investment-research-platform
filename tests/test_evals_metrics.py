"""评测规则校验单测"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.metrics import run_rule_checks  # noqa: E402


def _completed_state():
    return {
        "current_phase": "completed",
        "perception_data": {"_agentic": {"succeeded_tools": ["t1", "t2"]}},
        "world_model": {"market_state": "震荡"},
        "reasoning_plans": [
            {"plan_id": "1", "data_validation": {"confidence_change": 0.1}},
            {"plan_id": "2", "data_validation": {"confidence_change": 0}},
        ],
        "selected_plan": {"risk_assessment": "有风险应对"},
        "reflection": {"passed": True},
        "final_report": "国产替代 算力 风险 回调 增长",
    }


def test_all_rules_pass_on_good_state():
    results = run_rule_checks(_completed_state(), {
        "min_plans": 2,
        "must_mention": ["国产替代", "风险"],
        "direction_keywords": ["增长", "回调"],
        "must_have_reflection": True,
        "must_have_risk": True,
        "min_tools_used": 2,
    })
    failed = [r.name for r in results if not r.passed]
    assert failed == [], f"应有全部通过，未通过: {failed}"


def test_rules_fail_on_incomplete_state():
    results = run_rule_checks({"current_phase": "error"}, {
        "min_plans": 2, "must_mention": ["风险"],
        "must_have_reflection": True, "must_have_risk": True,
    })
    by_name = {r.name: r.passed for r in results}
    assert by_name["task_completed"] is False
    assert by_name["min_plans"] is False
    assert by_name["output_format"] is False


def test_direction_keywords_check():
    state = _completed_state()
    results = run_rule_checks(state, {"direction_keywords": ["不存在的关键词XYZ"]})
    by_name = {r.name: r.passed for r in results}
    assert by_name["direction_keywords"] is False


def test_data_validation_check():
    state = _completed_state()
    state["reasoning_plans"] = [{"plan_id": "1"}, {"plan_id": "2"}]  # 无 data_validation
    results = run_rule_checks(state, {})
    by_name = {r.name: r.passed for r in results}
    assert by_name["data_validation"] is False
