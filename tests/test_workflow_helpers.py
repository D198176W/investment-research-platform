"""工作流核心逻辑单测：方向检测、真实数据校验、状态机路由、图构建"""
from agent_service.workflow import (
    PHASE_ORDER,
    _detect_direction_from_hypothesis,
    _detect_direction_from_real_data,
    _get_previous_phase,
    create_research_workflow,
    router,
    validate_plans_with_real_data,
)


# ---------- 假设方向检测 ----------

def test_detect_bullish_hypothesis():
    assert _detect_direction_from_hypothesis("行业需求持续增长，板块有望上涨") == "bullish"


def test_detect_bearish_hypothesis():
    assert _detect_direction_from_hypothesis("产能过剩导致价格持续下跌，板块走弱") == "bearish"


def test_detect_neutral_hypothesis():
    assert _detect_direction_from_hypothesis("行业处于平稳发展期") is None


# ---------- 真实数据方向检测 ----------

def test_detect_direction_market_bullish():
    data = {"market_overview": {"up_count": 3000, "down_count": 1000, "avg_change_pct": 1.2}}
    assert _detect_direction_from_real_data(data)["market"] == "bullish"


def test_detect_direction_string_values_no_typeerror():
    """回归：down_count 为字符串 'N/A' 时不应 TypeError（原 and/or 优先级 bug）"""
    data = {"market_overview": {"up_count": 0, "down_count": "N/A", "avg_change_pct": 1.0}}
    directions = _detect_direction_from_real_data(data)
    assert directions["market"] == "neutral"


def test_detect_direction_indices_majority():
    data = {"indices": [
        {"name": "上证", "change_pct": 1.0},
        {"name": "深证", "change_pct": 0.5},
        {"name": "创业板", "change_pct": -0.3},
    ]}
    assert _detect_direction_from_real_data(data)["indices_summary"] == "bullish"


# ---------- 真实数据校验（结论-数据一致性） ----------

def _perception(avg_change):
    return {"_realtime_data_raw": {
        "market_overview": {"up_count": 100, "down_count": 3000, "avg_change_pct": avg_change},
    }}


def test_validation_penalizes_contradiction():
    plans = [{
        "plan_id": "方案1", "hypothesis": "板块将持续上涨走强",
        "data_support_score": 0.7, "logic_coherence_score": 0.7,
        "risk_controllability_score": 0.7, "confidence_level": 0.7,
    }]
    validated = validate_plans_with_real_data(plans, _perception(-2.0))
    dv = validated[0]["data_validation"]
    assert dv["confidence_change"] < 0, "假设看涨但数据看跌，置信度应下调"
    assert len(dv["conflicts"]) > 0
    assert validated[0]["confidence_level"] < 0.7


def test_validation_rewards_consistency():
    plans = [{
        "plan_id": "方案1", "hypothesis": "板块将下跌走弱",
        "data_support_score": 0.5, "logic_coherence_score": 0.5,
        "risk_controllability_score": 0.5, "confidence_level": 0.5,
    }]
    validated = validate_plans_with_real_data(plans, _perception(-2.0))
    dv = validated[0]["data_validation"]
    assert dv["confidence_change"] > 0, "假设与数据吻合，置信度应上调"
    assert len(dv["consistencies"]) > 0


def test_validation_neutral_hypothesis_unchanged():
    plans = [{
        "plan_id": "方案1", "hypothesis": "行业处于结构分化阶段",
        "data_support_score": 0.6, "logic_coherence_score": 0.6,
        "risk_controllability_score": 0.6, "confidence_level": 0.6,
    }]
    validated = validate_plans_with_real_data(plans, _perception(-2.0))
    assert validated[0]["data_validation"]["confidence_change"] == 0


# ---------- 状态机路由 ----------

def test_router_terminal_states():
    assert router({"current_phase": "completed"}) == "completed"
    assert router({"current_phase": "error"}) == "error"


def test_router_passthrough():
    assert router({"current_phase": "reasoning"}) == "reasoning"


def test_get_previous_phase():
    assert _get_previous_phase("modeling") == "perception"
    assert _get_previous_phase("reflect") == "decision"
    assert _get_previous_phase("perception") is None


# ---------- LangGraph 图驱动执行 ----------

def _stream(graph, state):
    seen = []
    for update in graph.stream(state, stream_mode="updates"):
        for node_name, node_state in update.items():
            seen.append(node_name)
            state.update(node_state)
    return seen


def test_graph_drives_phases_in_order(monkeypatch):
    """图驱动的真实执行：状态机按条件边逐阶段推进直至完成"""
    import agent_service.workflow as wf
    monkeypatch.setattr(wf, "perception_node",
                        lambda s: {**s, "current_phase": "modeling"})
    monkeypatch.setattr(wf, "modeling_node",
                        lambda s: {**s, "current_phase": "completed"})

    graph = wf.create_research_workflow()
    state = {"current_phase": "perception"}
    seen = _stream(graph, state)
    assert seen == ["perception", "modeling"]
    assert state["current_phase"] == "completed"


def test_graph_conditional_entry_resumes_mid_graph(monkeypatch):
    """条件入口：从检查点恢复的任务应直接从中断阶段续跑"""
    import agent_service.workflow as wf
    monkeypatch.setattr(wf, "report_node",
                        lambda s: {**s, "current_phase": "completed"})

    graph = wf.create_research_workflow()
    state = {"current_phase": "report"}
    seen = _stream(graph, state)
    assert seen == ["report"], "应从 report 阶段直接进入，而非重头执行"


def test_graph_retry_loop(monkeypatch):
    """节点失败保持原阶段时应沿条件边重试，达到重试上限后进入 error 终止"""
    import agent_service.workflow as wf
    calls = []

    def flaky(s):
        calls.append(1)
        if len(calls) >= 3:
            return {**s, "current_phase": "error", "error": "重试超限"}
        return {**s, "current_phase": "perception"}  # 重试当前阶段

    monkeypatch.setattr(wf, "perception_node", flaky)
    graph = wf.create_research_workflow()
    state = {"current_phase": "perception"}
    seen = _stream(graph, state)
    assert seen == ["perception", "perception", "perception"]
    assert state["current_phase"] == "error"
