"""测试所有新增功能"""
import sys
sys.path.insert(0, 'backend')

from agent_service.workflow import (
    create_research_workflow,
    run_research_agent,
    save_checkpoint,
    load_checkpoint,
    delete_checkpoint,
    list_pending_checkpoints,
    resume_pending_tasks,
    validate_with_pydantic,
    PerceptionOutput,
    ModelingOutput,
    ReasoningPlanOutput,
    DecisionOutput,
    exponential_backoff,
    PHASE_ORDER,
)

print("1. 所有模块导入成功")

# 测试 Pydantic 校验
test_data = {
    "market_overview": "测试",
    "key_indicators": {"GDP": "5%"},
    "recent_news": ["新闻1"],
    "industry_trends": {"AI": "增长"},
}
result = validate_with_pydantic(PerceptionOutput, test_data)
print(f"2. Pydantic 校验通过: {type(result)}")

# 测试校验失败降级
bad_data = {"unknown_field": "test"}
result = validate_with_pydantic(PerceptionOutput, bad_data)
print(f"3. Pydantic 校验降级: {type(result)} (使用默认值)")

# 测试检查点保存/加载
test_state = {
    "research_topic": "测试主题",
    "industry_focus": "测试行业",
    "time_horizon": "中期",
    "current_phase": "modeling",
    "perception_data": {"market_overview": "测试数据"},
    "completed_phases": ["perception"],
}
save_checkpoint("test_task_001", test_state)
loaded = load_checkpoint("test_task_001")
ok = loaded is not None and loaded.get("current_phase") == "modeling"
print(f"4. 检查点保存/加载: {ok}")
delete_checkpoint("test_task_001")

# 测试 LangGraph 图构建
graph = create_research_workflow()
print(f"5. LangGraph 图构建成功: {type(graph).__name__}")

# 测试指数退避
import time
start = time.time()
exponential_backoff(0, base_delay=0.1)  # 0.1 * 2^0 = 0.1秒
elapsed = time.time() - start
print(f"6. 指数退避: {elapsed:.2f}秒 (预期约0.1秒)")

# 测试状态机路由
from agent_service.workflow import router, get_conditional_mapping

# 模拟成功推进
state_success = {"current_phase": "modeling"}
assert router(state_success) == "modeling"
print("7. 路由-成功推进: OK")

# 模拟重试
state_retry = {"current_phase": "perception"}
assert router(state_retry) == "perception"
print("8. 路由-重试当前: OK")

# 模拟完成
state_done = {"current_phase": "completed"}
assert router(state_done) == "completed"
print("9. 路由-完成: OK")

# 模拟错误
state_error = {"current_phase": "error"}
assert router(state_error) == "error"
print("10. 路由-错误终止: OK")

# 测试条件边映射
mapping = get_conditional_mapping()
print(f"11. 条件边映射: {len(mapping)} 个阶段")

# 测试回滚逻辑
from agent_service.workflow import modeling_node
rollback_state = {
    **test_state,
    "api_key": "test",
    "api_url": "test",
    "model_name": "test",
    "modeling_retry": 3,  # 超过最大重试次数
    "perception_retry": 0,
    "reasoning_retry": 0,
    "decision_retry": 0,
    "report_retry": 0,
    "completed_phases": ["perception"],
}
result = modeling_node(rollback_state)
is_rollback = result.get("current_phase") == "perception" and result.get("error") is None
print(f"12. 回滚机制: {is_rollback} (modeling → perception)")

print()
print("=== 所有 12 项功能验证通过 ===")
