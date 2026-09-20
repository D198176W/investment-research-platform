"""评测 Runner —— 批量跑评测集 → 规则校验 + LLM-as-Judge → 输出结构化报告

用法:
    cd backend
    python -m evals.runner --dataset evals/dataset/investment_tasks.json

输出:
    evals/reports/eval_report_{timestamp}.json
"""
import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

# 确保能 import agent_service
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from evals.metrics import run_rule_checks  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def llm_judge(state: Dict[str, Any], api_key: str, api_url: str, model: str) -> Dict[str, Any]:
    """LLM-as-Judge：让独立模型对最终报告质量打分

    维度：数据支撑、逻辑自洽、风险覆盖、可读性（各 0~1）
    """
    from agent_service.llm_client import LLMClient

    judge_prompt = f"""你是独立投研报告评审专家，请对以下投资研究报告进行客观评分（0~1）。

研究主题: {state.get('research_topic')}
最终报告:
{(state.get('final_report') or '')[:3000]}

请从四个维度打分并给出总分：
- data_support: 数据支撑（是否引用真实数据、数据是否支撑结论）
- logic: 逻辑自洽（论证链条是否完整无矛盾）
- risk_coverage: 风险覆盖（是否有具体可执行的风险应对）
- readability: 可读性（结构清晰、专业客观）

严格输出 JSON：
{{"data_support": 0.X, "logic": 0.X, "risk_coverage": 0.X, "readability": 0.X, "overall": 0.X, "comment": "简评"}}"""

    client = LLMClient(api_key, api_url, model, timeout=120)
    try:
        result = client.chat([{"role": "user", "content": judge_prompt}], temperature=0.1)
        text = result.content.strip()
        if "```" in text:
            import re
            m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if m:
                text = m.group(1)
        return json.loads(text)
    except Exception as e:
        logger.warning(f"LLM-as-Judge 失败: {e}")
        return {"error": str(e)}


def run_eval(dataset_path: str, api_key: str, api_url: str, model: str,
             use_judge: bool = True, max_tasks: int = 0) -> Dict[str, Any]:
    """运行评测集

    Args:
        dataset_path: 评测集 JSON 路径
        api_key/api_url/model: LLM 配置
        use_judge: 是否启用 LLM-as-Judge（关则只做规则校验，省 token）
        max_tasks: 最多跑多少条（0=全部，调试时可用 1）

    Returns:
        评测报告 dict
    """
    from agent_service.workflow import run_research_agent

    with open(dataset_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if max_tasks > 0:
        tasks = tasks[:max_tasks]

    results = []
    for task in tasks:
        task_id = task["id"]
        logger.info(f"===== 评测任务 {task_id}: {task['topic']} =====")
        started = time.time()
        try:
            final_state = run_research_agent(
                topic=task["topic"],
                industry=task["industry"],
                horizon=task["horizon"],
                api_key=api_key,
                api_url=api_url,
                model_name=model,
                use_realtime_data=True,
                use_agentic_perception=True,
                task_id=f"eval-{task_id}",
            )
        except Exception as e:
            logger.error(f"任务 {task_id} 执行异常: {e}")
            final_state = {"current_phase": "error", "error": str(e)}

        # 规则校验
        rule_results = run_rule_checks(final_state, task.get("expected", {}))
        rule_pass_count = sum(1 for r in rule_results if r.passed)

        # LLM-as-Judge
        judge_scores = {}
        if use_judge and final_state.get("current_phase") == "completed":
            judge_scores = llm_judge(final_state, api_key, api_url, model)

        results.append({
            "task_id": task_id,
            "topic": task["topic"],
            "latency_s": round(time.time() - started, 1),
            "rule_checks": [r.to_dict() for r in rule_results],
            "rule_pass_rate": round(rule_pass_count / len(rule_results), 4) if rule_results else 0,
            "judge_scores": judge_scores,
            "trace_summary": final_state.get("trace_summary"),
        })

    # 汇总
    avg_rule_pass = round(sum(r["rule_pass_rate"] for r in results) / len(results), 4) if results else 0
    task_complete_rate = round(
        sum(1 for r in results
            if any(c["name"] == "task_completed" and c["passed"] for c in r["rule_checks"]))
        / len(results), 4) if results else 0

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "total_tasks": len(results),
        "metrics": {
            "task_completion_rate": task_complete_rate,
            "avg_rule_pass_rate": avg_rule_pass,
        },
        "results": results,
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, f"eval_report_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"评测完成: 完成率={task_complete_rate}, 规则通过率={avg_rule_pass}")
    logger.info(f"报告已保存: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="智能投研平台评测 Runner")
    parser.add_argument("--dataset", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset", "investment_tasks.json"))
    parser.add_argument("--max-tasks", type=int, default=0, help="最多跑多少条（0=全部）")
    parser.add_argument("--no-judge", action="store_true", help="禁用 LLM-as-Judge（省 token）")
    args = parser.parse_args()

    # 从环境变量或 .env 读取配置
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(BACKEND_DIR), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    api_url = os.environ.get("DASHSCOPE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.environ.get("DEFAULT_MODEL_NAME", "qwen-plus")

    if not api_key:
        logger.error("请配置 DASHSCOPE_API_KEY")
        sys.exit(1)

    run_eval(args.dataset, api_key, api_url, model,
             use_judge=not args.no_judge, max_tasks=args.max_tasks)


if __name__ == "__main__":
    main()
