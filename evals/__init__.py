"""评测体系 —— 评测集 + 规则校验 + LLM-as-Judge + 端到端/Tool 评测

对齐 JD「评测驱动的研发体系」「端到端任务评测、模型与工程专项评测、Tool 评测」：
- evals/dataset/ : 带期望要点的投研任务集
- evals/runner.py: 批量跑任务 → 规则校验 + LLM 打分 → 输出结构化评测报告
- evals/metrics.py: 任务完成率、结论-数据一致性、引用准确率、格式合规率
"""
