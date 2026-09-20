---
name: industry_chain
description: 产业链分析框架 —— 适用于产业链上下游传导、国产替代、供需格局类研究任务
triggers: [产业链, 供应链, 上下游, 国产替代, 产能, 供需, 芯片, 材料]
recommended_tools: [get_sector_performance, get_sector_top_stocks, get_stock_quote, get_financial_news, get_major_indices]
---

# 产业链分析框架

## 分析顺序（规划指引）
1. 用 `get_sector_performance` 对比上中下游各环节景气度
2. 用 `get_sector_top_stocks` 找出各环节龙头，用 `get_stock_quote` 逐一核查龙头表现
3. 用 `get_major_indices` 与 `get_market_overview` 判断整体市场对该链条的定价阶段

## 评估标准
- 明确产业链中利润正在向哪个环节转移（量/价/份额三个维度）
- 国产替代类任务必须给出"替代进度"的可验证证据（订单、产能、份额数据）
- 风险提示必须覆盖"产能过剩"与"技术路线切换"两类系统性风险
