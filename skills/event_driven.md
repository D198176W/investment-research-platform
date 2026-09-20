---
name: event_driven
description: 事件驱动分析框架 —— 适用于政策发布、技术突破、行业重大事件引发的投资机会
triggers: [政策, 事件, 新闻, 发布, 突破, 催化, 热点, 概念]
recommended_tools: [get_financial_news, get_macro_news, get_sector_top_stocks, get_top_gainers, get_capital_flow]
---

# 事件驱动分析框架

## 分析顺序（规划指引）
1. 用 `get_financial_news` / `get_macro_news` 还原事件全貌，区分"一次性脉冲"与"趋势性催化"
2. 用 `get_sector_top_stocks` / `get_top_gainers` 验证资金是否已实际流入相关标的
3. 用 `get_capital_flow` 判断主力资金态度，避免"利好出尽是利空"

## 评估标准
- 事件必须有可验证的来源与时间（引用具体新闻）
- 区分情绪溢价与业绩兑现路径，明确兑现时间表
- 必须给出"证伪条件"：出现什么信号就放弃该方案
