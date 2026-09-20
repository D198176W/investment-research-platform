---
name: value_investing
description: 价值投资分析框架 —— 适用于基本面扎实、寻求中长期低估机会的研究任务
triggers: [价值, 基本面, 低估, 长期, 蓝筹, 财务, 分红]
recommended_tools: [get_stock_financial, get_stock_profit, get_stock_kline, get_sector_performance, get_financial_news]
---

# 价值投资分析框架

## 分析顺序（规划指引）
1. 先用 `get_sector_performance` 判断行业景气度，确认不是衰退行业
2. 用 `get_stock_financial` / `get_stock_profit` 核查核心标的的 ROE、毛利率、资产负债率
3. 用 `get_stock_kline` 确认当前价格相对近一年区间的位置（避免追高）
4. 用 `get_financial_news` 检索近期有无重大利空

## 评估标准
- ROE 连续 > 12% 为佳；资产负债率 < 60%（金融地产除外）
- 估值与业绩增速匹配（PEG 思路）
- 必须给出明确的安全边际与风险应对，不接受"看好但无止损"的方案
