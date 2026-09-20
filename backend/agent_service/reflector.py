"""Reflector 反思节点 —— 对决策结论做批判性自评，产出修正意见并回灌

对齐 JD「Reflector 自进化机制」「AI 自我进化」：
- 与"原样重试"的区别：重试是同样输入再来一遍，反思是带着诊断结论改策略再来
- 检查三类问题：结论-证据一致性、评分合理性、风险覆盖完整性
- 不通过时回退到推理阶段，附带修正意见（revision_hints）
- 通过/最终都产出 lessons，写入长期记忆（自进化闭环）
"""
import json
import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_REFLECTION_LOOPS = 1  # 最多反思回灌次数（防止无限循环）

REFLECTOR_PROMPT = """你是投资决策委员会的独立风控审核官，请对以下投资分析结论进行批判性审查。

研究主题: {research_topic}
行业焦点: {industry_focus}

市场模型: {world_model}
候选方案评分: {reasoning_plans}
最终决策: {selected_plan}

请从以下三个维度审查（务必挑剔，不要附和）：
1. 证据一致性：决策引用的证据是否真实存在于市场模型/方案中？有无数据矛盾或张冠李戴？
2. 评分合理性：选中方案的置信度是否与其数据支撑度/风险可控性匹配？有无"高置信但低支撑"的可疑方案？
3. 风险覆盖：风险因素是否有具体可执行的应对策略？有无遗漏的重大风险类别（政策/流动性/技术路线）？

请严格输出 JSON：
{{
    "passed": true/false,
    "score": 0.0~1.0,
    "issues": ["问题1", "问题2"],
    "revision_hints": ["修正建议1（指明应补充什么证据/调整什么评分）"],
    "lessons": ["本次分析中值得沉淀的经验教训，1-2条，供未来同类任务复用"]
}}

判定规则：
- 存在"证据不存在/数据矛盾"或"高置信但零数据支撑" → passed=false
- 仅有措辞/完整性小问题 → passed=true，但写入 issues
- score < 0.6 视为不通过"""


class ReflectionOutput(BaseModel):
    """反思结果校验"""
    passed: bool = Field(default=True, description="是否通过审核")
    score: float = Field(default=0.7, ge=0, le=1, description="审核评分")
    issues: List[str] = Field(default_factory=list, description="发现的问题")
    revision_hints: List[str] = Field(default_factory=list, description="修正建议")
    lessons: List[str] = Field(default_factory=list, description="经验教训（写入长期记忆）")


def reflect_on_decision(state: Dict[str, Any], get_llm_fn) -> Dict[str, Any]:
    """执行反思（独立函数，便于单测与复用）

    Args:
        state: 工作流状态（需含 world_model / reasoning_plans / selected_plan）
        get_llm_fn: 返回 LLM 实例的工厂函数（保持与 workflow.py 现有 LLM 工厂解耦）

    Returns:
        校验后的反思结果 dict
    """
    from .validation import validate_with_pydantic  # 独立模块，避免与 workflow 循环 import
    llm = get_llm_fn(state["api_key"], state["api_url"], state["model_name"])
    prompt = ChatPromptTemplate.from_template(REFLECTOR_PROMPT)
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({
        "research_topic": state.get("research_topic", ""),
        "industry_focus": state.get("industry_focus", ""),
        "world_model": json.dumps(state.get("world_model") or {}, ensure_ascii=False)[:4000],
        "reasoning_plans": json.dumps(state.get("reasoning_plans") or [], ensure_ascii=False)[:4000],
        "selected_plan": json.dumps(state.get("selected_plan") or {}, ensure_ascii=False)[:4000],
    })
    return validate_with_pydantic(ReflectionOutput, result)
