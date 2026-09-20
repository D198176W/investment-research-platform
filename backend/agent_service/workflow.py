"""LangGraph 五阶段工作流 - 支持状态机回滚、指数退避重试、检查点持久化、Pydantic 校验

v2 升级（对齐大厂 Agent 工程能力要求）：
- Agentic 感知：工具调用循环（Function Calling）替代固定数据抓取，模型自主选择工具
- Reflector 反思节点：决策后独立审核，产出修正意见回灌 + 经验教训写入长期记忆
- 三层记忆：工作记忆(state) / 短期记忆(上下文窗口) / 长期记忆(向量库)
- 全链路 Trace：每阶段记录 LLM/Tool 调用明细，落盘 traces/{task_id}.json
"""
import json
import os
import time
import logging
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Tongyi
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, ValidationError

from .validation import validate_with_pydantic
from .tracer import Tracer, tool_metrics

logger = logging.getLogger(__name__)

# ==================== 常量配置 ====================

MAX_RETRIES = 3
BASE_RETRY_DELAY = 2  # 指数退避基础延迟（秒）
# 项目根/data/checkpoints（与 memory_store/traces 同级，便于容器化挂载）
CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "checkpoints")


# ==================== Pydantic 校验模型 ====================

class PerceptionOutput(BaseModel):
    """感知阶段 LLM 输出校验"""
    market_overview: str = Field(default="", description="市场概况")
    key_indicators: Dict[str, str] = Field(default_factory=dict, description="关键指标")
    recent_news: List[str] = Field(default_factory=list, description="近期新闻")
    industry_trends: Dict[str, str] = Field(default_factory=dict, description="行业趋势")


class ModelingOutput(BaseModel):
    """建模阶段 LLM 输出校验"""
    market_state: str = Field(default="", description="市场状态")
    economic_cycle: str = Field(default="", description="经济周期")
    risk_factors: List[str] = Field(default_factory=list, description="风险因素")
    opportunity_areas: List[str] = Field(default_factory=list, description="机会领域")
    market_sentiment: str = Field(default="", description="市场情绪")


class ReasoningPlanOutput(BaseModel):
    """推理阶段单个方案校验 — 支持分维度评分与综合置信度"""
    plan_id: str = Field(default="", description="方案ID")
    hypothesis: str = Field(default="", description="投资假设")
    analysis_approach: str = Field(default="", description="分析方法")
    expected_outcome: str = Field(default="", description="预期结果")
    # ---- 三维度评分（各0~1，由LLM按规则给出）----
    data_support_score: float = Field(default=0.5, ge=0, le=1, description="数据支撑度(0~1)：假设是否有真实市场数据支撑")
    logic_coherence_score: float = Field(default=0.5, ge=0, le=1, description="逻辑自洽性(0~1)：论证链条是否完整一致")
    risk_controllability_score: float = Field(default=0.5, ge=0, le=1, description="风险可控性(0~1)：风险因素是否有应对策略")
    # ---- 综合置信度（三维度加权：40%+30%+30%）----
    confidence_level: float = Field(default=0.5, ge=0, le=1, description="综合置信度(0~1)=数据支撑40%+逻辑自洽30%+风险可控30%")
    confidence_explanation: str = Field(default="", description="置信度评分理由简述")
    pros: List[str] = Field(default_factory=list, description="优势")
    cons: List[str] = Field(default_factory=list, description="劣势")


class DecisionOutput(BaseModel):
    """决策阶段 LLM 输出校验"""
    selected_plan_id: str = Field(default="", description="选中方案")
    investment_thesis: str = Field(default="", description="投资论点")
    supporting_evidence: List[str] = Field(default_factory=list, description="支持证据")
    risk_assessment: str = Field(default="", description="风险评估")
    recommendation: str = Field(default="", description="投资建议")
    timeframe: str = Field(default="", description="时间框架")


# ==================== Pydantic 校验辅助函数 ====================
# validate_with_pydantic 已迁移至 agent_service.validation（此处仅保留导入，供下游兼容）

# ==================== 真实数据校验（改进二） ====================

# 关键词映射：方案假设中出现的"看涨"类关键词和"看跌"类关键词
_BULLISH_KEYWORDS = ["上涨", "看涨", "利好", "回升", "反弹", "增长", "突破", "强势", "牛市", "上行", "攀升", "走高"]
_BEARISH_KEYWORDS = ["下跌", "看跌", "利空", "回调", "回落", "衰退", "破位", "弱势", "熊市", "下行", "下滑", "走低"]

# 置信度校验下调系数：假设与真实数据矛盾时，每处矛盾下调的幅度
_CONFLICT_PENALTY = 0.10
# 置信度校验上调系数：假设与真实数据吻合时，每处吻合上调的幅度
_CONSISTENCY_BONUS = 0.05
# 置信度校验后的上下限
_CONFIDENCE_MIN = 0.10
_CONFIDENCE_MAX = 1.00


def _detect_direction_from_hypothesis(hypothesis: str) -> Optional[str]:
    """从方案假设文本中检测方向倾向

    Returns:
        "bullish"（看涨）、"bearish"（看跌）或 None（无明显方向）
    """
    bullish_count = sum(1 for kw in _BULLISH_KEYWORDS if kw in hypothesis)
    bearish_count = sum(1 for kw in _BEARISH_KEYWORDS if kw in hypothesis)
    if bullish_count > bearish_count:
        return "bullish"
    elif bearish_count > bullish_count:
        return "bearish"
    return None


def _detect_direction_from_real_data(perception_data: Dict[str, Any]) -> Dict[str, str]:
    """从感知阶段的真实行情数据中提取各维度方向

    Returns:
        字典，包含各维度的实际方向：
        {
            "market": "bullish"/"bearish"/"neutral",
            "sector": "bullish"/"bearish"/"neutral",
            "indices_summary": "bullish"/"bearish"/"neutral"
        }
    """
    directions = {"market": "neutral", "sector": "neutral", "indices_summary": "neutral"}

    # 1. 市场概览方向（上涨家数 vs 下跌家数）
    mo = perception_data.get("market_overview")
    if isinstance(mo, dict):
        up = mo.get("up_count", 0)
        down = mo.get("down_count", 0)
        avg_change = mo.get("avg_change_pct", 0)
        try:
            avg_change = float(avg_change) if avg_change != "N/A" else 0
        except (ValueError, TypeError):
            avg_change = 0
        if (isinstance(up, (int, float)) and isinstance(down, (int, float))
                and (up > 0 or down > 0)):
            if avg_change > 0.5:
                directions["market"] = "bullish"
            elif avg_change < -0.5:
                directions["market"] = "bearish"

    # 2. 板块方向（板块涨跌幅）
    sec = perception_data.get("sector")
    if isinstance(sec, dict) and "error" not in sec:
        change_pct = sec.get("change_pct", "N/A")
        try:
            change_pct = float(change_pct) if change_pct != "N/A" else 0
        except (ValueError, TypeError):
            change_pct = 0
        if change_pct > 0.5:
            directions["sector"] = "bullish"
        elif change_pct < -0.5:
            directions["sector"] = "bearish"

    # 3. 指数汇总方向（多数指数涨还是跌）
    indices = perception_data.get("indices")
    if isinstance(indices, list) and len(indices) > 0:
        up_indices = 0
        down_indices = 0
        for idx in indices:
            if "error" not in idx:
                change_pct = idx.get("change_pct", "N/A")
                try:
                    change_pct = float(change_pct) if change_pct != "N/A" else 0
                except (ValueError, TypeError):
                    change_pct = 0
                if change_pct > 0:
                    up_indices += 1
                elif change_pct < 0:
                    down_indices += 1
        if up_indices > down_indices:
            directions["indices_summary"] = "bullish"
        elif down_indices > up_indices:
            directions["indices_summary"] = "bearish"

    return directions


def validate_plans_with_real_data(plans: List[Dict[str, Any]], perception_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """改进二：用感知阶段的真实行情数据校验方案置信度

    对比每个方案的假设方向与真实数据方向，自动调整 confidence_level：
    - 假设看涨但实际数据看跌 → 置信度下调（矛盾惩罚）
    - 假设看涨且实际数据看涨 → 置信度上调（吻合奖励）
    - 无明显方向 → 不调整

    同时为每个方案新增 data_validation 字段，记录校验结果。

    Args:
        plans: 推理阶段生成的3个方案列表
        perception_data: 感知阶段的真实数据（含 realtime_data 字段）

    Returns:
        校验后的方案列表，每个方案新增 data_validation 字段
    """
    # 从 perception_data 中提取真实数据（优先用 _realtime_data_sources 标记的原始数据）
    # perception_data 本身是 LLM 结构化输出，需要从 state 中获取原始 realtime_data
    # 这里通过 perception_data 内嵌的原始数据字段来判断方向
    realtime_raw = perception_data.get("_realtime_data_raw")
    if not realtime_raw:
        # 尝试从 perception_data 的 key_indicators 等字段推断
        realtime_raw = perception_data

    real_directions = _detect_direction_from_real_data(realtime_raw)
    logger.info(f"真实数据方向检测: {real_directions}")

    validated_plans = []
    for plan in plans:
        hypothesis = plan.get("hypothesis", "")
        plan_direction = _detect_direction_from_hypothesis(hypothesis)

        # 计算方向冲突
        conflicts = []
        consistencies = []
        if plan_direction:
            # 对比市场方向
            for dimension, real_dir in real_directions.items():
                if real_dir != "neutral":
                    if plan_direction != real_dir:
                        conflicts.append(f"假设{plan_direction}但{dimension}实际{real_dir}")
                    else:
                        consistencies.append(f"假设{plan_direction}与{dimension}实际{real_dir}吻合")

        # 调整置信度
        original_confidence = plan.get("confidence_level", 0.5)
        adjustment = 0.0
        adjustment -= len(conflicts) * _CONFLICT_PENALTY
        adjustment += len(consistencies) * _CONSISTENCY_BONUS
        adjusted_confidence = max(_CONFIDENCE_MIN, min(_CONFIDENCE_MAX, original_confidence + adjustment))

        # 同步调整 data_support_score（如果与真实数据矛盾，数据支撑度应降低）
        original_data_support = plan.get("data_support_score", 0.5)
        data_support_adjustment = -len(conflicts) * 0.08 + len(consistencies) * 0.04
        adjusted_data_support = max(0.0, min(1.0, original_data_support + data_support_adjustment))

        # 重新计算综合置信度（加权公式：40%+30%+30%）
        logic_score = plan.get("logic_coherence_score", 0.5)
        risk_score = plan.get("risk_controllability_score", 0.5)
        recalculated_confidence = adjusted_data_support * 0.4 + logic_score * 0.3 + risk_score * 0.3
        recalculated_confidence = max(_CONFIDENCE_MIN, min(_CONFIDENCE_MAX, recalculated_confidence))

        # 记录校验结果
        validation_result = {
            "original_confidence": original_confidence,
            "adjusted_confidence": recalculated_confidence,
            "confidence_change": round(recalculated_confidence - original_confidence, 3),
            "plan_direction": plan_direction or "neutral",
            "real_data_direction": real_directions,
            "conflicts": conflicts,
            "consistencies": consistencies,
            "validation_note": ""
        }

        if conflicts:
            validation_result["validation_note"] = f"⚠️ 假设与{len(conflicts)}处真实数据矛盾，置信度已下调"
        elif consistencies:
            validation_result["validation_note"] = f"✅ 假设与{len(consistencies)}处真实数据吻合，置信度已上调"
        else:
            validation_result["validation_note"] = "— 假设无明显方向倾向，置信度未调整"

        # 更新方案字段
        updated_plan = {**plan}
        updated_plan["confidence_level"] = round(recalculated_confidence, 3)
        updated_plan["data_support_score"] = round(adjusted_data_support, 3)
        updated_plan["data_validation"] = validation_result

        logger.info(f"方案 {plan.get('plan_id')} 校验: "
                     f"原始置信度={original_confidence:.3f} → 调整后={recalculated_confidence:.3f} "
                     f"(冲突{len(conflicts)}处, 吻合{len(consistencies)}处, {validation_result['validation_note']})")

        validated_plans.append(updated_plan)

    return validated_plans


# ==================== 多模型评估（改进三） ====================

# 可用于交叉评估的模型列表（按可靠性排序）
_CROSS_EVAL_MODELS = [
    {"model_name": "qwen-max", "weight": 0.6, "label": "qwen-max(高可靠)"},
    {"model_name": "qwen-plus", "weight": 0.4, "label": "qwen-plus(中等可靠)"},
]

# 多模型评估 Prompt：只评估不生成，输入已有方案让模型独立打分
_CROSS_EVAL_PROMPT = """你是独立的投资方案评审专家，请对以下候选方案进行客观评分。
不要修改方案内容，只需对每个方案在三个维度上独立打分(0~1)。

候选方案: {plans_json}

请对每个方案评分，输出JSON数组:
[
    {{"plan_id": "方案1", "data_support_score": 0.X, "logic_coherence_score": 0.X, "risk_controllability_score": 0.X}},
    {{"plan_id": "方案2", "data_support_score": 0.X, "logic_coherence_score": 0.X, "risk_controllability_score": 0.X}},
    {{"plan_id": "方案3", "data_support_score": 0.X, "logic_coherence_score": 0.X, "risk_controllability_score": 0.X}}
]

评分标准：
- 数据支撑度: 假设是否有真实市场数据支撑，引用数据是否准确
- 逻辑自洽性: 从假设到预期结果的论证链条是否完整无矛盾
- 风险可控性: 风险因素是否有明确应对策略"""


def multi_model_score(plans: List[Dict[str, Any]], api_key: str, api_url: str,
                      primary_model: str) -> List[Dict[str, Any]]:
    """改进三：多模型交叉评估方案置信度

    让不同模型（qwen-max, qwen-plus）对同一组方案独立评分，
    再将各模型评分与主模型评分加权融合，减少单一模型偏差。

    融合规则：
    - 主模型评分权重 0.5（因为主模型既生成了方案又评分，有自评偏差）
    - 辅助模型评分权重 0.5（独立视角，更客观）
    - 融合后 confidence_level = 主模型×0.5 + 辅助模型均值×0.5
    - 融合后各维度 score 也按同样权重融合

    Args:
        plans: 已生成的3个方案列表
        api_key: DashScope API密钥
        api_url: DashScope API地址
        primary_model: 主模型名称

    Returns:
        融合评分后的方案列表，每个方案新增 cross_eval_results 字段
    """
    logger.info(f"开始多模型交叉评估，主模型: {primary_model}")

    # 筛选辅助模型（排除主模型，避免自己评自己）
    aux_models = [m for m in _CROSS_EVAL_MODELS if m["model_name"] != primary_model]
    if not aux_models:
        logger.warning(f"没有可用的辅助模型进行交叉评估（主模型{primary_model}与所有辅助模型相同），跳过")
        return plans

    # 只提取方案核心内容给辅助模型评分（不含主模型的自评分，避免锚定效应）
    plans_for_eval = []
    for p in plans:
        plans_for_eval.append({
            "plan_id": p.get("plan_id"),
            "hypothesis": p.get("hypothesis"),
            "analysis_approach": p.get("analysis_approach"),
            "expected_outcome": p.get("expected_outcome"),
            "pros": p.get("pros"),
            "cons": p.get("cons"),
        })

    plans_json = json.dumps(plans_for_eval, ensure_ascii=False)

    cross_eval_scores = {}  # {model_label: {plan_id: {data_support, logic, risk}}}

    for aux_model_info in aux_models:
        model_name = aux_model_info["model_name"]
        model_label = aux_model_info["label"]
        logger.info(f"辅助模型 {model_label} 开始独立评分...")

        try:
            llm = get_llm(api_key, api_url, model_name)
            prompt = ChatPromptTemplate.from_template(_CROSS_EVAL_PROMPT)
            chain = prompt | llm | JsonOutputParser()
            result = chain.invoke({"plans_json": plans_json})

            # 处理结果格式
            if isinstance(result, list):
                scores_list = result
            elif isinstance(result, dict):
                # LLM 可能用 key 包裹
                for v in result.values():
                    if isinstance(v, list):
                        scores_list = v
                        break
                else:
                    scores_list = []
            else:
                scores_list = []

            # 按方案ID组织评分
            model_scores = {}
            for score_item in scores_list:
                pid = score_item.get("plan_id", "")
                model_scores[pid] = {
                    "data_support_score": float(score_item.get("data_support_score", 0.5)),
                    "logic_coherence_score": float(score_item.get("logic_coherence_score", 0.5)),
                    "risk_controllability_score": float(score_item.get("risk_controllability_score", 0.5)),
                }
            cross_eval_scores[model_label] = model_scores
            logger.info(f"辅助模型 {model_label} 评分完成: {model_scores}")

        except Exception as e:
            logger.warning(f"辅助模型 {model_label} 评估失败: {e}，跳过该模型")

    if not cross_eval_scores:
        logger.warning("所有辅助模型评估均失败，保留主模型原始评分")
        return plans

    # 融合评分：主模型权重0.5 + 辅助模型均值权重0.5
    PRIMARY_WEIGHT = 0.5
    AUX_WEIGHT = 0.5
    num_aux = len(cross_eval_scores)
    per_aux_weight = AUX_WEIGHT / num_aux  # 每个辅助模型的权重

    fused_plans = []
    for plan in plans:
        pid = plan.get("plan_id")
        updated_plan = {**plan}

        # 获取主模型评分
        primary_ds = plan.get("data_support_score", 0.5)
        primary_lc = plan.get("logic_coherence_score", 0.5)
        primary_rc = plan.get("risk_controllability_score", 0.5)

        # 获取辅助模型评分并加权融合
        fused_ds = primary_ds * PRIMARY_WEIGHT
        fused_lc = primary_lc * PRIMARY_WEIGHT
        fused_rc = primary_rc * PRIMARY_WEIGHT

        aux_details = []
        for model_label, model_scores in cross_eval_scores.items():
            aux_score = model_scores.get(pid, {})
            aux_ds = aux_score.get("data_support_score", 0.5)
            aux_lc = aux_score.get("logic_coherence_score", 0.5)
            aux_rc = aux_score.get("risk_controllability_score", 0.5)

            fused_ds += aux_ds * per_aux_weight
            fused_lc += aux_lc * per_aux_weight
            fused_rc += aux_rc * per_aux_weight

            aux_details.append({
                "model": model_label,
                "data_support_score": round(aux_ds, 3),
                "logic_coherence_score": round(aux_lc, 3),
                "risk_controllability_score": round(aux_rc, 3),
            })

        # 重新计算综合置信度
        fused_confidence = fused_ds * 0.4 + fused_lc * 0.3 + fused_rc * 0.3
        fused_confidence = max(_CONFIDENCE_MIN, min(_CONFIDENCE_MAX, fused_confidence))

        # 更新方案字段
        updated_plan["data_support_score"] = round(fused_ds, 3)
        updated_plan["logic_coherence_score"] = round(fused_lc, 3)
        updated_plan["risk_controllability_score"] = round(fused_rc, 3)
        updated_plan["confidence_level"] = round(fused_confidence, 3)
        updated_plan["cross_eval_results"] = {
            "primary_model": primary_model,
            "primary_scores": {
                "data_support_score": round(primary_ds, 3),
                "logic_coherence_score": round(primary_lc, 3),
                "risk_controllability_score": round(primary_rc, 3),
            },
            "aux_models": aux_details,
            "fusion_weights": {"primary": PRIMARY_WEIGHT, "per_aux": round(per_aux_weight, 3)},
            "fusion_note": f"置信度融合: 主模型({primary_model})×{PRIMARY_WEIGHT} + "
                           f"{num_aux}个辅助模型均值×{AUX_WEIGHT}",
        }

        logger.info(f"方案 {pid} 多模型融合: "
                     f"数据支撑={fused_ds:.3f}, 逻辑自洽={fused_lc:.3f}, "
                     f"风险可控={fused_rc:.3f}, 综合置信度={fused_confidence:.3f}")

        fused_plans.append(updated_plan)

    return fused_plans


# ==================== 检查点持久化 ====================

def _ensure_checkpoint_dir():
    """确保检查点目录存在"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_checkpoint(task_id: str, state: Dict[str, Any]):
    """将当前工作流状态保存为检查点文件

    保存每个阶段已完成的中间结果，支持断点续跑。
    """
    _ensure_checkpoint_dir()
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{task_id}.json")

    # 只保存可序列化的关键字段
    save_fields = [
        "research_topic", "industry_focus", "time_horizon",
        "api_key", "api_url", "model_name", "use_realtime_data", "use_agentic_perception",
        "current_phase", "perception_data", "world_model",
        "reasoning_plans", "selected_plan", "reflection", "final_report",
        "error",
        "perception_retry", "modeling_retry", "reasoning_retry",
        "decision_retry", "reflect_retry", "report_retry",
        "reflection_count",
        "last_completed_phase", "completed_phases",
    ]

    checkpoint = {}
    for field in save_fields:
        value = state.get(field)
        if value is not None:
            # 尝试序列化，失败则跳过
            try:
                json.dumps(value, ensure_ascii=False)
                checkpoint[field] = value
            except (TypeError, ValueError):
                pass

    checkpoint["saved_at"] = datetime.now().isoformat()

    try:
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        logger.info(f"检查点已保存: {task_id} (阶段: {state.get('current_phase')})")
    except Exception as e:
        logger.error(f"保存检查点失败: {e}")


def load_checkpoint(task_id: str) -> Optional[Dict[str, Any]]:
    """加载检查点文件，用于断点续跑

    Returns:
        检查点状态字典，不存在则返回 None
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{task_id}.json")
    if not os.path.exists(checkpoint_path):
        return None

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        logger.info(f"检查点已加载: {task_id} (保存于 {checkpoint.get('saved_at', '未知')})")
        return checkpoint
    except Exception as e:
        logger.error(f"加载检查点失败: {e}")
        return None


def delete_checkpoint(task_id: str):
    """删除检查点文件（任务完成后清理）"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{task_id}.json")
    try:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info(f"检查点已删除: {task_id}")
    except Exception as e:
        logger.error(f"删除检查点失败: {e}")


def list_pending_checkpoints() -> List[str]:
    """列出所有未完成的检查点（用于服务重启后恢复）"""
    _ensure_checkpoint_dir()
    pending = []
    for filename in os.listdir(CHECKPOINT_DIR):
        if filename.endswith(".json"):
            task_id = filename[:-5]
            checkpoint = load_checkpoint(task_id)
            if checkpoint and checkpoint.get("current_phase") not in ("completed", "error"):
                pending.append(task_id)
    return pending


# ==================== 实时数据获取 ====================

def _fetch_realtime_data(topic: str, industry: str) -> Dict[str, Any]:
    """从 AKShare 抓取真实市场数据，用于注入感知阶段 Prompt"""
    import concurrent.futures

    realtime_data = {
        "quotes": None, "sector": None, "news": None,
        "indices": None, "market_overview": None,
        "data_sources": [], "fetch_errors": [],
    }

    try:
        from data_fetcher.stock_data import StockDataFetcher
        from data_fetcher.news_fetcher import NewsFetcher
        from data_fetcher.market_indices import MarketIndicesFetcher
        from data_fetcher.config import get_data_fetcher_settings

        settings = get_data_fetcher_settings()
        stock_fetcher = StockDataFetcher()
        news_fetcher = NewsFetcher()
        indices_fetcher = MarketIndicesFetcher()
        sector_name = settings.INDUSTRY_SECTOR_MAP.get(industry, industry)
        keyword = settings.TOPIC_KEYWORD_MAP.get(topic, topic)

        def fetch_news():
            try:
                data = news_fetcher.get_financial_news(keyword=keyword, limit=5)
                if data:
                    return ("news", data, f"财经新闻({keyword})")
                return ("skip", None, None)
            except Exception as e:
                return ("error", None, f"新闻异常: {e}")

        def fetch_indices():
            try:
                data = indices_fetcher.get_major_indices()
                if data and not (len(data) == 1 and data[0].get("error")):
                    return ("indices", data, "主要指数")
                return ("error", None, "指数数据获取失败")
            except Exception as e:
                return ("error", None, f"指数异常: {e}")

        def fetch_sector():
            try:
                data = stock_fetcher.get_sector_performance(sector_name)
                if isinstance(data, dict) and "error" not in data:
                    return ("sector", data, f"板块行情({sector_name})")
                return ("error", None, "板块数据不可用")
            except Exception as e:
                return ("error", None, f"板块数据异常: {e}")

        def fetch_sector_stocks():
            try:
                data = stock_fetcher.get_sector_top_stocks(sector_name, limit=5)
                if data:
                    return ("sector_top_stocks", data, f"板块成分股({sector_name})")
                return ("skip", None, None)
            except Exception as e:
                return ("error", None, f"成分股异常: {e}")

        def fetch_overview():
            try:
                data = indices_fetcher.get_market_overview()
                if isinstance(data, dict) and "error" not in data:
                    return ("market_overview", data, "市场概览")
                return ("error", None, "市场概览不可用")
            except Exception as e:
                return ("error", None, f"市场概览异常: {e}")

        tasks = [fetch_news, fetch_indices, fetch_sector, fetch_sector_stocks, fetch_overview]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(task) for task in tasks]
            for future in concurrent.futures.as_completed(futures, timeout=20):
                try:
                    result_type, data, source = future.result(timeout=15)
                    if result_type == "error":
                        realtime_data["fetch_errors"].append(source)
                    elif result_type != "skip":
                        realtime_data[result_type] = data
                        if source:
                            realtime_data["data_sources"].append(source)
                except Exception as e:
                    realtime_data["fetch_errors"].append(f"并发任务异常: {e}")

    except ImportError:
        realtime_data["fetch_errors"].append("AKShare 未安装，将使用 LLM 生成数据")
    except Exception as e:
        realtime_data["fetch_errors"].append(f"数据抓取整体异常: {e}")

    return realtime_data


def _format_realtime_data_for_prompt(realtime_data: Dict[str, Any]) -> str:
    """将实时数据格式化为可注入 Prompt 的文本"""
    if not realtime_data:
        return "（未获取到实时数据，请基于自身知识分析）"

    parts = []

    if realtime_data.get("market_overview"):
        mo = realtime_data["market_overview"]
        parts.append(f"【A股市场概览】上涨 {mo.get('up_count', 'N/A')} 家，下跌 {mo.get('down_count', 'N/A')} 家，"
                      f"涨停 {mo.get('limit_up_count', 'N/A')} 家，跌停 {mo.get('limit_down_count', 'N/A')} 家，"
                      f"平均涨跌幅 {mo.get('avg_change_pct', 'N/A')}%，"
                      f"总成交额 {mo.get('total_turnover', 0)/1e8:.0f} 亿元")

    if realtime_data.get("indices"):
        indices_lines = []
        for idx in realtime_data["indices"]:
            if "error" not in idx:
                indices_lines.append(f"  {idx.get('name', '')}: {idx.get('price', 'N/A')} ({idx.get('change_pct', 'N/A')}%)")
        if indices_lines:
            parts.append("【主要指数】\n" + "\n".join(indices_lines))

    if realtime_data.get("sector"):
        sec = realtime_data["sector"]
        if "error" not in sec:
            parts.append(f"【板块行情 - {sec.get('sector', '')}】涨跌幅 {sec.get('change_pct', 'N/A')}%，"
                          f"上涨 {sec.get('up_count', 'N/A')} 家，下跌 {sec.get('down_count', 'N/A')} 家，"
                          f"成交额 {sec.get('turnover', 0)/1e8:.2f} 亿元")

    if realtime_data.get("sector_top_stocks"):
        stock_lines = []
        for s in realtime_data["sector_top_stocks"][:5]:
            stock_lines.append(f"  {s.get('name', '')}({s.get('symbol', '')}): {s.get('price', 'N/A')} ({s.get('change_pct', 'N/A')}%)")
        if stock_lines:
            parts.append("【板块成分股】\n" + "\n".join(stock_lines))

    if realtime_data.get("news"):
        news_lines = []
        for n in realtime_data["news"][:5]:
            news_lines.append(f"  [{n.get('publish_time', '')}] {n.get('title', '')} - {n.get('source', '')}")
        if news_lines:
            parts.append("【近期财经新闻】\n" + "\n".join(news_lines))

    if realtime_data.get("data_sources"):
        parts.append(f"【数据来源】{', '.join(realtime_data['data_sources'])} (数据获取时间: 实时)")

    if realtime_data.get("fetch_errors"):
        logger.info(f"数据获取部分失败(不注入Prompt): {realtime_data['fetch_errors']}")

    if not parts:
        return "（未获取到有效实时数据，请基于自身知识分析）"

    return "\n\n".join(parts)


# ==================== Prompt 模板 ====================

PERCEPTION_PROMPT = """你是一个专业的投资研究分析师，请基于以下真实市场数据进行分析:
研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}

以下是实时抓取的真实市场数据，请充分利用这些数据进行分析。对于未覆盖的领域，可结合你的专业知识补充分析:

{realtime_data_section}

请基于以上数据，从以下方面进行深度分析：
1.市场概况和最新动态（优先引用上述真实数据，数据未覆盖的领域可用专业知识补充）
2.关键经济和市场指标（基于真实行情数据提炼，缺少的数据用合理估算补充）
3.近期重要新闻解读（基于真实新闻分析其影响，缺少新闻时基于行业常识分析）
4.行业趋势分析（至少三个细分领域，结合已有数据分析）

重要：请专注于分析投资主题本身，不要评论数据获取情况。直接给出分析结论。

请严格按以下JSON格式输出:
{{
    "market_overview": "基于数据的市场概况分析",
    "key_indicators": {{"指标名": "指标值及数据来源"}},
    "recent_news": ["基于新闻的分析1", "基于新闻的分析2", "基于新闻的分析3"],
    "industry_trends": {{"细分领域": "趋势分析"}}
}}"""

MODELING_PROMPT = """你是资深投资策略师，请根据市场数据构建内部模型:
研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}
市场数据: {perception_data}

请构建全面的市场模型，包括：
1.当前市场状态评估
2.经济周期判断
3.主要风险因素（至少三个）
4.潜在机会领域（至少三个）
5.市场情绪分析

输出JSON格式:
{{
    "market_state": "市场状态",
    "economic_cycle": "经济周期",
    "risk_factors": ["风险1", "风险2", "风险3"],
    "opportunity_areas": ["机会1", "机会2", "机会3"],
    "market_sentiment": "市场情绪"
}}"""

REASONING_PROMPT = """你是战略投资顾问，请生成3个不同的投资方案:
研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}
市场模型: {world_model}

请为每个方案提供以下字段：
- 方案ID、投资假设、分析方法、预期结果
- 三维度评分（各0~1浮点数，按以下标准）：
  1. 数据支撑度(data_support_score)：假设是否有真实市场数据支撑（如板块实际涨跌、指数走势、新闻事件）
  2. 逻辑自洽性(logic_coherence_score)：从假设→分析方法→预期结果的论证链条是否完整一致，无矛盾
  3. 风险可控性(risk_controllability_score)：列出的风险因素是否有明确应对策略，而非仅描述风险
- 综合置信度(confidence_level) = 数据支撑度×0.4 + 逻辑自洽性×0.3 + 风险可控性×0.3
- 置信度评分理由(confidence_explanation)：简要说明三个维度评分依据
- 优势(至少3点)、劣势(至少3点)
方案应有明显差异，代表不同投资思路（如激进型、稳健型、防御型）。

输出JSON数组:
[
    {{"plan_id": "方案1", "hypothesis": "假设", "analysis_approach": "方法", "expected_outcome": "结果", "data_support_score": 0.8, "logic_coherence_score": 0.85, "risk_controllability_score": 0.6, "confidence_level": 0.76, "confidence_explanation": "数据支撑较高因板块实际上涨3.2%，逻辑完整但风险应对不够充分", "pros": ["优势1"], "cons": ["劣势1"]}},
    ...
]"""

DECISION_PROMPT = """你是投资决策委员会主席，请评估候选方案并选择最优:
研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}
市场模型: {world_model}
候选方案: {reasoning_plans}

每个候选方案包含以下评分维度（请重点关注）：
1. 三维度评分：数据支撑度(data_support_score)、逻辑自洽性(logic_coherence_score)、风险可控性(risk_controllability_score)
2. 综合置信度(confidence_level) = 数据支撑×40% + 逻辑自洽×30% + 风险可控×30%
3. 数据校验结果(data_validation)：方案假设与真实行情数据的吻合/矛盾情况
4. 多模型评估(cross_eval_results)：不同LLM模型独立评分的融合结果

选择最优方案时请综合考量：
- 优先选择数据支撑度高且与真实数据吻合的方案（data_validation无冲突）
- 综合置信度是重要参考但不唯一标准——论证深度和风险应对同样重要
- 注意data_validation中标注的矛盾，有矛盾的方案应谨慎选择
- 多模型评估结果一致的方案可信度更高

给出详细理由，特别是为什么选该方案而不选其他方案。

输出JSON格式:
{{ 
    "selected_plan_id": "选中方案ID",
    "investment_thesis": "投资论点",
    "supporting_evidence": ["证据1（引用真实数据）", "证据2"],
    "risk_assessment": "风险评估",
    "recommendation": "投资建议",
    "timeframe": "时间框架"
}}"""

REPORT_PROMPT = """你是专业投研报告撰写人，请生成完整投资报告:
研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}
市场数据: {perception_data}
市场模型: {world_model}
选定策略: {selected_plan}

请生成结构完整、逻辑清晰的专业报告，包括：
1.标题和摘要 2.市场背景（引用真实数据来源） 3.核心观点 4.分析论证 5.风险因素 6.投资建议 7.时间框架

重要：报告中的市场数据必须标注数据来源和获取时间，确保分析可追溯。
报告应专业、客观、有深度。"""


# ==================== LLM 工厂 ====================

def get_llm(api_key: str, api_url: str, model_name: str) -> Tongyi:
    """创建 LLM 实例

    注意：langchain-community 的 Tongyi 在部分版本会丢弃构造参数中的 dashscope_api_key，
    只从 DASHSCOPE_API_KEY 环境变量读取，因此这里显式写入环境变量与 dashscope 全局 key。
    """
    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key
        try:
            import dashscope
            dashscope.api_key = api_key
        except ImportError:
            pass
    return Tongyi(
        model_name=model_name,
        dashscope_api_key=api_key,
        base_url=api_url,
        model_kwargs={}
    )


# ==================== 指数退避 ====================

def exponential_backoff(retry_count: int, base_delay: float = BASE_RETRY_DELAY):
    """指数退避等待: 2^retry_count 秒（2s, 4s, 8s）"""
    delay = base_delay * (2 ** retry_count)
    logger.info(f"指数退避等待 {delay}秒 (重试第{retry_count + 1}次)")
    time.sleep(delay)


# ==================== LangGraph 节点函数 ====================

# 阶段执行顺序（用于回滚时确定上一阶段）
# v2: 在 decision 与 report 之间插入 reflect 阶段
PHASE_ORDER = ["perception", "modeling", "reasoning", "decision", "reflect", "report"]


def _get_previous_phase(current_phase: str) -> Optional[str]:
    """获取当前阶段的前一个阶段"""
    idx = PHASE_ORDER.index(current_phase) if current_phase in PHASE_ORDER else -1
    if idx > 0:
        return PHASE_ORDER[idx - 1]
    return None


def _record_chain_llm(state: Dict[str, Any], purpose: str, model: str,
                      input_text: str, output_text: str, started_at: float):
    """记录一次 langchain 链式 LLM 调用到 Trace（token 为 tiktoken 估算值）

    langchain Tongyi 链式调用不暴露 usage，用 tiktoken 精确计数做估算，
    保证"token 成本可度量"（Agentic 路径则使用 API 返回的真实 usage）。
    """
    tracer = state.get("_tracer")
    if not tracer:
        return
    try:
        from .memory.context import count_tokens
        tracer.record_llm(
            purpose=purpose, model=model,
            prompt_tokens=count_tokens(input_text),
            completion_tokens=count_tokens(output_text),
            latency_ms=int((time.time() - started_at) * 1000),
            input_preview=input_text, output_preview=output_text,
        )
    except Exception as e:
        logger.debug(f"Trace LLM 记录失败（不影响流程）: {e}")


def _get_memory(state: Dict[str, Any]):
    """构建长期记忆实例（每次构建轻量，VectorStore 有文件缓存）"""
    from .memory.memory import LongTermMemory
    from .memory.embeddings import EmbeddingProvider
    return LongTermMemory(embedding=EmbeddingProvider(
        api_key=state.get("api_key", ""), base_url=state.get("api_url", "")))


def _recall_memory_context(state: Dict[str, Any]) -> str:
    """召回长期记忆，格式化为可注入 Prompt 的上下文"""
    try:
        memory = _get_memory(state)
        query = f"{state.get('research_topic','')} {state.get('industry_focus','')}"
        return memory.recall_as_context(query, top_k=3, industry=state.get("industry_focus"))
    except Exception as e:
        logger.warning(f"长期记忆召回失败（不阻塞流程）: {e}")
        return ""


# ==================== Agentic 感知（工具调用循环） ====================

AGENTIC_PERCEPTION_SYSTEM = """你是一个专业的投资研究分析师，具备自主规划与工具使用能力。

你的任务：围绕研究主题自主收集真实市场数据。你可以使用以下工具，每次调用一个，根据返回结果决定下一步：
- 先用 get_market_overview / get_major_indices 判断大盘环境
- 用 get_sector_performance / get_sector_top_stocks 定位行业与标的
- 用 get_financial_news / get_stock_news 收集事件与舆情
- 需要时用 search_stock 把公司名/概念解析为股票代码，再查 get_stock_quote / get_stock_financial

工作原则：
1. 每次只调用最有价值的一个工具，不要重复调用相同参数
2. 工具失败时，根据错误提示换参数或换工具
3. 收集到足够证据（通常 3-6 次调用）后停止调用，直接输出结构化分析
4. 最终输出必须是严格 JSON（不要调用工具，直接输出文本）：
{{
    "market_overview": "基于真实数据的市场概况",
    "key_indicators": {{"指标名": "指标值及来源"}},
    "recent_news": ["新闻分析1", "新闻分析2"],
    "industry_trends": {{"细分领域": "趋势分析"}}
}}"""


def _perception_agentic(state: Dict[str, Any]) -> Dict[str, Any]:
    """Agentic 感知：模型自主规划并调用工具收集数据（Function Calling 循环）"""
    from .llm_client import ModelRouter, ModelRoute
    from .agent_loop import run_tool_calling_loop
    from .skills import skill_loader
    from .tools import market_tools  # noqa: F401  触发工具注册
    from .tracer import tool_metrics as _tm
    from .config import get_agent_settings

    # 多模型路由：重推理走 heavy 路由（主模型 → 降级模型），
    # 单模型配额耗尽(403)时自动沿降级链切换，不再直接导致任务失败
    settings = get_agent_settings()
    heavy_chain = [state["model_name"]]
    if settings.FALLBACK_MODEL and settings.FALLBACK_MODEL != state["model_name"]:
        heavy_chain.append(settings.FALLBACK_MODEL)
    llm = ModelRouter(
        state["api_key"], state["api_url"],
        routes={"heavy": ModelRoute("heavy", heavy_chain)},
        default_route="heavy", timeout=120,
    )

    # 注入 Skill 方法论与长期记忆
    task_text = f"{state.get('research_topic','')} {state.get('industry_focus','')}"
    skill = skill_loader.select(task_text)
    system_prompt = AGENTIC_PERCEPTION_SYSTEM + (skill_loader.render(skill) if skill else "")

    memory_context = _recall_memory_context(state)
    user_prompt = (
        f"研究主题: {state['research_topic']}\n"
        f"行业焦点: {state['industry_focus']}\n"
        f"时间范围: {state['time_horizon']}\n\n"
        f"{memory_context}\n\n"
        f"请开始收集数据并完成分析。"
    )

    tracer = state.get("_tracer")

    # 工具执行通道：默认进程内执行（低时延）；配置 USE_MCP_EXECUTOR 后
    # 经 MCP Server（stdio 子进程）执行，实现工具与 Agent 进程级解耦
    mcp_executor = None
    if settings.USE_MCP_EXECUTOR:
        try:
            from .tools.mcp_client import mcp_client
            if mcp_client.ping():
                mcp_executor = mcp_client.call_tool
                logger.info("工具执行通道: MCP Server (stdio)")
        except Exception as e:
            logger.warning(f"MCP 执行通道不可用，回退进程内执行: {e}")

    loop_result = run_tool_calling_loop(
        llm, system_prompt, user_prompt, max_rounds=8,
        executor=mcp_executor,
        on_tool_call=lambda rec: (
            _tm.record(rec.tool_name, rec.ok, rec.latency_ms, rec.error_type),
            tracer.record_tool(rec.to_dict()) if tracer else None,
        ),
    )

    if tracer:
        tracer.record_llm(
            purpose="perception_agentic",
            model=llm.last_used_model or state["model_name"],
            prompt_tokens=loop_result.total_prompt_tokens,
            completion_tokens=loop_result.total_completion_tokens,
            latency_ms=loop_result.total_latency_ms,
            input_preview=user_prompt, output_preview=loop_result.final_text,
        )

    # 关键护栏：LLM 调用彻底失败（无输出且无成功工具调用）时必须抛错，
    # 让感知节点走重试/回退逻辑，而不是静默返回空数据造成感知⇄建模死循环
    if not loop_result.final_text.strip() and not loop_result.succeeded_tools():
        raise RuntimeError(
            f"Agentic 感知 LLM 调用失败（stopped_reason={loop_result.stopped_reason}），"
            f"已重试 {loop_result.rounds} 轮无有效产出。请检查模型配额与 API Key。"
        )

    # 解析模型最终 JSON 输出
    result: Dict[str, Any] = {}
    try:
        text = loop_result.final_text.strip()
        # 提取 JSON 块
        if "```" in text:
            import re
            m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if m:
                text = m.group(1)
        result = json.loads(text)
    except Exception as e:
        logger.warning(f"Agentic 感知输出解析失败，回退为文本摘要: {e}")
        result = {"market_overview": loop_result.final_text[:2000]}

    result = validate_with_pydantic(PerceptionOutput, result)

    # 汇总工具证据作为 _realtime_data_raw（供推理阶段真实数据校验）
    evidence_raw: Dict[str, Any] = {}
    for rec in loop_result.records:
        if rec.ok:
            try:
                payload = json.loads(rec.content)
                data = payload.get("data", payload)
            except Exception:
                data = rec.content
            # 映射到既有校验所需的键
            if rec.tool_name == "get_market_overview":
                evidence_raw["market_overview"] = data
            elif rec.tool_name == "get_major_indices":
                evidence_raw["indices"] = data
            elif rec.tool_name == "get_sector_performance":
                evidence_raw["sector"] = data

    result["_realtime_data_sources"] = list(dict.fromkeys(loop_result.succeeded_tools()))
    result["_realtime_data_raw"] = evidence_raw
    result["_agentic"] = {
        "rounds": loop_result.rounds,
        "tool_calls": len(loop_result.records),
        "succeeded_tools": loop_result.succeeded_tools(),
        "stopped_reason": loop_result.stopped_reason,
        "skill": skill.name if skill else None,
    }
    return result


def perception_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """感知阶段 - 收集市场数据（支持 Agentic 工具调用循环 / 固定抓取两种模式）"""
    logger.info("阶段1: 感知 - 收集市场数据")
    retry_count = state.get("perception_retry", 0)

    if retry_count >= MAX_RETRIES:
        logger.error(f"感知阶段重试超过{MAX_RETRIES}次")
        return {**state, "error": f"感知阶段重试超过{MAX_RETRIES}次", "current_phase": "error"}

    # 指数退避
    if retry_count > 0:
        exponential_backoff(retry_count - 1)

    try:
        use_realtime = state.get("use_realtime_data", True)
        use_agentic = state.get("use_agentic_perception", True) and use_realtime
        realtime_data_section = ""
        realtime_data_raw = None

        if use_agentic:
            # Agentic 模式：模型自主选择工具收集数据
            logger.info("使用 Agentic 感知（工具调用循环）...")
            try:
                result = _perception_agentic(state)
            except Exception as e:
                logger.warning(f"Agentic 感知失败，回退到固定抓取: {e}")
                result = None
                use_agentic = False
        else:
            result = None

        if not use_agentic or result is None:
            # 固定模式：并发抓取预定义数据集
            if use_realtime:
                logger.info("正在从 AKShare 抓取实时市场数据...")
                try:
                    realtime_data_raw = _fetch_realtime_data(
                        topic=state["research_topic"],
                        industry=state["industry_focus"]
                    )
                    realtime_data_section = _format_realtime_data_for_prompt(realtime_data_raw)
                    logger.info(f"实时数据抓取完成，数据源: {realtime_data_raw.get('data_sources', [])}")
                except Exception as e:
                    logger.warning(f"实时数据抓取失败，将回退到 LLM 生成: {e}")
                    realtime_data_section = "（实时数据获取失败，请基于自身知识分析当前市场状况）"

            llm = get_llm(state["api_key"], state["api_url"], state["model_name"])
            prompt = ChatPromptTemplate.from_template(PERCEPTION_PROMPT)
            chain = prompt | llm | JsonOutputParser()
            result = chain.invoke({
                "research_topic": state["research_topic"],
                "industry_focus": state["industry_focus"],
                "time_horizon": state["time_horizon"],
                "realtime_data_section": realtime_data_section
            })

            # Pydantic 校验
            result = validate_with_pydantic(PerceptionOutput, result)

            if realtime_data_raw:
                result["_realtime_data_sources"] = realtime_data_raw.get("data_sources", [])
                result["_realtime_fetch_errors"] = realtime_data_raw.get("fetch_errors", [])
                result["_realtime_data_raw"] = realtime_data_raw

        return {
            **state,
            "perception_data": result,
            "current_phase": "modeling",
            "perception_retry": 0,
            "error": None,
            "last_completed_phase": "perception",
            "completed_phases": state.get("completed_phases", []) + ["perception"],
        }
    except Exception as e:
        logger.error(f"感知阶段错误: {e}")
        return {**state, "error": f"感知阶段出错: {e}", "current_phase": "perception", "perception_retry": retry_count + 1}


def modeling_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """建模阶段 - 构建内部模型"""
    logger.info("阶段2: 建模 - 构建内部模型")
    retry_count = state.get("modeling_retry", 0)

    if retry_count >= MAX_RETRIES:
        logger.error(f"建模阶段重试超过{MAX_RETRIES}次，回滚到感知阶段")
        # 回滚：清空感知数据，重新执行感知阶段
        return {
            **state,
            "error": None,
            "perception_data": None,
            "current_phase": "perception",
            "perception_retry": state.get("perception_retry", 0),  # 保留感知的重试计数
            "modeling_retry": 0,
        }

    if retry_count > 0:
        exponential_backoff(retry_count - 1)

    try:
        if not state.get("perception_data"):
            logger.warning("建模阶段缺少感知数据，回退到感知阶段")
            return {**state, "error": None, "current_phase": "perception", "modeling_retry": retry_count + 1}

        llm = get_llm(state["api_key"], state["api_url"], state["model_name"])
        prompt = ChatPromptTemplate.from_template(MODELING_PROMPT)
        chain = prompt | llm | JsonOutputParser()
        _t0 = time.time()
        result = chain.invoke({
            "research_topic": state["research_topic"],
            "industry_focus": state["industry_focus"],
            "time_horizon": state["time_horizon"],
            "perception_data": json.dumps(state["perception_data"], ensure_ascii=False)
        })

        result = validate_with_pydantic(ModelingOutput, result)
        _record_chain_llm(state, "modeling", state["model_name"],
                          MODELING_PROMPT, json.dumps(result, ensure_ascii=False, default=str), _t0)

        return {
            **state,
            "world_model": result,
            "current_phase": "reasoning",
            "modeling_retry": 0,
            "error": None,
            "last_completed_phase": "modeling",
            "completed_phases": state.get("completed_phases", []) + ["modeling"],
        }
    except Exception as e:
        logger.error(f"建模阶段错误: {e}")
        return {**state, "error": f"建模阶段出错: {e}", "current_phase": "modeling", "modeling_retry": retry_count + 1}


def reasoning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """推理阶段 - 生成候选方案"""
    logger.info("阶段3: 推理 - 生成候选方案")
    retry_count = state.get("reasoning_retry", 0)

    if retry_count >= MAX_RETRIES:
        logger.error(f"推理阶段重试超过{MAX_RETRIES}次，回滚到建模阶段")
        return {
            **state,
            "error": None,
            "world_model": None,
            "current_phase": "modeling",
            "modeling_retry": state.get("modeling_retry", 0),
            "reasoning_retry": 0,
        }

    if retry_count > 0:
        exponential_backoff(retry_count - 1)

    try:
        if not state.get("world_model"):
            logger.warning("推理阶段缺少世界模型，回退到建模阶段")
            return {**state, "error": None, "current_phase": "modeling", "reasoning_retry": retry_count + 1}

        llm = get_llm(state["api_key"], state["api_url"], state["model_name"])

        # 反思回灌闭环：Reflector 未通过时写回的修正意见（_revision_hints）
        # 必须注入推理 Prompt，否则"反思"退化为"原样重试"
        revision_hints = state.get("_revision_hints") or []
        prompt_text = REASONING_PROMPT
        if revision_hints:
            hints_block = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(revision_hints))
            prompt_text += (
                "\n\n【独立风控审核修正意见（最高优先级）】\n"
                "上一轮生成的方案未通过独立风控审核，必须针对以下问题逐条修正后重新生成：\n"
                f"{hints_block}\n"
                "请确保新方案不再出现上述问题（如补充缺失的证据、下调与数据矛盾的置信度、完善风险应对策略）。"
            )
            logger.info(f"推理阶段携带 {len(revision_hints)} 条反思修正意见重新生成方案")

        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | llm | JsonOutputParser()
        _t0 = time.time()
        result = chain.invoke({
            "research_topic": state["research_topic"],
            "industry_focus": state["industry_focus"],
            "time_horizon": state["time_horizon"],
            "world_model": json.dumps(state["world_model"], ensure_ascii=False)
        })
        _record_chain_llm(state, "reasoning", state["model_name"],
                          prompt_text, json.dumps(result, ensure_ascii=False, default=str), _t0)

        # 推理阶段返回列表，逐项校验
        if isinstance(result, list):
            result = validate_with_pydantic(ReasoningPlanOutput, result)
        else:
            # LLM 可能返回包含列表的 dict
            if isinstance(result, dict) and any(isinstance(v, list) for v in result.values()):
                for k, v in result.items():
                    if isinstance(v, list):
                        result = v
                        break
            result = validate_with_pydantic(ReasoningPlanOutput, result)

        # ========== 改进二：真实数据校验 ==========
        # 用感知阶段的真实行情数据校验方案假设与实际数据的吻合度，自动调整置信度
        perception_data = state.get("perception_data")
        if perception_data and isinstance(result, list):
            try:
                logger.info("开始真实数据校验：对比方案假设与感知阶段真实行情数据...")
                result = validate_plans_with_real_data(result, perception_data)
            except Exception as e:
                logger.warning(f"真实数据校验失败（不阻塞流程）: {e}")

        # ========== 改进三：多模型交叉评估 ==========
        # 让辅助模型独立评分，与主模型评分加权融合，减少单一模型偏差
        if isinstance(result, list):
            try:
                logger.info("开始多模型交叉评估...")
                result = multi_model_score(
                    result,
                    api_key=state["api_key"],
                    api_url=state["api_url"],
                    primary_model=state["model_name"]
                )
            except Exception as e:
                logger.warning(f"多模型评估失败（不阻塞流程）: {e}")

        return {
            **state,
            "reasoning_plans": result,
            "current_phase": "decision",
            "reasoning_retry": 0,
            "error": None,
            "_revision_hints": [],  # 修正意见已消费，清空避免残留
            "last_completed_phase": "reasoning",
            "completed_phases": state.get("completed_phases", []) + ["reasoning"],
        }
    except Exception as e:
        logger.error(f"推理阶段错误: {e}")
        return {**state, "error": f"推理阶段出错: {e}", "current_phase": "reasoning", "reasoning_retry": retry_count + 1}


def decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """决策阶段 - 选择最优方案"""
    logger.info("阶段4: 决策 - 选择最优方案")
    retry_count = state.get("decision_retry", 0)

    if retry_count >= MAX_RETRIES:
        logger.error(f"决策阶段重试超过{MAX_RETRIES}次，回滚到推理阶段")
        return {
            **state,
            "error": None,
            "reasoning_plans": None,
            "current_phase": "reasoning",
            "reasoning_retry": state.get("reasoning_retry", 0),
            "decision_retry": 0,
        }

    if retry_count > 0:
        exponential_backoff(retry_count - 1)

    try:
        if not state.get("reasoning_plans"):
            logger.warning("决策阶段缺少候选方案，回退到推理阶段")
            return {**state, "error": None, "current_phase": "reasoning", "decision_retry": retry_count + 1}

        llm = get_llm(state["api_key"], state["api_url"], state["model_name"])
        prompt = ChatPromptTemplate.from_template(DECISION_PROMPT)
        chain = prompt | llm | JsonOutputParser()
        _t0 = time.time()
        result = chain.invoke({
            "research_topic": state["research_topic"],
            "industry_focus": state["industry_focus"],
            "time_horizon": state["time_horizon"],
            "world_model": json.dumps(state["world_model"], ensure_ascii=False),
            "reasoning_plans": json.dumps(state["reasoning_plans"], ensure_ascii=False)
        })

        result = validate_with_pydantic(DecisionOutput, result)
        _record_chain_llm(state, "decision", state["model_name"],
                          DECISION_PROMPT, json.dumps(result, ensure_ascii=False, default=str), _t0)

        return {
            **state,
            "selected_plan": result,
            "current_phase": "reflect",
            "decision_retry": 0,
            "error": None,
            "last_completed_phase": "decision",
            "completed_phases": state.get("completed_phases", []) + ["decision"],
        }
    except Exception as e:
        logger.error(f"决策阶段错误: {e}")
        return {**state, "error": f"决策阶段出错: {e}", "current_phase": "decision", "decision_retry": retry_count + 1}


def reflect_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """反思阶段 - Reflector 独立审核决策，产出修正意见回灌 + 经验教训写回记忆

    与"原样重试"的区别：反思是带着诊断结论改策略再来一遍。
    - 通过 → 进入报告阶段
    - 不通过且未超反思次数 → 回滚到推理阶段（附 revision_hints）
    - 不通过但已超反思次数 → 记录问题但仍进入报告（避免无限循环）
    """
    from .reflector import reflect_on_decision, MAX_REFLECTION_LOOPS
    from .config import get_agent_settings

    logger.info("阶段5: 反思 - Reflector 审核决策")
    reflection_count = state.get("reflection_count", 0)
    tracer = state.get("_tracer")

    try:
        # 分析-审核双角色结构：审核官使用独立模型（与主分析模型解耦），
        # 避免同一模型"既当运动员又当裁判"的自评偏差
        reviewer_model = get_agent_settings().REVIEWER_MODEL or state["model_name"]
        review_state = {**state, "model_name": reviewer_model}
        _t0 = time.time()
        reflection = reflect_on_decision(review_state, get_llm)
        reflection["reviewer_model"] = reviewer_model

        if tracer:
            from .memory.context import count_tokens
            _in = json.dumps(state.get("selected_plan"), ensure_ascii=False, default=str)
            _out = json.dumps(reflection, ensure_ascii=False, default=str)
            tracer.record_llm(
                purpose="reflect", model=reviewer_model,
                prompt_tokens=count_tokens(_in), completion_tokens=count_tokens(_out),
                latency_ms=int((time.time() - _t0) * 1000),
                input_preview=_in, output_preview=_out,
            )

        passed = reflection.get("passed", True)
        score = reflection.get("score", 0.7)
        logger.info(f"Reflector 审核: passed={passed}, score={score}, issues={len(reflection.get('issues', []))}")

        # 经验教训写入长期记忆（自进化闭环：无论是否通过都沉淀）
        lessons = reflection.get("lessons") or []
        if lessons:
            try:
                memory = _get_memory(state)
                for lesson in lessons:
                    memory.write_reflection(
                        topic=state.get("research_topic", ""),
                        industry=state.get("industry_focus", ""),
                        lesson=lesson,
                        confidence=min(0.9, score),
                    )
            except Exception as e:
                logger.warning(f"经验教训写入长期记忆失败（不阻塞流程）: {e}")

        # 不通过且还有反思次数 → 回滚到推理阶段，附修正意见
        if not passed and reflection_count < MAX_REFLECTION_LOOPS:
            logger.warning(f"Reflector 未通过，回滚到推理阶段（第 {reflection_count + 1} 次反思）: "
                           f"{reflection.get('revision_hints')}")
            return {
                **state,
                "reflection": reflection,
                "reflection_count": reflection_count + 1,
                "reasoning_plans": None,
                "selected_plan": None,
                "current_phase": "reasoning",
                "reasoning_retry": 0,
                "error": None,
                "_revision_hints": reflection.get("revision_hints", []),
            }

        # 通过 或 已超反思次数 → 进入报告阶段
        if not passed:
            logger.warning(f"Reflector 已达最大反思次数({MAX_REFLECTION_LOOPS})，仍进入报告阶段，问题已记录")

        return {
            **state,
            "reflection": reflection,
            "current_phase": "report",
            "error": None,
            "last_completed_phase": "reflect",
            "completed_phases": state.get("completed_phases", []) + ["reflect"],
        }
    except Exception as e:
        logger.error(f"反思阶段错误（不阻塞流程，直接进入报告）: {e}")
        # 反思失败不阻塞主流程
        return {
            **state,
            "reflection": {"passed": True, "score": 0.5, "issues": [], "lessons": [],
                           "note": f"反思阶段异常，跳过审核: {e}"},
            "current_phase": "report",
            "error": None,
        }


def report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """报告阶段 - 生成完整报告"""
    logger.info("阶段5: 报告 - 生成完整报告")
    retry_count = state.get("report_retry", 0)

    if retry_count >= MAX_RETRIES:
        logger.error(f"报告阶段重试超过{MAX_RETRIES}次，回滚到决策阶段")
        return {
            **state,
            "error": None,
            "selected_plan": None,
            "current_phase": "decision",
            "decision_retry": state.get("decision_retry", 0),
            "report_retry": 0,
        }

    if retry_count > 0:
        exponential_backoff(retry_count - 1)

    try:
        if not state.get("selected_plan"):
            logger.warning("报告阶段缺少选定方案，回退到决策阶段")
            return {**state, "error": None, "current_phase": "decision", "report_retry": retry_count + 1}

        llm = get_llm(state["api_key"], state["api_url"], state["model_name"])
        prompt = ChatPromptTemplate.from_template(REPORT_PROMPT)
        chain = prompt | llm | StrOutputParser()
        _t0 = time.time()
        result = chain.invoke({
            "research_topic": state["research_topic"],
            "industry_focus": state["industry_focus"],
            "time_horizon": state["time_horizon"],
            "perception_data": json.dumps(state["perception_data"], ensure_ascii=False),
            "world_model": json.dumps(state["world_model"], ensure_ascii=False),
            "selected_plan": json.dumps(state["selected_plan"], ensure_ascii=False)
        })

        # 任务结论写入长期记忆（供下次同类任务召回）
        try:
            memory = _get_memory(state)
            selected = state.get("selected_plan") or {}
            conclusion = (
                f"主题[{state['research_topic']}] 行业[{state['industry_focus']}] "
                f"结论: {selected.get('recommendation', '')} | "
                f"论点: {selected.get('investment_thesis', '')[:200]}"
            )
            # DecisionOutput 没有 confidence_level 字段，
            # 从推理方案中按 selected_plan_id 反查真实置信度
            selected_confidence = 0.5
            for p in state.get("reasoning_plans") or []:
                if p.get("plan_id") == selected.get("selected_plan_id"):
                    selected_confidence = float(p.get("confidence_level", 0.5) or 0.5)
                    break
            memory.write_conclusion(
                topic=state["research_topic"],
                industry=state["industry_focus"],
                conclusion=conclusion,
                confidence=selected_confidence,
            )
        except Exception as e:
            logger.warning(f"任务结论写入长期记忆失败（不阻塞流程）: {e}")

        _record_chain_llm(state, "report", state["model_name"],
                          REPORT_PROMPT, result, _t0)

        return {
            **state,
            "final_report": result,
            "current_phase": "completed",
            "report_retry": 0,
            "error": None,
            "last_completed_phase": "report",
            "completed_phases": state.get("completed_phases", []) + ["report"],
        }
    except Exception as e:
        logger.error(f"报告阶段错误: {e}")
        return {**state, "error": f"报告阶段出错: {e}", "current_phase": "report", "report_retry": retry_count + 1}


# ==================== 条件路由（状态机） ====================

def router(state: Dict[str, Any]) -> str:
    """根据 current_phase 决定下一步路由

    状态机逻辑：
    - 成功: current_phase 指向下一阶段 → 路由到该阶段
    - 失败重试: current_phase 不变 → 路由回当前阶段重试
    - 超过重试次数回滚: current_phase 指向上一阶段 → 路由到上一阶段
    - 完成: current_phase == "completed" → 结束
    - 错误: current_phase == "error" → 结束
    """
    current = state.get("current_phase", "perception")

    # 终止状态
    if current == "completed":
        return "completed"
    if current == "error":
        return "error"

    # 正常路由：current_phase 就是下一个要执行的阶段
    return current


def get_conditional_mapping():
    """获取各阶段的条件边映射

    每个阶段执行后，router 返回的值决定路由：
    - "modeling"/"reasoning" 等 → 正常推进
    - "perception"/"modeling" 等 → 回滚重试
    - "completed" → 结束
    - "error" → 结束
    """
    return {
        "perception": {
            "modeling": "modeling",     # 成功 → 下一阶段
            "perception": "perception",  # 重试当前阶段
            "error": END,               # 超过重试次数 → 结束
            "completed": END,
        },
        "modeling": {
            "reasoning": "reasoning",   # 成功 → 下一阶段
            "modeling": "modeling",     # 重试当前阶段
            "perception": "perception", # 回滚到上一阶段
            "error": END,
            "completed": END,
        },
        "reasoning": {
            "decision": "decision",     # 成功 → 下一阶段
            "reasoning": "reasoning",   # 重试当前阶段
            "modeling": "modeling",     # 回滚到上一阶段
            "error": END,
            "completed": END,
        },
        "decision": {
            "reflect": "reflect",       # 成功 → 反思阶段
            "decision": "decision",     # 重试当前阶段
            "reasoning": "reasoning",   # 回滚到上一阶段
            "error": END,
            "completed": END,
        },
        "reflect": {
            "report": "report",         # 审核通过 → 报告阶段
            "reasoning": "reasoning",   # 审核未通过 → 回滚到推理阶段（带修正意见）
            "reflect": "reflect",       # 重试当前阶段
            "decision": "decision",     # 回滚到决策阶段
            "error": END,
            "completed": END,
        },
        "report": {
            "completed": END,           # 成功 → 结束
            "report": "report",         # 重试当前阶段
            "decision": "decision",     # 回滚到决策阶段
            "error": END,
        },
    }


# ==================== LangGraph 工作流构建 ====================

def create_research_workflow():
    """创建 LangGraph 工作流（状态机 + 条件边 + 自动回滚）

    入口使用条件路由：根据 state["current_phase"] 决定首个执行节点，
    使检查点恢复的任务能直接从中断阶段续跑，而非重头开始。
    """
    workflow = StateGraph(dict)

    # 添加节点
    workflow.add_node("perception", perception_node)
    workflow.add_node("modeling", modeling_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("report", report_node)

    # 条件入口：断点续跑时从 current_phase 指向的阶段直接进入
    entry_mapping = {phase: phase for phase in PHASE_ORDER}
    entry_mapping["completed"] = END
    entry_mapping["error"] = END
    workflow.set_conditional_entry_point(router, entry_mapping)

    # 添加条件边（状态机路由）
    mapping = get_conditional_mapping()
    for phase, phase_mapping in mapping.items():
        workflow.add_conditional_edges(phase, router, phase_mapping)

    return workflow.compile()


# ==================== 主入口函数 ====================

def run_research_agent(
    topic: str,
    industry: str,
    horizon: str,
    api_key: str,
    api_url: str,
    model_name: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    use_realtime_data: bool = True,
    use_agentic_perception: bool = True,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """运行完整分析流程（LangGraph 图驱动 + 检查点持久化 + 断点续跑 + 全链路 Trace）

    Args:
        topic: 研究主题
        industry: 行业焦点
        horizon: 时间范围
        api_key: DashScope API Key
        api_url: API 地址
        model_name: 模型名称
        progress_callback: 进度回调函数
        use_realtime_data: 是否使用 AKShare 实时数据
        use_agentic_perception: 感知阶段是否用 Agentic 工具调用循环（默认开启，失败自动回退固定抓取）
        task_id: 任务ID（用于检查点持久化，不传则不持久化）

    Returns:
        最终状态字典
    """
    # 初始化 Trace（任务级轨迹记录）
    tracer = Tracer(task_id or f"ephemeral-{int(time.time())}")

    # 初始状态
    initial_state = {
        "research_topic": topic,
        "industry_focus": industry,
        "time_horizon": horizon,
        "api_key": api_key,
        "api_url": api_url,
        "model_name": model_name,
        "use_realtime_data": use_realtime_data,
        "use_agentic_perception": use_agentic_perception,
        "current_phase": "perception",
        "perception_data": None,
        "world_model": None,
        "reasoning_plans": None,
        "selected_plan": None,
        "reflection": None,
        "reflection_count": 0,
        "final_report": None,
        "error": None,
        "perception_retry": 0,
        "modeling_retry": 0,
        "reasoning_retry": 0,
        "decision_retry": 0,
        "reflect_retry": 0,
        "report_retry": 0,
        "last_completed_phase": None,
        "completed_phases": [],
        "_tracer": tracer,  # 非序列化字段，不进入检查点
    }

    # 断点续跑：尝试加载检查点
    if task_id:
        checkpoint = load_checkpoint(task_id)
        if checkpoint:
            logger.info(f"从检查点恢复任务 {task_id}，阶段: {checkpoint.get('current_phase')}")
            # 合并检查点数据到初始状态（保留已完成的中间结果）
            for key in ["perception_data", "world_model", "reasoning_plans", "selected_plan",
                        "current_phase", "perception_retry", "modeling_retry",
                        "reasoning_retry", "decision_retry", "report_retry",
                        "last_completed_phase", "completed_phases"]:
                if checkpoint.get(key) is not None:
                    initial_state[key] = checkpoint[key]

    # 创建并执行 LangGraph 工作流（图驱动：状态机路由 + 回滚 + 重试全部由图承担）
    graph = create_research_workflow()
    state = initial_state
    # 递归上限即原 while 循环的安全阀（最大迭代 30 → 留出反射/回滚余量）
    max_iterations = 30
    last_tick = time.time()

    try:
        # stream_mode="updates"：每执行完一个节点产出 {节点名: 状态增量}
        for update in graph.stream(
            state,
            config={"recursion_limit": max_iterations * 2},
            stream_mode="updates",
        ):
            for node_name, node_state in update.items():
                tick = time.time()
                logger.info(f"图节点执行完成: {node_name}")

                # 合并节点输出到主状态
                state.update(node_state)

                # 通知进度
                if progress_callback:
                    progress_callback(node_name)

                # Trace：节点内的 record_llm/record_tool 已进入挂起缓冲，
                # start_step 时冲刷归属到本步骤；started_at 对齐节点真实开始时刻
                ts = tracer.start_step(node_name)
                ts.started_at = last_tick
                tracer.end_step(
                    status="error" if state.get("current_phase") == "error" else "ok",
                    error=state.get("error"),
                    output_summary=f"下一阶段: {state.get('current_phase')}",
                )
                last_tick = tick

                # 每个阶段执行后保存检查点
                if task_id:
                    save_checkpoint(task_id, state)

                # 回滚日志（current_phase 指向了更早的阶段）
                new_phase = state.get("current_phase", "perception")
                if new_phase in PHASE_ORDER and node_name in PHASE_ORDER:
                    if PHASE_ORDER.index(new_phase) < PHASE_ORDER.index(node_name):
                        logger.warning(f"回滚: {node_name} → {new_phase}，保留已完成的中间数据")
    except Exception as e:
        # 图递归上限（重试/回滚环路耗尽）或其他图执行异常
        if "recursion" in str(e).lower():
            logger.error(f"工作流达到递归上限({max_iterations * 2})，判定为反复失败: {e}")
        else:
            logger.error(f"LangGraph 图执行异常: {e}")
            raise

    # 兜底护栏：工作流未正常完成时必须写入明确错误信息，
    # 否则前端拿不到失败原因（error=None 导致界面无提示）
    if state.get("current_phase") not in ("completed", "error"):
        failed_phase = state.get("current_phase", "unknown")
        state["error"] = (
            f"工作流在阶段「{failed_phase}」反复失败/回滚，已达到最大迭代次数({max_iterations})未完成。"
            f"常见原因：模型配额耗尽(403，请更换模型)、API Key 无效、数据源超时。"
        )
        state["current_phase"] = "error"
        logger.error(f"工作流兜底触发: {state['error']}")

    # 通知最终状态
    if progress_callback:
        progress_callback(state.get("current_phase", "error"))

    # 任务完成后清理检查点
    if task_id and state.get("current_phase") == "completed":
        delete_checkpoint(task_id)

    # 落盘执行轨迹（Trace）
    try:
        trace_path = tracer.save()
        state["trace_path"] = trace_path
        state["trace_summary"] = tracer.summary()
    except Exception as e:
        logger.warning(f"Trace 保存失败: {e}")

    # 移除非序列化的 tracer 引用，避免下游 json.dumps(state) 报错
    state.pop("_tracer", None)

    return state


def resume_pending_tasks(progress_callback: Optional[Callable] = None) -> List[str]:
    """恢复所有未完成的任务（服务重启后调用）

    Args:
        progress_callback: 进度回调函数

    Returns:
        恢复的任务ID列表
    """
    pending = list_pending_checkpoints()
    resumed = []

    for task_id in pending:
        checkpoint = load_checkpoint(task_id)
        if not checkpoint:
            continue

        logger.info(f"恢复未完成任务: {task_id}")
        try:
            result = run_research_agent(
                topic=checkpoint["research_topic"],
                industry=checkpoint["industry_focus"],
                horizon=checkpoint["time_horizon"],
                api_key=checkpoint["api_key"],
                api_url=checkpoint["api_url"],
                model_name=checkpoint["model_name"],
                progress_callback=progress_callback,
                use_realtime_data=checkpoint.get("use_realtime_data", True),
                use_agentic_perception=checkpoint.get("use_agentic_perception", True),
                task_id=task_id,
            )
            resumed.append(task_id)
            logger.info(f"任务 {task_id} 恢复完成，状态: {result.get('current_phase')}")
        except Exception as e:
            logger.error(f"恢复任务 {task_id} 失败: {e}")

    return resumed
