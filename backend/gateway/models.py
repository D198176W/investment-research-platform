"""Pydantic 数据模型定义"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


# ==================== 请求模型 ====================

class ResearchRequest(BaseModel):
    """研究分析请求"""
    topic: str = Field(..., description="研究主题", examples=["人工智能芯片市场"])
    industry: str = Field(..., description="行业焦点", examples=["半导体"])
    horizon: str = Field("中期(6-12个月)", description="时间范围", examples=["中期(6-12个月)"])
    model_name: Optional[str] = Field(None, description="模型名称（可选，默认使用配置中的模型）")
    use_realtime_data: bool = Field(True, description="是否使用 AKShare 实时数据注入分析")
    use_agentic_perception: bool = Field(True, description="感知阶段是否启用 Agentic 工具调用循环（默认开启，失败自动回退固定抓取）")
    stock_symbols: Optional[List[str]] = Field(None, description="关联股票代码列表，如 ['000001', '600519']")


class HistoryRequest(BaseModel):
    """历史记录请求"""
    limit: int = Field(20, ge=1, le=100, description="返回记录数量限制")


# ==================== 响应模型 ====================

class PerceptionResponse(BaseModel):
    """感知阶段响应"""
    market_overview: str = Field(..., description="市场概况")
    key_indicators: Dict[str, str] = Field(..., description="关键指标")
    recent_news: List[str] = Field(..., description="近期新闻")
    industry_trends: Dict[str, str] = Field(..., description="行业趋势")


class ModelingResponse(BaseModel):
    """建模阶段响应"""
    market_state: str = Field(..., description="市场状态")
    economic_cycle: str = Field(..., description="经济周期")
    risk_factors: List[str] = Field(..., description="风险因素")
    opportunity_areas: List[str] = Field(..., description="机会领域")
    market_sentiment: str = Field(..., description="市场情绪")


class ReasoningPlanResponse(BaseModel):
    """推理阶段方案"""
    plan_id: str = Field(..., description="方案ID")
    hypothesis: str = Field(..., description="投资假设")
    analysis_approach: str = Field(..., description="分析方法")
    expected_outcome: str = Field(..., description="预期结果")
    confidence_level: float = Field(..., description="置信度(0~1)")
    pros: List[str] = Field(..., description="优势")
    cons: List[str] = Field(..., description="劣势")


class DecisionResponse(BaseModel):
    """决策阶段响应"""
    selected_plan_id: str = Field(..., description="选中方案")
    investment_thesis: str = Field(..., description="投资论点")
    supporting_evidence: List[str] = Field(..., description="支持证据")
    risk_assessment: str = Field(..., description="风险评估")
    recommendation: str = Field(..., description="投资建议")
    timeframe: str = Field(..., description="时间框架")


class ResearchProgressResponse(BaseModel):
    """研究进度响应"""
    task_id: str
    status: str  # pending, running, completed, failed
    current_phase: Optional[str] = None
    progress_percent: int = Field(0, ge=0, le=100)
    perception_data: Optional[Dict[str, Any]] = None
    world_model: Optional[Dict[str, Any]] = None
    reasoning_plans: Optional[List[Dict[str, Any]]] = None
    selected_plan: Optional[Dict[str, Any]] = None
    reflection: Optional[Dict[str, Any]] = None  # v2: Reflector 审核结果
    trace_summary: Optional[Dict[str, Any]] = None  # v2: 执行轨迹概要（失败排查用）
    final_report: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class ResearchCompleteResponse(BaseModel):
    """研究完成响应"""
    task_id: str
    topic: str
    industry: str
    horizon: str
    perception_data: Optional[Dict[str, Any]] = None
    world_model: Optional[Dict[str, Any]] = None
    reasoning_plans: Optional[List[Dict[str, Any]]] = None
    selected_plan: Optional[Dict[str, Any]] = None
    final_report: Optional[str] = None
    completed_at: Optional[str] = None


class HistoryRecordResponse(BaseModel):
    """历史记录响应"""
    id: str
    time: str
    topic: str
    industry: str
    horizon: str
    model: str
    report_preview: str


class APIResponse(BaseModel):
    """通用 API 响应"""
    success: bool = True
    message: str = "success"
    data: Optional[Any] = None
    error: Optional[str] = None


# ==================== 配置模型 ====================

class LLMConfigRequest(BaseModel):
    """LLM 配置请求"""
    api_key: str = Field(..., description="DashScope API密钥")
    api_url: str = Field("https://dashscope.aliyuncs.com/compatible-mode/v1", description="API地址")
    model_name: str = Field("qwen-plus", description="模型名称")
