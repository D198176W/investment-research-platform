"""FastAPI 网关主应用"""
import uuid
import json
import os
import sys
import logging
import threading
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager

# 确保能找到 backend 下的其他服务模块
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from models import (
    ResearchRequest, ResearchProgressResponse, ResearchCompleteResponse,
    HistoryRecordResponse, APIResponse, LLMConfigRequest
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 历史记录文件（统一到项目根 data/ 下，便于容器化目录挂载；兼容旧根目录文件自动迁移）
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
HISTORY_FILE = os.path.join(_PROJECT_ROOT, "data", "analysis_history.json")
_LEGACY_HISTORY_FILE = os.path.join(_PROJECT_ROOT, "analysis_history.json")

# 存储运行中的任务（简单内存存储，生产环境建议使用Redis）
active_tasks = {}
tasks_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("智能投研平台 API Gateway 启动中...")

    # 断点续跑：恢复未完成的任务
    try:
        from agent_service.workflow import list_pending_checkpoints, load_checkpoint, run_research_agent
        pending = list_pending_checkpoints()
        if pending:
            logger.info(f"发现 {len(pending)} 个未完成任务，开始恢复...")
            for task_id in pending:
                checkpoint = load_checkpoint(task_id)
                if checkpoint:
                    # 将检查点恢复到 active_tasks
                    with tasks_lock:
                        active_tasks[task_id] = {
                            "task_id": task_id,
                            "research_topic": checkpoint.get("research_topic", ""),
                            "industry_focus": checkpoint.get("industry_focus", ""),
                            "time_horizon": checkpoint.get("time_horizon", ""),
                            "api_key": checkpoint.get("api_key", ""),
                            "api_url": checkpoint.get("api_url", ""),
                            "model_name": checkpoint.get("model_name", ""),
                            "use_realtime_data": checkpoint.get("use_realtime_data", True),
                            "status": "running",
                            "current_phase": checkpoint.get("current_phase", "perception"),
                            "created_at": checkpoint.get("saved_at", datetime.now().isoformat()),
                        }
                    # 在后台线程中恢复执行
                    thread = threading.Thread(target=run_research_analysis, args=(task_id,), daemon=True)
                    thread.start()
                    logger.info(f"任务 {task_id} 已从检查点恢复执行")
    except Exception as e:
        logger.error(f"恢复未完成任务失败: {e}")

    yield
    logger.info("智能投研平台 API Gateway 关闭中...")


app = FastAPI(
    title="智能投研平台 API",
    description="基于 LangGraph 的五阶段 AI 投资研究系统",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 中间件（来源白名单由 CORS_ORIGINS 配置，默认仅本地前端；设为 * 则放开）
_cors_origins = [o.strip() for o in get_settings().CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API 限流（安全加固） ====================

class RateLimitMiddleware:
    """轻量内存令牌桶限流（每 IP 每分钟）

    - 普通接口: RATE_LIMIT_PER_MINUTE（默认 120 次/分钟）
    - 创建研究任务 POST /api/v1/research: RATE_LIMIT_RESEARCH_PER_MINUTE（默认 10 次/分钟，重资源接口）

    单进程内存实现，零依赖；多实例部署时可替换为 Redis 计数器（接口语义不变）。
    """

    def __init__(self, app):
        self.app = app
        self._buckets = {}  # {key: [timestamps]}
        self._lock = threading.Lock()

    def _allow(self, key: str, limit: int, window: float = 60.0) -> bool:
        import time as _time
        now = _time.time()
        with self._lock:
            hits = [t for t in self._buckets.get(key, []) if now - t < window]
            if len(hits) >= limit:
                self._buckets[key] = hits
                return False
            hits.append(now)
            self._buckets[key] = hits
            # 防止内存膨胀：桶数量超限时清理过期桶
            if len(self._buckets) > 10000:
                self._buckets = {k: v for k, v in self._buckets.items() if v and now - v[-1] < window}
            return True

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        settings = get_settings()
        client = scope.get("client") or ("unknown", 0)
        ip = client[0]
        path = scope.get("path", "")
        method = scope.get("method", "GET")

        # 健康检查不限流（Docker HEALTHCHECK 依赖）
        if path in ("/health", "/"):
            await self.app(scope, receive, send)
            return

        if method == "POST" and path == "/api/v1/research":
            allowed = self._allow(f"research:{ip}", settings.RATE_LIMIT_RESEARCH_PER_MINUTE)
        else:
            allowed = self._allow(f"general:{ip}", settings.RATE_LIMIT_PER_MINUTE)

        if not allowed:
            response = JSONResponse(
                status_code=429,
                content={"success": False, "message": "请求过于频繁，请稍后再试",
                         "error": "rate_limit_exceeded"},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


app.add_middleware(RateLimitMiddleware)


@app.get("/", response_model=APIResponse)
async def root():
    """根路径 - 健康检查"""
    settings = get_settings()
    return APIResponse(
        success=True,
        message="智能投研平台 API 运行正常",
        data={
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "running"
        }
    )


@app.get("/health", response_model=APIResponse)
async def health_check():
    """健康检查端点"""
    return APIResponse(success=True, message="服务正常", data={"status": "healthy"})


# ==================== 研究分析 API ====================

@app.post("/api/v1/research", response_model=APIResponse)
async def create_research_task(request: ResearchRequest, background_tasks: BackgroundTasks):
    """创建研究分析任务"""
    try:
        settings = get_settings()
        
        # 验证 API Key
        if not settings.DASHSCOPE_API_KEY:
            raise HTTPException(
                status_code=400,
                detail="请配置 DASHSCOPE_API_KEY 环境变量或在请求中提供"
            )
        
        # 生成任务 ID
        task_id = str(uuid.uuid4())
        
        # 创建任务状态
        task_state = {
            "task_id": task_id,
            "status": "pending",
            "current_phase": "初始化",
            "research_topic": request.topic,
            "industry_focus": request.industry,
            "time_horizon": request.horizon,
            "use_realtime_data": request.use_realtime_data,
            "use_agentic_perception": request.use_agentic_perception,
            "stock_symbols": request.stock_symbols,
            "api_key": settings.DASHSCOPE_API_KEY,
            "api_url": settings.DASHSCOPE_API_URL,
            "model_name": request.model_name or settings.DEFAULT_MODEL_NAME,
            "perception_data": None,
            "world_model": None,
            "reasoning_plans": None,
            "selected_plan": None,
            "final_report": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "completed_at": None
        }
        
        active_tasks[task_id] = task_state
        
        # 在独立线程中运行分析任务（不阻塞API）
        thread = threading.Thread(target=run_research_analysis, args=(task_id,), daemon=True)
        thread.start()
        
        return APIResponse(
            success=True,
            message="研究任务已创建，正在后台执行",
            data={"task_id": task_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建研究任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/research/{task_id}", response_model=APIResponse)
async def get_research_task(task_id: str):
    """获取研究任务状态"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = active_tasks[task_id]
    
    return APIResponse(
        success=True,
        message="任务状态获取成功",
        data=ResearchProgressResponse(
            task_id=task["task_id"],
            status=task["status"],
            current_phase=task["current_phase"],
            progress_percent=get_progress_percent(task),
            perception_data=task.get("perception_data"),
            world_model=task.get("world_model"),
            reasoning_plans=task.get("reasoning_plans"),
            selected_plan=task.get("selected_plan"),
            reflection=task.get("reflection"),
            trace_summary=task.get("trace_summary"),
            final_report=task.get("final_report"),
            error=task.get("error"),
            created_at=task.get("created_at"),
            completed_at=task.get("completed_at")
        ).model_dump()
    )


@app.get("/api/v1/research", response_model=APIResponse)
async def list_research_tasks(limit: int = 20):
    """列出研究任务"""
    tasks = list(active_tasks.values())
    # 按创建时间倒序
    tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    tasks = tasks[:limit]

    return APIResponse(
        success=True,
        message="任务列表获取成功",
        data=[ResearchProgressResponse(
            task_id=t["task_id"],
            status=t["status"],
            current_phase=t["current_phase"],
            progress_percent=get_progress_percent(t),
            created_at=t.get("created_at"),
            completed_at=t.get("completed_at")
        ).model_dump() for t in tasks]
    )


@app.get("/api/v1/research/{task_id}/stream")
async def stream_research_task(task_id: str):
    """SSE 流式推送任务进度（前端可实时展示阶段/工具调用轨迹）

    对齐「流式响应」「生成式 UI」：前端订阅此端点，随任务推进实时渲染卡片。
    """
    from fastapi.responses import StreamingResponse

    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    def event_stream():
        import time as _time
        last_phase = None
        # 最多推送 15 分钟，完成后发送 done 事件并关闭
        deadline = _time.time() + 900
        while _time.time() < deadline:
            task = active_tasks.get(task_id)
            if task is None:
                yield f"event: error\ndata: {json.dumps({'error': '任务已清除'})}\n\n"
                break

            current_phase = task.get("current_phase")
            status = task.get("status")
            payload = {
                "task_id": task_id,
                "status": status,
                "current_phase": current_phase,
                "progress_percent": get_progress_percent(task),
            }
            # 仅在阶段变化或完成时推送，避免刷屏
            if current_phase != last_phase or status in ("completed", "failed"):
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                last_phase = current_phase

            if status in ("completed", "failed"):
                # 推送最终结果摘要
                final_payload = {
                    "status": status,
                    "final_report_preview": (task.get("final_report") or "")[:300],
                    "reflection": task.get("reflection"),
                }
                yield f"event: done\ndata: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
                break

            _time.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ==================== 历史记录 API ====================

@app.get("/api/v1/history", response_model=APIResponse)
async def get_history(limit: int = 20):
    """获取分析历史记录"""
    history = load_history()
    return APIResponse(
        success=True,
        message="历史记录获取成功",
        data=history[:limit]
    )


@app.delete("/api/v1/history", response_model=APIResponse)
async def clear_history():
    """清空历史记录"""
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        return APIResponse(success=True, message="历史记录已清空")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/history/{index}", response_model=APIResponse)
async def delete_history_record(index: int):
    """删除单条历史记录"""
    try:
        history = load_history()
        if 0 <= index < len(history):
            history.pop(index)
            save_history(history)
            return APIResponse(success=True, message="记录已删除")
        else:
            raise HTTPException(status_code=404, detail="记录不存在")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 配置 API ====================

@app.post("/api/v1/config/llm", response_model=APIResponse)
async def update_llm_config(config: LLMConfigRequest):
    """更新 LLM 配置（运行时）"""
    # 注意：这里只是示例，生产环境应该使用数据库或配置中心
    return APIResponse(
        success=True,
        message="配置更新成功（仅当前会话有效）",
        data={
            "api_url": config.api_url,
            "model_name": config.model_name
        }
    )


@app.get("/api/v1/config", response_model=APIResponse)
async def get_config():
    """获取当前配置"""
    settings = get_settings()
    return APIResponse(
        success=True,
        message="配置获取成功",
        data={
            "api_url": settings.DASHSCOPE_API_URL,
            "model_name": settings.DEFAULT_MODEL_NAME,
            "api_key_configured": bool(settings.DASHSCOPE_API_KEY)
        }
    )


# ==================== 内部工作流函数 ====================

def get_progress_percent(task: dict) -> int:
    """计算任务进度百分比"""
    phase_progress = {
        "初始化": 0,
        "perception": 18,
        "modeling": 36,
        "reasoning": 54,
        "decision": 70,
        "reflect": 82,
        "report": 92,
        "completed": 100,
        "error": 100
    }
    return phase_progress.get(task.get("current_phase", "初始化"), 0)


def load_history():
    """加载历史记录（旧根目录文件自动迁移到 data/）"""
    try:
        if not os.path.exists(HISTORY_FILE) and os.path.exists(_LEGACY_HISTORY_FILE):
            import shutil
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            shutil.copy2(_LEGACY_HISTORY_FILE, HISTORY_FILE)
            logger.info("历史记录已从根目录迁移到 data/analysis_history.json")
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"加载历史记录失败: {e}")
    return []


def save_history(history):
    """保存历史记录"""
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存历史记录失败: {e}")


def run_research_analysis(task_id: str):
    """在独立线程中运行研究分析"""
    import traceback
    from agent_service.workflow import run_research_agent
    
    try:
        with tasks_lock:
            task = active_tasks[task_id]
            task["status"] = "running"
        
        # 更新阶段进度（线程安全）
        def update_phase(phase_name: str):
            with tasks_lock:
                active_tasks[task_id]["current_phase"] = phase_name
        
        # 运行研究分析（传入 task_id 启用检查点持久化）
        result = run_research_agent(
            topic=task["research_topic"],
            industry=task["industry_focus"],
            horizon=task["time_horizon"],
            api_key=task["api_key"],
            api_url=task["api_url"],
            model_name=task["model_name"],
            progress_callback=update_phase,
            use_realtime_data=task.get("use_realtime_data", True),
            use_agentic_perception=task.get("use_agentic_perception", True),
            task_id=task_id
        )
        
        # 更新任务结果
        with tasks_lock:
            active_tasks[task_id].update(result)
            active_tasks[task_id]["status"] = "completed" if result.get("current_phase") == "completed" else "failed"
            active_tasks[task_id]["completed_at"] = datetime.now().isoformat()
            # 兜底：失败任务必须带错误信息，否则前端无法展示失败原因
            if active_tasks[task_id]["status"] == "failed" and not active_tasks[task_id].get("error"):
                active_tasks[task_id]["error"] = (
                    f"分析未完成（最终阶段: {result.get('current_phase', '未知')}），"
                    f"详情请查看任务执行轨迹 /api/v1/research/{task_id}/trace"
                )
            task = active_tasks[task_id]
        
        # 保存到历史记录
        if task["status"] == "completed":
            record = {
                "id": task_id,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "topic": task["research_topic"],
                "industry": task["industry_focus"],
                "horizon": task["time_horizon"],
                "model": task["model_name"],
                "report_preview": (result.get("final_report", "")[:200] + "...") if result.get("final_report") else "无报告"
            }
            with tasks_lock:
                history = load_history()
                history.insert(0, record)
                if len(history) > 20:
                    history = history[:20]
                save_history(history)
        
        logger.info(f"任务 {task_id} 完成，状态: {task['status']}")
        
    except Exception as e:
        logger.error(f"任务 {task_id} 执行失败: {e}")
        logger.error(traceback.format_exc())
        with tasks_lock:
            if task_id in active_tasks:
                active_tasks[task_id].update({
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.now().isoformat()
                })


# ==================== PDF 导出 API ====================

@app.post("/api/v1/research/{task_id}/export-pdf")
async def export_pdf(task_id: str):
    """导出 PDF 报告"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = active_tasks[task_id]
    if not task.get("final_report"):
        raise HTTPException(status_code=400, detail="报告尚未生成完成")
    
    try:
        from report_service.pdf_generator import generate_pdf_report

        logger.info(f"开始生成PDF: task_id={task_id}, topic={task['research_topic']}")
        pdf_data = generate_pdf_report(
            report_text=task["final_report"],
            topic=task["research_topic"],
            industry=task["industry_focus"],
            horizon=task["time_horizon"]
        )
        logger.info(f"PDF生成成功: {len(pdf_data)} bytes")

        from fastapi.responses import Response
        return Response(
            content=pdf_data,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=report_{datetime.now().strftime('%Y%m%d')}.pdf"
            }
        )
    except Exception as e:
        import traceback
        logger.error(f"PDF 导出失败: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"PDF 生成失败: {str(e)}")


# ==================== 实时市场数据 API ====================

@app.get("/api/v1/market/quote/{symbol}", response_model=APIResponse)
async def get_stock_quote(symbol: str):
    """获取个股实时行情"""
    try:
        from data_fetcher.stock_data import StockDataFetcher
        fetcher = StockDataFetcher()
        data = fetcher.get_realtime_quote(symbol)
        if "error" in data:
            return APIResponse(success=False, message=data["error"], error=data["error"], data=data)
        return APIResponse(success=True, message="行情获取成功", data=data)
    except ImportError:
        raise HTTPException(status_code=500, detail="AKShare 数据服务未安装")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/market/indices", response_model=APIResponse)
async def get_market_indices():
    """获取主要指数行情"""
    try:
        from data_fetcher.market_indices import MarketIndicesFetcher
        fetcher = MarketIndicesFetcher()
        data = fetcher.get_major_indices()
        return APIResponse(success=True, message="指数获取成功", data=data)
    except ImportError:
        raise HTTPException(status_code=500, detail="AKShare 数据服务未安装")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/market/overview", response_model=APIResponse)
async def get_market_overview():
    """获取市场概览（涨跌家数、成交额等）"""
    try:
        from data_fetcher.market_indices import MarketIndicesFetcher
        fetcher = MarketIndicesFetcher()
        data = fetcher.get_market_overview()
        if "error" in data:
            return APIResponse(success=False, message=data["error"], data=data)
        return APIResponse(success=True, message="市场概览获取成功", data=data)
    except ImportError:
        raise HTTPException(status_code=500, detail="AKShare 数据服务未安装")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/market/sector/{sector_name}", response_model=APIResponse)
async def get_sector_data(sector_name: str, top_stocks: int = 5):
    """获取板块行情数据"""
    try:
        from data_fetcher.stock_data import StockDataFetcher
        fetcher = StockDataFetcher()
        sector_data = fetcher.get_sector_performance(sector_name)
        sector_stocks = fetcher.get_sector_top_stocks(sector_name, limit=top_stocks)
        return APIResponse(
            success=True,
            message="板块数据获取成功",
            data={"sector": sector_data, "top_stocks": sector_stocks}
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="AKShare 数据服务未安装")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/market/sectors", response_model=APIResponse)
async def get_sector_list():
    """获取所有行业板块列表"""
    try:
        from data_fetcher.market_indices import MarketIndicesFetcher
        fetcher = MarketIndicesFetcher()
        data = fetcher.get_sector_list()
        return APIResponse(success=True, message="板块列表获取成功", data=data)
    except ImportError:
        raise HTTPException(status_code=500, detail="AKShare 数据服务未安装")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/market/news", response_model=APIResponse)
async def get_market_news(keyword: str = "", limit: int = 10):
    """获取财经新闻"""
    try:
        from data_fetcher.news_fetcher import NewsFetcher
        fetcher = NewsFetcher()
        data = fetcher.get_financial_news(keyword=keyword, limit=limit)
        return APIResponse(success=True, message="新闻获取成功", data=data)
    except ImportError:
        raise HTTPException(status_code=500, detail="AKShare 数据服务未安装")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/market/gainers-losers", response_model=APIResponse)
async def get_gainers_losers(limit: int = 10):
    """获取涨跌排行"""
    try:
        from data_fetcher.stock_data import StockDataFetcher
        fetcher = StockDataFetcher()
        gainers = fetcher.get_top_gainers(limit=limit)
        losers = fetcher.get_top_losers(limit=limit)
        return APIResponse(
            success=True,
            message="涨跌排行获取成功",
            data={"gainers": gainers, "losers": losers}
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="AKShare 数据服务未安装")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/market/capital-flow", response_model=APIResponse)
async def get_capital_flow():
    """获取市场资金流向"""
    try:
        from data_fetcher.market_indices import MarketIndicesFetcher
        fetcher = MarketIndicesFetcher()
        data = fetcher.get_capital_flow()
        if "error" in data:
            return APIResponse(success=False, message=data["error"], data=data)
        return APIResponse(success=True, message="资金流向获取成功", data=data)
    except ImportError:
        raise HTTPException(status_code=500, detail="AKShare 数据服务未安装")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/market/search", response_model=APIResponse)
async def search_stock(keyword: str):
    """搜索股票"""
    try:
        from data_fetcher.stock_data import StockDataFetcher
        fetcher = StockDataFetcher()
        data = fetcher.search_stock(keyword)
        return APIResponse(success=True, message="搜索完成", data=data)
    except ImportError:
        raise HTTPException(status_code=500, detail="AKShare 数据服务未安装")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/market/financial/{symbol}", response_model=APIResponse)
async def get_stock_financial(symbol: str):
    """获取个股财务数据"""
    try:
        from data_fetcher.financials import FinancialsFetcher
        fetcher = FinancialsFetcher()
        data = fetcher.get_stock_financial(symbol)
        return APIResponse(success=True, message="财务数据获取完成", data=data)
    except ImportError:
        raise HTTPException(status_code=500, detail="AKShare 数据服务未安装")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/market/financial/{symbol}/profit", response_model=APIResponse)
async def get_stock_profit(symbol: str):
    """获取个股盈利能力数据"""
    try:
        from data_fetcher.financials import FinancialsFetcher
        fetcher = FinancialsFetcher()
        data = fetcher.get_stock_profit(symbol)
        return APIResponse(success=True, message="盈利数据获取完成", data=data)
    except ImportError:
        raise HTTPException(status_code=500, detail="AKShare 数据服务未安装")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/market/financial/{symbol}/balance", response_model=APIResponse)
async def get_stock_balance(symbol: str):
    """获取个股资产负债数据"""
    try:
        from data_fetcher.financials import FinancialsFetcher
        fetcher = FinancialsFetcher()
        data = fetcher.get_stock_balance(symbol)
        return APIResponse(success=True, message="资产负债数据获取完成", data=data)
    except ImportError:
        raise HTTPException(status_code=500, detail="AKShare 数据服务未安装")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/market/financial/{symbol}/cashflow", response_model=APIResponse)
async def get_stock_cashflow(symbol: str):
    """获取个股现金流量数据"""
    try:
        from data_fetcher.financials import FinancialsFetcher
        fetcher = FinancialsFetcher()
        data = fetcher.get_stock_cashflow(symbol)
        return APIResponse(success=True, message="现金流数据获取完成", data=data)
    except ImportError:
        raise HTTPException(status_code=500, detail="AKShare 数据服务未安装")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Agent 能力观测 API（v2 新增） ====================

@app.get("/api/v1/agent/tools", response_model=APIResponse)
async def list_agent_tools():
    """列出 Agent 可用的全部工具（Tool 注册中心动态发现）

    面试演示点：新增工具只需注册，此处自动可见，Agent 无需改代码。
    """
    try:
        from agent_service.tools.registry import registry
        from agent_service.tools import market_tools  # noqa: F401  触发注册
        tools = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "readonly": t.readonly,
                "timeout": t.timeout,
            }
            for t in registry.list_tools()
        ]
        return APIResponse(success=True, message=f"共 {len(tools)} 个工具", data=tools)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agent/mcp/status", response_model=APIResponse)
async def mcp_status():
    """MCP Server 探活 + 动态工具发现演示"""
    try:
        from agent_service.tools.mcp_client import mcp_client
        available = mcp_client.ping()
        tools = mcp_client.list_tools() if available else []
        return APIResponse(
            success=True,
            message="MCP 可用" if available else "MCP 不可用（已降级为进程内执行）",
            data={"available": available, "discovered_tools": [t.get("name") for t in tools]},
        )
    except Exception as e:
        return APIResponse(success=False, message=f"MCP 探测失败: {e}", error=str(e))


@app.get("/api/v1/agent/memory/stats", response_model=APIResponse)
async def memory_stats():
    """长期记忆统计（条数、Embedding 模式、容量）"""
    try:
        from agent_service.memory.memory import LongTermMemory
        from agent_service.memory.embeddings import EmbeddingProvider
        settings = get_settings()
        memory = LongTermMemory(embedding=EmbeddingProvider(
            api_key=settings.DASHSCOPE_API_KEY, base_url=settings.DASHSCOPE_API_URL))
        return APIResponse(success=True, message="记忆统计获取成功", data=memory.stats())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agent/tools/metrics", response_model=APIResponse)
async def tool_metrics_report():
    """工具级指标：成功率 / 平均时延 / P95 / 异常分布（定位 Tool 能力边界）"""
    try:
        from agent_service.tracer import tool_metrics
        return APIResponse(success=True, message="工具指标获取成功", data=tool_metrics.report())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/research/{task_id}/trace", response_model=APIResponse)
async def get_task_trace(task_id: str):
    """获取任务的完整执行轨迹（每步 LLM/Tool 调用明细）"""
    try:
        from agent_service.tracer import Tracer
        trace = Tracer.load(task_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="Trace 不存在（任务可能未启用 trace 或已被清理）")
        return APIResponse(success=True, message="Trace 获取成功", data=trace)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
