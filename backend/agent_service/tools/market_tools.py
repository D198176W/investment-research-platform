"""内置市场数据工具 —— 封装 data_fetcher 为标准 Tool

每个工具仅做薄封装：调用 data_fetcher → 统一返回 dict（含 source/retrieved_at 元数据，供报告引用溯源）。
异常分类：网络/超时类错误抛出 ToolRetryableError，由执行器决定重试或换参。
"""
import logging
from datetime import datetime
from typing import Any, Dict

from .registry import register_tool

logger = logging.getLogger(__name__)


class ToolRetryableError(Exception):
    """可重试错误（网络超时、限流等）—— 执行器可换参数或稍后重试"""


class ToolFatalError(Exception):
    """不可重试错误（参数非法、标的不存在等）—— 执行器应立即反馈给模型换参"""


def _wrap(data: Any, source: str) -> Dict[str, Any]:
    """统一包装工具返回，附加来源与时间戳（支持报告引用溯源）"""
    return {
        "data": data,
        "source": source,
        "retrieved_at": datetime.now().isoformat(timespec="seconds"),
    }


def _lazy_fetchers():
    """延迟导入 data_fetcher，避免 AKShare 未安装时阻塞模块加载"""
    try:
        from data_fetcher.stock_data import StockDataFetcher
        from data_fetcher.news_fetcher import NewsFetcher
        from data_fetcher.market_indices import MarketIndicesFetcher
        from data_fetcher.financials import FinancialsFetcher
        return StockDataFetcher(), NewsFetcher(), MarketIndicesFetcher(), FinancialsFetcher()
    except ImportError as e:
        raise ToolFatalError(f"AKShare 数据服务未安装: {e}")


def _guard(result: Any, source: str) -> Dict[str, Any]:
    """结果校验：data_fetcher 约定用 {"error": ...} 表示失败"""
    if isinstance(result, dict) and "error" in result:
        raise ToolRetryableError(f"{source} 返回错误: {result['error']}")
    if result is None or (isinstance(result, (list, dict)) and len(result) == 0):
        raise ToolRetryableError(f"{source} 返回空数据")
    return _wrap(result, source)


# ==================== 行情类工具 ====================

@register_tool(
    name="get_stock_quote",
    description="获取A股个股实时行情（最新价、涨跌幅、成交量、市值等）",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "6位股票代码，如 600519、000001"},
        },
        "required": ["symbol"],
    },
)
def tool_get_stock_quote(symbol: str):
    stock, *_ = _lazy_fetchers()
    if not symbol or not symbol.strip().isdigit():
        raise ToolFatalError(f"股票代码格式非法: {symbol!r}，应为6位数字")
    return _guard(stock.get_realtime_quote(symbol.strip()), f"个股行情({symbol})")


@register_tool(
    name="get_stock_kline",
    description="获取A股个股历史K线数据（近N日日线，含开高低收、成交量），用于趋势分析",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "6位股票代码"},
            "days": {"type": "integer", "description": "回溯天数，默认60", "default": 60},
        },
        "required": ["symbol"],
    },
)
def tool_get_stock_kline(symbol: str, days: int = 60):
    stock, *_ = _lazy_fetchers()
    days = max(5, min(int(days), 250))
    return _guard(stock.get_historical_kline(symbol.strip(), days=days), f"历史K线({symbol},{days}日)")


@register_tool(
    name="get_major_indices",
    description="获取A股主要指数实时行情（上证指数、深证成指、创业板指、沪深300等）",
    parameters={"type": "object", "properties": {}},
)
def tool_get_major_indices():
    _, _, indices, _ = _lazy_fetchers()
    return _guard(indices.get_major_indices(), "主要指数")


@register_tool(
    name="get_market_overview",
    description="获取A股市场整体概览（上涨/下跌家数、涨跌停家数、平均涨跌幅、总成交额），用于判断市场情绪",
    parameters={"type": "object", "properties": {}},
)
def tool_get_market_overview():
    _, _, indices, _ = _lazy_fetchers()
    return _guard(indices.get_market_overview(), "市场概览")


@register_tool(
    name="get_capital_flow",
    description="获取A股市场资金流向（主力/北向资金净流入等），用于判断资金面",
    parameters={"type": "object", "properties": {}},
)
def tool_get_capital_flow():
    _, _, indices, _ = _lazy_fetchers()
    return _guard(indices.get_capital_flow(), "资金流向")


@register_tool(
    name="get_sector_performance",
    description="获取行业板块行情（板块涨跌幅、上涨/下跌家数、成交额），用于行业景气度分析",
    parameters={
        "type": "object",
        "properties": {
            "sector_name": {"type": "string", "description": "板块名称，如 半导体、新能源、人工智能"},
        },
        "required": ["sector_name"],
    },
)
def tool_get_sector_performance(sector_name: str):
    stock, *_ = _lazy_fetchers()
    return _guard(stock.get_sector_performance(sector_name.strip()), f"板块行情({sector_name})")


@register_tool(
    name="get_sector_top_stocks",
    description="获取行业板块成分股行情（板块内个股涨跌幅排行），用于挖掘板块内强势标的",
    parameters={
        "type": "object",
        "properties": {
            "sector_name": {"type": "string", "description": "板块名称"},
            "limit": {"type": "integer", "description": "返回数量，默认10", "default": 10},
        },
        "required": ["sector_name"],
    },
)
def tool_get_sector_top_stocks(sector_name: str, limit: int = 10):
    stock, *_ = _lazy_fetchers()
    limit = max(1, min(int(limit), 30))
    return _guard(stock.get_sector_top_stocks(sector_name.strip(), limit=limit), f"板块成分股({sector_name})")


@register_tool(
    name="get_top_gainers",
    description="获取全市场涨幅榜前列个股，用于发现市场热点",
    parameters={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "返回数量，默认10", "default": 10},
        },
    },
)
def tool_get_top_gainers(limit: int = 10):
    stock, *_ = _lazy_fetchers()
    return _guard(stock.get_top_gainers(limit=max(1, min(int(limit), 30))), "涨幅榜")


@register_tool(
    name="get_top_losers",
    description="获取全市场跌幅榜前列个股，用于识别市场风险集中区",
    parameters={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "返回数量，默认10", "default": 10},
        },
    },
)
def tool_get_top_losers(limit: int = 10):
    stock, *_ = _lazy_fetchers()
    return _guard(stock.get_top_losers(limit=max(1, min(int(limit), 30))), "跌幅榜")


@register_tool(
    name="search_stock",
    description="按关键词搜索A股股票（返回代码与名称），用于将公司名/概念解析为股票代码",
    parameters={
        "type": "object",
        "properties": {
            "keyword": {"type": "string", "description": "搜索关键词，如 茅台、宁德、AI"},
        },
        "required": ["keyword"],
    },
)
def tool_search_stock(keyword: str):
    stock, *_ = _lazy_fetchers()
    if not keyword or not keyword.strip():
        raise ToolFatalError("搜索关键词不能为空")
    return _guard(stock.search_stock(keyword.strip()), f"股票搜索({keyword})")


# ==================== 财务类工具 ====================

@register_tool(
    name="get_stock_financial",
    description="获取A股个股核心财务指标（营收、净利润、ROE、资产负债率等），用于基本面分析",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "6位股票代码"},
        },
        "required": ["symbol"],
    },
    timeout=20,
)
def tool_get_stock_financial(symbol: str):
    *_, fin = _lazy_fetchers()
    return _guard(fin.get_stock_financial(symbol.strip()), f"财务指标({symbol})")


@register_tool(
    name="get_stock_profit",
    description="获取A股个股盈利能力数据（毛利率、净利率、ROE趋势）",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "6位股票代码"},
        },
        "required": ["symbol"],
    },
    timeout=20,
)
def tool_get_stock_profit(symbol: str):
    *_, fin = _lazy_fetchers()
    return _guard(fin.get_stock_profit(symbol.strip()), f"盈利能力({symbol})")


# ==================== 新闻类工具 ====================

@register_tool(
    name="get_financial_news",
    description="按关键词获取最新财经新闻（含发布时间、来源），用于事件驱动与舆情分析",
    parameters={
        "type": "object",
        "properties": {
            "keyword": {"type": "string", "description": "新闻关键词，如 AI芯片、新能源，留空获取全市场要闻", "default": ""},
            "limit": {"type": "integer", "description": "返回数量，默认10", "default": 10},
        },
    },
)
def tool_get_financial_news(keyword: str = "", limit: int = 10):
    _, news, *_ = _lazy_fetchers()
    limit = max(1, min(int(limit), 20))
    return _guard(news.get_financial_news(keyword=keyword.strip(), limit=limit), f"财经新闻({keyword or '要闻'})")


@register_tool(
    name="get_stock_news",
    description="获取个股相关新闻与公告，用于个股事件跟踪",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "6位股票代码"},
            "limit": {"type": "integer", "description": "返回数量，默认10", "default": 10},
        },
        "required": ["symbol"],
    },
)
def tool_get_stock_news(symbol: str, limit: int = 10):
    _, news, *_ = _lazy_fetchers()
    return _guard(news.get_stock_news(symbol.strip(), limit=max(1, min(int(limit), 20))), f"个股新闻({symbol})")


@register_tool(
    name="get_macro_news",
    description="获取宏观经济新闻（货币政策、财政政策、宏观数据），用于宏观环境分析",
    parameters={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "返回数量，默认10", "default": 10},
        },
    },
)
def tool_get_macro_news(limit: int = 10):
    _, news, *_ = _lazy_fetchers()
    return _guard(news.get_macro_news(limit=max(1, min(int(limit), 20))), "宏观新闻")
