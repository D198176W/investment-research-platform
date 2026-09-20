"""股票行情数据抓取 - 基于 AKShare"""
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta

from .cache_manager import cache
from .config import get_data_fetcher_settings

logger = logging.getLogger(__name__)

# AKShare 请求超时时间（秒）
AKSHARE_TIMEOUT = 8


def _safe_import_akshare():
    """安全导入 AKShare"""
    try:
        import akshare as ak
        return ak
    except ImportError:
        logger.error("AKShare 未安装，请执行: pip install akshare")
        return None


def _run_with_timeout(func: Callable, timeout_seconds: int = AKSHARE_TIMEOUT, default_return=None):
    """在独立线程中执行函数，支持超时控制

    Args:
        func: 要执行的函数（无参数）
        timeout_seconds: 超时时间
        default_return: 超时或异常时的默认返回值

    Returns:
        函数执行结果，或 default_return
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        logger.warning(f"AKShare 请求超时({timeout_seconds}秒)")
        return default_return
    except Exception as e:
        logger.warning(f"AKShare 请求异常: {e}")
        return default_return


class StockDataFetcher:
    """A股股票行情数据抓取器"""

    # 全市场快照缓存时长（秒）—— 全市场数据可被多次查询复用
    MARKET_SNAPSHOT_TTL = 60

    def __init__(self):
        self.settings = get_data_fetcher_settings()

    def _get_ak(self):
        ak = _safe_import_akshare()
        if ak is None:
            raise RuntimeError("AKShare 未安装")
        return ak

    @staticmethod
    def _clean_symbol(symbol: str) -> str:
        """清理股票代码，去掉交易所前缀，保留6位数字"""
        return symbol.replace("sh", "").replace("sz", "").replace("SH", "").replace("SZ", "").strip()

    def _get_market_snapshot(self):
        """获取全市场快照（带缓存），返回 DataFrame 或 None

        全市场快照会被多个股票查询复用，避免每次查询都拉取整个市场数据。
        """
        cache_key = "market:snapshot:spot_em"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        def _fetch_market():
            ak = self._get_ak()
            return ak.stock_zh_a_spot_em()

        df = _run_with_timeout(_fetch_market, timeout_seconds=15, default_return=None)
        if df is not None and not df.empty:
            cache.set(cache_key, df, self.MARKET_SNAPSHOT_TTL)
            return df
        return None

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        """安全转 float，None/NaN/异常时返回默认值"""
        try:
            if value is None:
                return default
            import math
            f = float(value)
            return default if math.isnan(f) else f
        except (ValueError, TypeError):
            return default

    def _build_quote_from_spot_row(self, clean_symbol: str, row_data) -> Dict[str, Any]:
        """从全市场快照的一行数据构建行情字典"""
        return {
            "symbol": clean_symbol,
            "name": str(row_data.get("名称", "")),
            "price": self._to_float(row_data.get("最新价")),
            "change_pct": self._to_float(row_data.get("涨跌幅")),
            "change_amount": self._to_float(row_data.get("涨跌额")),
            "volume": self._to_float(row_data.get("成交量")),
            "turnover": self._to_float(row_data.get("成交额")),
            "amplitude": self._to_float(row_data.get("振幅")),
            "high": self._to_float(row_data.get("最高")),
            "low": self._to_float(row_data.get("最低")),
            "open": self._to_float(row_data.get("今开")),
            "prev_close": self._to_float(row_data.get("昨收")),
            "volume_ratio": self._to_float(row_data.get("量比")),
            "pe_ratio": self._to_float(row_data.get("市盈率-动态")),
            "pb_ratio": self._to_float(row_data.get("市净率")),
            "total_mv": self._to_float(row_data.get("总市值")),
            "circ_mv": self._to_float(row_data.get("流通市值")),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_source": "AKShare-东方财富(实时)"
        }

    def _build_quote_from_hist_row(self, clean_symbol: str, row_data, prev_close: float = 0.0) -> Dict[str, Any]:
        """从历史K线最新一条构建行情字典（降级数据源）"""
        close = self._to_float(row_data.get("收盘"))
        # 历史数据没有实时涨跌幅字段，用昨收手动计算
        change_pct = ((close - prev_close) / prev_close * 100) if prev_close else self._to_float(row_data.get("涨跌幅"))
        return {
            "symbol": clean_symbol,
            "name": str(row_data.get("名称", "")) if "名称" in row_data else "",
            "price": close,
            "change_pct": change_pct,
            "change_amount": self._to_float(close - prev_close) if prev_close else 0.0,
            "volume": self._to_float(row_data.get("成交量")),
            "turnover": self._to_float(row_data.get("成交额")),
            "amplitude": 0.0,
            "high": self._to_float(row_data.get("最高")),
            "low": self._to_float(row_data.get("最低")),
            "open": self._to_float(row_data.get("开盘")),
            "prev_close": prev_close,
            "volume_ratio": 0.0,
            "pe_ratio": 0.0,
            "pb_ratio": 0.0,
            "total_mv": 0.0,
            "circ_mv": 0.0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_source": "AKShare-东方财富(日线)"
        }

    def get_realtime_quote(self, symbol: str) -> Dict[str, Any]:
        """获取个股实时行情（三级降级策略）

        降级顺序：
            1. 查询缓存
            2. 从全市场快照中筛选（快照本身带60秒缓存，多只股票查询可复用）
            3. 单只股票历史日线接口取最新一条（轻量、快速、稳定）

        Args:
            symbol: 股票代码，如 "600519" 或 "sh600519"

        Returns:
            包含实时行情数据的字典，失败时包含 error 字段
        """
        # ① 查询缓存
        cache_key = f"quote:{symbol}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        clean_symbol = self._clean_symbol(symbol)
        if not clean_symbol or not clean_symbol.isdigit() or len(clean_symbol) != 6:
            return {"symbol": symbol, "error": f"股票代码格式错误：{symbol}（应为6位数字，如 600519）"}

        # ② 优先从全市场快照中筛选（带缓存，多查询复用）
        try:
            snapshot = self._get_market_snapshot()
            if snapshot is not None and not snapshot.empty:
                row = snapshot[snapshot["代码"] == clean_symbol]
                if not row.empty:
                    result = self._build_quote_from_spot_row(clean_symbol, row.iloc[0])
                    cache.set(cache_key, result, self.settings.QUOTE_CACHE_TTL)
                    return result
        except Exception as e:
            logger.warning(f"全市场快照查询失败，降级到单只查询: {e}")

        # ③ 降级方案：用单只股票历史日线接口取最新一条（轻量快速）
        def _fetch_single():
            ak = self._get_ak()
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=15)).strftime("%Y%m%d")
            df = ak.stock_zh_a_hist(
                symbol=clean_symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            if df is None or df.empty:
                return {"symbol": clean_symbol, "error": f"未找到股票 {clean_symbol}（请确认是正确的6位A股代码）"}
            latest_row = df.iloc[-1]
            prev_close = self._to_float(df.iloc[-2]["收盘"]) if len(df) >= 2 else 0.0
            return self._build_quote_from_hist_row(clean_symbol, latest_row, prev_close)

        result = _run_with_timeout(
            _fetch_single,
            timeout_seconds=10,
            default_return={"symbol": clean_symbol, "error": "请求超时（10秒），AKShare 数据源响应缓慢，请稍后重试"}
        )

        if result and "error" not in result:
            cache.set(cache_key, result, self.settings.QUOTE_CACHE_TTL)

        return result if result else {"symbol": clean_symbol, "error": "请求失败，请检查网络或稍后重试"}

    def get_realtime_quotes_batch(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """批量获取实时行情（一次拉取全市场数据，从中筛选）"""
        cache_key = "quotes:batch_all"
        cached = cache.get(cache_key)
        if cached is not None:
            # 从缓存中筛选
            clean_symbols = [s.replace("sh", "").replace("sz", "").replace("SH", "").replace("SZ", "") for s in symbols]
            return [q for q in cached if q.get("symbol") in clean_symbols]

        try:
            ak = self._get_ak()
            df = ak.stock_zh_a_spot_em()

            results = []
            for _, row in df.iterrows():
                results.append({
                    "symbol": str(row.get("代码", "")),
                    "name": str(row.get("名称", "")),
                    "price": float(row.get("最新价", 0)),
                    "change_pct": float(row.get("涨跌幅", 0)),
                    "change_amount": float(row.get("涨跌额", 0)),
                    "volume": float(row.get("成交量", 0)),
                    "turnover": float(row.get("成交额", 0)),
                    "amplitude": float(row.get("振幅", 0)),
                    "high": float(row.get("最高", 0)),
                    "low": float(row.get("最低", 0)),
                    "open": float(row.get("今开", 0)),
                    "prev_close": float(row.get("昨收", 0)),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data_source": "AKShare-东方财富"
                })

            cache.set(cache_key, results, self.settings.QUOTE_CACHE_TTL)

            # 筛选目标股票
            clean_symbols = [s.replace("sh", "").replace("sz", "").replace("SH", "").replace("SZ", "") for s in symbols]
            return [q for q in results if q.get("symbol") in clean_symbols]

        except Exception as e:
            logger.error(f"批量获取行情失败: {e}")
            return [{"symbol": s, "error": str(e)} for s in symbols]

    def get_historical_kline(self, symbol: str, period: str = "daily", days: int = 60) -> List[Dict[str, Any]]:
        """获取历史K线数据

        Args:
            symbol: 股票代码
            period: 周期 daily/weekly/monthly
            days: 获取天数
        """
        cache_key = f"kline:{symbol}:{period}:{days}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            ak = self._get_ak()
            clean_symbol = symbol.replace("sh", "").replace("sz", "").replace("SH", "").replace("SZ", "")

            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

            period_map = {"daily": "101", "weekly": "102", "monthly": "103"}
            df = ak.stock_zh_a_hist(
                symbol=clean_symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )

            results = []
            for _, row in df.iterrows():
                results.append({
                    "date": str(row.get("日期", "")),
                    "open": float(row.get("开盘", 0)),
                    "close": float(row.get("收盘", 0)),
                    "high": float(row.get("最高", 0)),
                    "low": float(row.get("最低", 0)),
                    "volume": float(row.get("成交量", 0)),
                    "turnover": float(row.get("成交额", 0)),
                    "change_pct": float(row.get("涨跌幅", 0)),
                })

            cache.set(cache_key, results, self.settings.KLINE_CACHE_TTL)
            return results

        except Exception as e:
            logger.error(f"获取股票 {symbol} K线数据失败: {e}")
            return [{"symbol": symbol, "error": str(e)}]

    def get_sector_performance(self, sector_name: str) -> Dict[str, Any]:
        """获取板块行情表现

        Args:
            sector_name: 板块名称，如 "半导体"、"人工智能"
        """
        cache_key = f"sector:{sector_name}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        def _fetch():
            ak = self._get_ak()
            df = ak.stock_board_industry_name_em()
            row = df[df["板块名称"] == sector_name]

            if row.empty:
                row = df[df["板块名称"].str.contains(sector_name, na=False)]

            if row.empty:
                return {"sector": sector_name, "error": "未找到该板块"}

            row_data = row.iloc[0]
            return {
                "sector": sector_name,
                "change_pct": float(row_data.get("涨跌幅", 0)),
                "change_amount": float(row_data.get("涨跌额", 0)),
                "total_mv": float(row_data.get("总市值", 0)) if row_data.get("总市值") else 0,
                "turnover": float(row_data.get("成交额", 0)) if row_data.get("成交额") else 0,
                "leading_stock": str(row_data.get("领涨股票", "")) if row_data.get("领涨股票") else "",
                "leading_pct": float(row_data.get("领涨股票涨跌幅", 0)) if row_data.get("领涨股票涨跌幅") else 0,
                "stock_count": int(row_data.get("上涨家数", 0)) + int(row_data.get("下跌家数", 0)),
                "up_count": int(row_data.get("上涨家数", 0)) if row_data.get("上涨家数") else 0,
                "down_count": int(row_data.get("下跌家数", 0)) if row_data.get("下跌家数") else 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": "AKShare-东方财富"
            }

        result = _run_with_timeout(_fetch, default_return={"sector": sector_name, "error": "请求超时"})

        if result and "error" not in result:
            cache.set(cache_key, result, self.settings.QUOTE_CACHE_TTL)

        return result if result else {"sector": sector_name, "error": "请求失败"}

    def get_sector_top_stocks(self, sector_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取板块成分股排行"""
        cache_key = f"sector_stocks:{sector_name}:{limit}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        def _fetch():
            ak = self._get_ak()
            df = ak.stock_board_industry_cons_em(symbol=sector_name)

            if "涨跌幅" in df.columns:
                df = df.sort_values("涨跌幅", ascending=False)

            results = []
            for _, row in df.head(limit).iterrows():
                results.append({
                    "symbol": str(row.get("代码", "")),
                    "name": str(row.get("名称", "")),
                    "price": float(row.get("最新价", 0)) if row.get("最新价") else 0,
                    "change_pct": float(row.get("涨跌幅", 0)) if row.get("涨跌幅") else 0,
                })
            return results

        result = _run_with_timeout(_fetch, default_return=[])

        if result:
            cache.set(cache_key, result, self.settings.QUOTE_CACHE_TTL)

        return result if result else []

    def get_top_gainers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取涨幅排行"""
        cache_key = f"top_gainers:{limit}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            ak = self._get_ak()
            df = ak.stock_zh_a_spot_em()
            df = df.sort_values("涨跌幅", ascending=False)

            results = []
            for _, row in df.head(limit).iterrows():
                results.append({
                    "symbol": str(row.get("代码", "")),
                    "name": str(row.get("名称", "")),
                    "price": float(row.get("最新价", 0)),
                    "change_pct": float(row.get("涨跌幅", 0)),
                })

            cache.set(cache_key, results, self.settings.QUOTE_CACHE_TTL)
            return results

        except Exception as e:
            logger.error(f"获取涨幅排行失败: {e}")
            return []

    def get_top_losers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取跌幅排行"""
        cache_key = f"top_losers:{limit}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            ak = self._get_ak()
            df = ak.stock_zh_a_spot_em()
            df = df.sort_values("涨跌幅", ascending=True)

            results = []
            for _, row in df.head(limit).iterrows():
                results.append({
                    "symbol": str(row.get("代码", "")),
                    "name": str(row.get("名称", "")),
                    "price": float(row.get("最新价", 0)),
                    "change_pct": float(row.get("涨跌幅", 0)),
                })

            cache.set(cache_key, results, self.settings.QUOTE_CACHE_TTL)
            return results

        except Exception as e:
            logger.error(f"获取跌幅排行失败: {e}")
            return []

    def search_stock(self, keyword: str) -> List[Dict[str, Any]]:
        """搜索股票（按名称或代码模糊匹配）"""
        try:
            ak = self._get_ak()
            df = ak.stock_zh_a_spot_em()

            mask = df["名称"].str.contains(keyword, na=False) | df["代码"].str.contains(keyword, na=False)
            matched = df[mask].head(10)

            results = []
            for _, row in matched.iterrows():
                results.append({
                    "symbol": str(row.get("代码", "")),
                    "name": str(row.get("名称", "")),
                    "price": float(row.get("最新价", 0)),
                    "change_pct": float(row.get("涨跌幅", 0)),
                })
            return results

        except Exception as e:
            logger.error(f"搜索股票 {keyword} 失败: {e}")
            return []
