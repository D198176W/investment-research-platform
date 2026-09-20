"""大盘指数与宏观数据抓取 - 基于 AKShare（多数据源自动切换）"""
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, List, Any, Callable
from datetime import datetime

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
    """在独立线程中执行函数，支持超时控制"""
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


class MarketIndexFetcher:
    """大盘指数与宏观数据抓取器（多数据源自动切换：东方财富 → 新浪财经）"""

    MAJOR_INDICES = {
        "上证指数": "000001",
        "深证成指": "399001",
        "创业板指": "399006",
        "科创50": "000688",
        "沪深300": "000300",
        "中证500": "000905",
    }

    # 新浪指数代码映射
    SINA_INDEX_CODES = {
        "上证指数": "sh000001",
        "深证成指": "sz399001",
        "创业板指": "sz399006",
        "科创50": "sh000688",
        "沪深300": "sh000300",
        "中证500": "sh000905",
    }

    def __init__(self):
        self.settings = get_data_fetcher_settings()

    def _get_ak(self):
        ak = _safe_import_akshare()
        if ak is None:
            raise RuntimeError("AKShare 未安装")
        return ak

    def get_major_indices(self) -> List[Dict[str, Any]]:
        """获取主要指数实时行情（自动切换：东方财富 → 新浪财经）"""
        cache_key = "indices:major"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        # 数据源1: 东方财富
        def _fetch_eastmoney():
            ak = self._get_ak()
            df = ak.stock_zh_index_spot_em()
            results = []
            for name, code in self.MAJOR_INDICES.items():
                row = df[df["代码"] == code]
                if not row.empty:
                    row_data = row.iloc[0]
                    results.append({
                        "name": name,
                        "code": code,
                        "price": float(row_data.get("最新价", 0)),
                        "change_pct": float(row_data.get("涨跌幅", 0)),
                        "change_amount": float(row_data.get("涨跌额", 0)),
                        "volume": float(row_data.get("成交量", 0)) if row_data.get("成交量") else 0,
                        "turnover": float(row_data.get("成交额", 0)) if row_data.get("成交额") else 0,
                        "amplitude": float(row_data.get("振幅", 0)) if row_data.get("振幅") else 0,
                    })
                else:
                    results.append({"name": name, "code": code, "error": "未获取到数据"})
            for r in results:
                r["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                r["data_source"] = "AKShare-东方财富"
            return results

        # 数据源2: 新浪财经（通过 stock_zh_index_daily 获取最近一天数据）
        def _fetch_sina():
            ak = self._get_ak()
            results = []
            for name, code in self.SINA_INDEX_CODES.items():
                try:
                    df = ak.stock_zh_index_daily(symbol=code)
                    if df.empty:
                        results.append({"name": name, "code": code, "error": "未获取到数据"})
                        continue
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else latest
                    close = float(latest.get("close", 0))
                    prev_close = float(prev.get("close", 0))
                    change_pct = round((close - prev_close) / prev_close * 100, 2) if prev_close else 0
                    results.append({
                        "name": name,
                        "code": code,
                        "price": close,
                        "change_pct": change_pct,
                        "change_amount": round(close - prev_close, 2),
                        "volume": 0,
                        "turnover": 0,
                        "amplitude": 0,
                    })
                except Exception as e:
                    results.append({"name": name, "code": code, "error": str(e)[:50]})
            for r in results:
                r["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                r["data_source"] = "AKShare-新浪财经"
            return results

        # 先尝试东方财富，失败切换新浪
        result = _run_with_timeout(_fetch_eastmoney, default_return=None)
        if result is None:
            logger.info("东方财富指数接口失败，切换到新浪财经...")
            result = _run_with_timeout(_fetch_sina, timeout_seconds=10, default_return=[{"error": "请求超时"}])

        if result and not (len(result) == 1 and result[0].get("error") == "请求超时"):
            cache.set(cache_key, result, self.settings.INDICES_CACHE_TTL)

        return result if result else [{"error": "请求失败"}]

    def get_market_overview(self) -> Dict[str, Any]:
        """获取市场概览（涨跌家数、成交额等，自动切换数据源）"""
        cache_key = "market:overview"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        # 数据源1: 东方财富
        def _fetch_eastmoney():
            ak = self._get_ak()
            df = ak.stock_zh_a_spot_em()
            total = len(df)
            up_count = len(df[df["涨跌幅"] > 0])
            down_count = len(df[df["涨跌幅"] < 0])
            flat_count = total - up_count - down_count
            total_turnover = df["成交额"].sum() if "成交额" in df.columns else 0
            avg_change = df["涨跌幅"].mean() if "涨跌幅" in df.columns else 0
            limit_up = len(df[df["涨跌幅"] >= 9.9])
            limit_down = len(df[df["涨跌幅"] <= -9.9])
            return {
                "total_stocks": total,
                "up_count": up_count,
                "down_count": down_count,
                "flat_count": flat_count,
                "limit_up_count": limit_up,
                "limit_down_count": limit_down,
                "total_turnover": float(total_turnover),
                "avg_change_pct": round(float(avg_change), 2),
                "up_ratio": round(up_count / total * 100, 1) if total > 0 else 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": "AKShare-东方财富"
            }

        # 数据源2: 新浪财经
        def _fetch_sina():
            ak = self._get_ak()
            df = ak.stock_zh_a_spot()
            total = len(df)
            up_count = len(df[df["涨跌幅"] > 0])
            down_count = len(df[df["涨跌幅"] < 0])
            flat_count = total - up_count - down_count
            total_turnover = df["成交额"].sum() if "成交额" in df.columns else 0
            avg_change = df["涨跌幅"].mean() if "涨跌幅" in df.columns else 0
            return {
                "total_stocks": total,
                "up_count": up_count,
                "down_count": down_count,
                "flat_count": flat_count,
                "limit_up_count": 0,
                "limit_down_count": 0,
                "total_turnover": float(total_turnover),
                "avg_change_pct": round(float(avg_change), 2),
                "up_ratio": round(up_count / total * 100, 1) if total > 0 else 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": "AKShare-新浪财经"
            }

        result = _run_with_timeout(_fetch_eastmoney, default_return=None)
        if result is None:
            # 新浪 stock_zh_a_spot 需要抓取70页数据太慢(60秒+)，不适合作为备用
            # 直接返回错误，让 LLM 用自身知识补充
            logger.info("东方财富市场概览接口失败，跳过（新浪备用太慢）")
            return {"error": "市场概览数据不可用"}

        if isinstance(result, dict) and "error" not in result:
            cache.set(cache_key, result, self.settings.INDICES_CACHE_TTL)

        return result

    def get_capital_flow(self) -> Dict[str, Any]:
        """获取市场资金流向"""
        cache_key = "market:capital_flow"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        def _fetch():
            ak = self._get_ak()
            df = ak.stock_market_fund_flow()

            if df.empty:
                return {"error": "无资金流向数据"}

            latest = df.iloc[-1]
            return {
                "date": str(latest.get("日期", "")),
                "north_flow": float(latest.get("北向资金", 0)) if "北向资金" in df.columns else 0,
                "main_flow": float(latest.get("主力资金", 0)) if "主力资金" in df.columns else 0,
                "retail_flow": float(latest.get("散户资金", 0)) if "散户资金" in df.columns else 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": "AKShare-东方财富"
            }

        result = _run_with_timeout(_fetch, default_return={"error": "请求超时"})

        if result and "error" not in result:
            cache.set(cache_key, result, self.settings.INDICES_CACHE_TTL)

        return result if result else {"error": "请求失败"}

    def get_sector_list(self) -> List[Dict[str, Any]]:
        """获取所有行业板块列表"""
        cache_key = "sector:list"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        def _fetch():
            ak = self._get_ak()
            df = ak.stock_board_industry_name_em()

            results = []
            for _, row in df.iterrows():
                results.append({
                    "name": str(row.get("板块名称", "")),
                    "change_pct": float(row.get("涨跌幅", 0)),
                    "total_mv": float(row.get("总市值", 0)) if row.get("总市值") else 0,
                    "turnover": float(row.get("成交额", 0)) if row.get("成交额") else 0,
                    "up_count": int(row.get("上涨家数", 0)) if row.get("上涨家数") else 0,
                    "down_count": int(row.get("下跌家数", 0)) if row.get("下跌家数") else 0,
                })
            return results

        result = _run_with_timeout(_fetch, default_return=[])

        if result:
            cache.set(cache_key, result, self.settings.INDICES_CACHE_TTL)

        return result if result else []


# 保持向后兼容的别名
MarketIndicesFetcher = MarketIndexFetcher
