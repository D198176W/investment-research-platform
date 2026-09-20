"""财务报表数据抓取 - 基于 AKShare"""
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, Any, Callable, Optional
from datetime import datetime

from .cache_manager import cache
from .config import get_data_fetcher_settings

logger = logging.getLogger(__name__)

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


class FinancialsFetcher:
    """A股财务报表数据抓取器"""

    def __init__(self):
        self.settings = get_data_fetcher_settings()

    def _get_ak(self):
        ak = _safe_import_akshare()
        if ak is None:
            raise RuntimeError("AKShare 未安装")
        return ak

    def get_stock_financial(self, symbol: str) -> Dict[str, Any]:
        """获取个股核心财务指标

        Args:
            symbol: 股票代码，如 "600519"

        Returns:
            包含财务指标的字典
        """
        cache_key = f"financial:{symbol}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        def _fetch():
            ak = self._get_ak()
            clean_symbol = symbol.replace("sh", "").replace("sz", "").replace("SH", "").replace("SZ", "")

            result = {
                "symbol": clean_symbol,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": "AKShare-东方财富",
            }

            # 1. 获取个股基本信息（总市值、流通市值、行业等）
            try:
                info_df = ak.stock_individual_info_em(symbol=clean_symbol)
                if not info_df.empty:
                    info_dict = dict(zip(info_df.iloc[:, 0], info_df.iloc[:, 1]))
                    result["name"] = str(info_dict.get("股票简称", ""))
                    result["industry"] = str(info_dict.get("行业", ""))
                    result["total_mv"] = str(info_dict.get("总市值", ""))
                    result["circ_mv"] = str(info_dict.get("流通市值", ""))
                    result["list_date"] = str(info_dict.get("上市时间", ""))
            except Exception as e:
                logger.warning(f"获取个股信息失败({clean_symbol}): {e}")

            # 2. 获取主要财务指标（按报告期）
            try:
                fin_df = ak.stock_financial_abstract_ths(symbol=clean_symbol)
                if not fin_df.empty:
                    # 取最近4期报告
                    recent = fin_df.head(4)
                    reports = []
                    for _, row in recent.iterrows():
                        report = {}
                        for col in fin_df.columns:
                            val = row.get(col)
                            if val is not None and str(val) != "nan":
                                report[col] = str(val)
                        reports.append(report)
                    result["financial_reports"] = reports
                    # 提取最新一期关键指标
                    if reports:
                        latest = reports[0]
                        result["latest_report_date"] = latest.get("报告期", "")
                        result["revenue"] = latest.get("营业总收入", "")
                        result["net_profit"] = latest.get("净利润", "")
                        result["roe"] = latest.get("净资产收益率", "")
                        result["gross_margin"] = latest.get("销售毛利率", "")
                        result["net_margin"] = latest.get("销售净利率", "")
                        result["debt_ratio"] = latest.get("资产负债率", "")
            except Exception as e:
                logger.warning(f"获取财务摘要失败({clean_symbol}): {e}")
                result["financial_reports"] = []

            return result

        result = _run_with_timeout(_fetch, timeout_seconds=10, default_return={"symbol": symbol, "error": "请求超时"})

        if result and isinstance(result, dict) and "error" not in result:
            cache.set(cache_key, result, self.settings.FINANCIAL_CACHE_TTL)

        return result if result else {"symbol": symbol, "error": "请求失败"}

    def get_stock_financial_batch(self, symbols: list) -> list:
        """批量获取个股财务指标

        Args:
            symbols: 股票代码列表

        Returns:
            财务指标字典列表
        """
        results = []
        for symbol in symbols:
            result = self.get_stock_financial(symbol)
            results.append(result)
        return results

    def get_stock_profit(self, symbol: str) -> Dict[str, Any]:
        """获取个股盈利能力数据

        Args:
            symbol: 股票代码

        Returns:
            盈利能力指标字典
        """
        cache_key = f"profit:{symbol}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        def _fetch():
            ak = self._get_ak()
            clean_symbol = symbol.replace("sh", "").replace("sz", "").replace("SH", "").replace("SZ", "")
            df = ak.stock_profit_sheet_by_report_em(symbol=clean_symbol)
            if df.empty:
                return {"symbol": clean_symbol, "error": "无盈利数据"}

            latest = df.iloc[0]
            return {
                "symbol": clean_symbol,
                "report_date": str(latest.get("报告期", "")),
                "revenue": str(latest.get("营业总收入", "")),
                "operating_profit": str(latest.get("营业利润", "")),
                "net_profit": str(latest.get("净利润", "")),
                "eps": str(latest.get("基本每股收益", "")),
                "diluted_eps": str(latest.get("稀释每股收益", "")),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": "AKShare-东方财富",
            }

        result = _run_with_timeout(_fetch, timeout_seconds=10, default_return={"symbol": symbol, "error": "请求超时"})

        if result and isinstance(result, dict) and "error" not in result:
            cache.set(cache_key, result, self.settings.FINANCIAL_CACHE_TTL)

        return result if result else {"symbol": symbol, "error": "请求失败"}

    def get_stock_balance(self, symbol: str) -> Dict[str, Any]:
        """获取个股资产负债表数据

        Args:
            symbol: 股票代码

        Returns:
            资产负债指标字典
        """
        cache_key = f"balance:{symbol}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        def _fetch():
            ak = self._get_ak()
            clean_symbol = symbol.replace("sh", "").replace("sz", "").replace("SH", "").replace("SZ", "")
            df = ak.stock_balance_sheet_by_report_em(symbol=clean_symbol)
            if df.empty:
                return {"symbol": clean_symbol, "error": "无资产负债数据"}

            latest = df.iloc[0]
            return {
                "symbol": clean_symbol,
                "report_date": str(latest.get("报告期", "")),
                "total_assets": str(latest.get("资产总计", "")),
                "total_liabilities": str(latest.get("负债合计", "")),
                "total_equity": str(latest.get("所有者权益合计", "")),
                "current_assets": str(latest.get("流动资产合计", "")),
                "current_liabilities": str(latest.get("流动负债合计", "")),
                "cash": str(latest.get("货币资金", "")),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": "AKShare-东方财富",
            }

        result = _run_with_timeout(_fetch, timeout_seconds=10, default_return={"symbol": symbol, "error": "请求超时"})

        if result and isinstance(result, dict) and "error" not in result:
            cache.set(cache_key, result, self.settings.FINANCIAL_CACHE_TTL)

        return result if result else {"symbol": symbol, "error": "请求失败"}

    def get_stock_cashflow(self, symbol: str) -> Dict[str, Any]:
        """获取个股现金流量表数据

        Args:
            symbol: 股票代码

        Returns:
            现金流指标字典
        """
        cache_key = f"cashflow:{symbol}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        def _fetch():
            ak = self._get_ak()
            clean_symbol = symbol.replace("sh", "").replace("sz", "").replace("SH", "").replace("SZ", "")
            df = ak.stock_cash_flow_sheet_by_report_em(symbol=clean_symbol)
            if df.empty:
                return {"symbol": clean_symbol, "error": "无现金流数据"}

            latest = df.iloc[0]
            return {
                "symbol": clean_symbol,
                "report_date": str(latest.get("报告期", "")),
                "operating_cashflow": str(latest.get("经营活动产生的现金流量净额", "")),
                "investing_cashflow": str(latest.get("投资活动产生的现金流量净额", "")),
                "financing_cashflow": str(latest.get("筹资活动产生的现金流量净额", "")),
                "net_cash_increase": str(latest.get("现金及现金等价物净增加额", "")),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": "AKShare-东方财富",
            }

        result = _run_with_timeout(_fetch, timeout_seconds=10, default_return={"symbol": symbol, "error": "请求超时"})

        if result and isinstance(result, dict) and "error" not in result:
            cache.set(cache_key, result, self.settings.FINANCIAL_CACHE_TTL)

        return result if result else {"symbol": symbol, "error": "请求失败"}
