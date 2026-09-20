"""财经新闻数据抓取 - 基于 AKShare"""
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, List, Any, Callable

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


class NewsFetcher:
    """财经新闻数据抓取器"""

    def __init__(self):
        self.settings = get_data_fetcher_settings()

    def _get_ak(self):
        ak = _safe_import_akshare()
        if ak is None:
            raise RuntimeError("AKShare 未安装")
        return ak

    def get_financial_news(self, keyword: str = "", limit: int = 10) -> List[Dict[str, Any]]:
        """获取财经新闻"""
        cache_key = f"news:{keyword}:{limit}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        def _fetch():
            ak = self._get_ak()
            if keyword:
                df = ak.stock_news_em(symbol=keyword)
            else:
                df = ak.news_cctv(date="")

            results = []
            for _, row in df.head(limit).iterrows():
                # 兼容不同列名：东方财富用"新闻标题"，央视用"title"
                title = str(row.get("新闻标题", row.get("标题", row.get("title", ""))))
                content = str(row.get("新闻内容", row.get("内容", row.get("content", ""))))[:200]
                source = str(row.get("文章来源", row.get("来源", row.get("source", "东方财富"))))
                publish_time = str(row.get("发布时间", row.get("datetime", "")))
                url = str(row.get("新闻链接", row.get("链接", row.get("url", ""))))
                results.append({
                    "title": title,
                    "content": content,
                    "source": source,
                    "publish_time": publish_time,
                    "url": url,
                })
            return results

        result = _run_with_timeout(_fetch, default_return=[])

        if result:
            cache.set(cache_key, result, self.settings.NEWS_CACHE_TTL)
            return result

        # 降级方案
        return self._fallback_news(keyword, limit)

    def get_stock_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取个股相关新闻"""
        cache_key = f"stock_news:{symbol}:{limit}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        def _fetch():
            ak = self._get_ak()
            clean_symbol = symbol.replace("sh", "").replace("sz", "").replace("SH", "").replace("SZ", "")
            df = ak.stock_news_em(symbol=clean_symbol)

            results = []
            for _, row in df.head(limit).iterrows():
                title = str(row.get("新闻标题", row.get("标题", row.get("title", ""))))
                content = str(row.get("新闻内容", row.get("内容", row.get("content", ""))))[:200]
                source = str(row.get("文章来源", row.get("来源", row.get("source", ""))))
                publish_time = str(row.get("发布时间", row.get("datetime", "")))
                url = str(row.get("新闻链接", row.get("链接", row.get("url", ""))))
                results.append({
                    "title": title,
                    "content": content,
                    "source": source,
                    "publish_time": publish_time,
                    "url": url,
                })
            return results

        result = _run_with_timeout(_fetch, default_return=[])

        if result:
            cache.set(cache_key, result, self.settings.NEWS_CACHE_TTL)

        return result if result else []

    def get_macro_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取宏观经济新闻"""
        cache_key = f"macro_news:{limit}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        def _fetch():
            ak = self._get_ak()
            df = ak.news_cctv(date="")

            results = []
            for _, row in df.head(limit).iterrows():
                results.append({
                    "title": str(row.get("title", "")),
                    "content": str(row.get("content", ""))[:200],
                    "source": "央视新闻",
                    "publish_time": str(row.get("datetime", "")),
                })
            return results

        result = _run_with_timeout(_fetch, default_return=[])

        if result:
            cache.set(cache_key, result, self.settings.NEWS_CACHE_TTL)
            return result

        return self._fallback_news("", limit)

    def _fallback_news(self, keyword: str, limit: int) -> List[Dict[str, Any]]:
        """降级方案：当主要新闻接口失败时使用"""
        def _fetch():
            ak = self._get_ak()
            df = ak.news_cctv(date="")

            results = []
            for _, row in df.head(limit).iterrows():
                results.append({
                    "title": str(row.get("title", "")),
                    "content": str(row.get("content", ""))[:200],
                    "source": "央视新闻(降级)",
                    "publish_time": str(row.get("datetime", "")),
                })
            return results

        result = _run_with_timeout(_fetch, default_return=[])
        return result if result else []
