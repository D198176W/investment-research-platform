"""数据缓存管理器 - 基于 TTL 的内存缓存"""
import time
import threading
import logging
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)


class CacheItem:
    """缓存项"""
    def __init__(self, data: Any, ttl: int):
        self.data = data
        self.expire_at = time.time() + ttl

    def is_expired(self) -> bool:
        return time.time() > self.expire_at


class CacheManager:
    """线程安全的 TTL 内存缓存管理器"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._cache: Dict[str, CacheItem] = {}
                    cls._instance._cache_lock = threading.Lock()
        return cls._instance

    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据，过期返回 None"""
        with self._cache_lock:
            item = self._cache.get(key)
            if item is None:
                return None
            if item.is_expired():
                del self._cache[key]
                return None
            return item.data

    def set(self, key: str, data: Any, ttl: int):
        """设置缓存数据"""
        with self._cache_lock:
            self._cache[key] = CacheItem(data, ttl)

    def delete(self, key: str):
        """删除缓存项"""
        with self._cache_lock:
            self._cache.pop(key, None)

    def clear(self):
        """清空所有缓存"""
        with self._cache_lock:
            self._cache.clear()

    def cleanup(self):
        """清理过期缓存"""
        with self._cache_lock:
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            for k in expired_keys:
                del self._cache[k]
            if expired_keys:
                logger.info(f"清理过期缓存: {len(expired_keys)} 项")

    def stats(self) -> dict:
        """获取缓存统计"""
        with self._cache_lock:
            total = len(self._cache)
            expired = sum(1 for v in self._cache.values() if v.is_expired())
            return {"total": total, "expired": expired, "active": total - expired}


# 全局缓存实例
cache = CacheManager()
