"""测试 AKShare 超时控制是否生效"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from data_fetcher.stock_data import StockDataFetcher
from data_fetcher.news_fetcher import NewsFetcher
from data_fetcher.market_indices import MarketIndicesFetcher
from agent_service.workflow import _fetch_realtime_data
import time

print("=" * 60)
print("测试 AKShare 超时控制")
print("=" * 60)

# 测试1: 板块行情（带超时）
print("\n[测试1] 获取板块行情（预期8秒内返回）...")
start = time.time()
fetcher = StockDataFetcher()
result = fetcher.get_sector_performance("半导体")
elapsed = time.time() - start
print(f"  耗时: {elapsed:.1f}秒")
print(f"  结果: {result}")

# 测试2: 新闻（带超时）
print("\n[测试2] 获取财经新闻（预期8秒内返回）...")
start = time.time()
news_fetcher = NewsFetcher()
result = news_fetcher.get_financial_news(keyword="生物医药", limit=3)
elapsed = time.time() - start
print(f"  耗时: {elapsed:.1f}秒")
print(f"  结果条数: {len(result)}")
if result:
    print(f"  第一条: {result[0].get('title', 'N/A')[:50]}...")

# 测试3: 指数（带超时）
print("\n[测试3] 获取主要指数（预期8秒内返回）...")
start = time.time()
indices_fetcher = MarketIndicesFetcher()
result = indices_fetcher.get_major_indices()
elapsed = time.time() - start
print(f"  耗时: {elapsed:.1f}秒")
print(f"  结果: {result}")

# 测试4: 并发数据获取（workflow中的函数）
print("\n[测试4] 并发获取实时数据（预期15秒内返回）...")
start = time.time()
result = _fetch_realtime_data(topic="创新药研发管线", industry="生物医药")
elapsed = time.time() - start
print(f"  耗时: {elapsed:.1f}秒")
print(f"  数据源: {result.get('data_sources', [])}")
print(f"  错误: {result.get('fetch_errors', [])}")

print("\n" + "=" * 60)
print("测试完成！如果每项都在10秒内返回，说明超时控制生效。")
print("=" * 60)
