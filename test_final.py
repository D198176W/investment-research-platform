"""测试完整修复后的数据获取和Prompt注入"""
import sys, time
sys.path.insert(0, 'backend')

# 清除缓存
from data_fetcher.cache_manager import cache
cache.clear()

from agent_service.workflow import _fetch_realtime_data, _format_realtime_data_for_prompt

print('=== 测试完整数据获取流程 ===')
start = time.time()
data = _fetch_realtime_data(topic='创新药研发管线', industry='生物医药')
elapsed = time.time() - start

prompt_text = _format_realtime_data_for_prompt(data)
print(f'\n数据获取耗时: {elapsed:.1f}秒')
print(f'成功获取的数据源: {data.get("data_sources", [])}')
print(f'获取失败(仅日志): {data.get("fetch_errors", [])}')
print(f'\n=== 注入Prompt的文本 ===')
print(prompt_text)
print(f'\n文本长度: {len(prompt_text)}字符')
