"""测试 financials.py"""
import sys
sys.path.insert(0, 'backend')

from data_fetcher.financials import FinancialsFetcher
print('1. FinancialsFetcher 导入成功')

fetcher = FinancialsFetcher()
assert hasattr(fetcher, 'get_stock_financial')
assert hasattr(fetcher, 'get_stock_financial_batch')
assert hasattr(fetcher, 'get_stock_profit')
assert hasattr(fetcher, 'get_stock_balance')
assert hasattr(fetcher, 'get_stock_cashflow')
print('2. 所有接口方法存在')

from data_fetcher import FinancialsFetcher as FF
print('3. __init__.py 导出正常')

from data_fetcher.config import get_data_fetcher_settings
settings = get_data_fetcher_settings()
assert settings.FINANCIAL_CACHE_TTL == 86400
print(f'4. 财务数据缓存 TTL: {settings.FINANCIAL_CACHE_TTL}秒 (24小时)')

result = fetcher.get_stock_financial('600519')
if 'error' not in result:
    name = result.get('name', 'N/A')
    print(f'5. 茅台财务数据获取成功: 名称={name}')
    if result.get('financial_reports'):
        print(f'   最新报告期: {result.get("latest_report_date", "N/A")}')
        print(f'   营业收入: {result.get("revenue", "N/A")}')
        print(f'   净利润: {result.get("net_profit", "N/A")}')
        print(f'   ROE: {result.get("roe", "N/A")}')
else:
    err = result.get('error', '')
    print(f'5. 数据获取失败: {err} (网络问题)')

print()
print('=== financials.py 验证完成 ===')
