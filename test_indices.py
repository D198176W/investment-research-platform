"""测试指数接口"""
import sys, time
sys.path.insert(0, 'backend')
from data_fetcher.market_indices import MarketIndicesFetcher

fetcher = MarketIndicesFetcher()
start = time.time()
result = fetcher.get_major_indices()
elapsed = time.time() - start
print(f'耗时: {elapsed:.1f}秒')
for idx in result:
    name = idx.get('name', '?')
    if 'error' not in idx:
        price = idx.get('price', 0)
        pct = idx.get('change_pct', 0)
        src = idx.get('data_source', '')
        print(f'  {name}: {price} ({pct}%) [{src}]')
    else:
        err = idx.get('error', '')
        print(f'  {name}: ERROR - {err}')
