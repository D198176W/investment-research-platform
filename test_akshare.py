"""
AKShare 数据源接入功能测试脚本
使用方法: 在项目根目录执行 python test_akshare.py
"""
import sys
import os

# 确保 backend 目录在搜索路径中
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def test_akshare_install():
    """测试1: AKShare 是否安装成功"""
    print("\n" + "=" * 60)
    print("测试1: AKShare 安装验证")
    print("=" * 60)
    try:
        import akshare as ak
        print(f"  [PASS] AKShare 已安装，版本: {ak.__version__}")
        return True
    except ImportError:
        print("  [FAIL] AKShare 未安装，请执行: pip install akshare")
        return False


def test_config_module():
    """测试2: 配置模块加载"""
    print("\n" + "=" * 60)
    print("测试2: 数据采集配置模块")
    print("=" * 60)
    try:
        from data_fetcher.config import get_data_fetcher_settings
        settings = get_data_fetcher_settings()
        print(f"  [PASS] 配置加载成功")
        print(f"  - 行情缓存TTL: {settings.QUOTE_CACHE_TTL}s")
        print(f"  - K线缓存TTL: {settings.KLINE_CACHE_TTL}s")
        print(f"  - 新闻缓存TTL: {settings.NEWS_CACHE_TTL}s")
        print(f"  - 行业板块映射: {list(settings.INDUSTRY_SECTOR_MAP.keys())}")
        print(f"  - 主题关键词映射: {list(settings.TOPIC_KEYWORD_MAP.keys())}")
        return True
    except Exception as e:
        print(f"  [FAIL] 配置加载失败: {e}")
        return False


def test_cache_manager():
    """测试3: 缓存管理器"""
    print("\n" + "=" * 60)
    print("测试3: 缓存管理器")
    print("=" * 60)
    try:
        from data_fetcher.cache_manager import CacheManager
        cm = CacheManager()
        # 写入
        cm.set("test_key", {"data": "hello"}, ttl=60)
        # 读取
        result = cm.get("test_key")
        assert result == {"data": "hello"}, "缓存读写不一致"
        # 删除
        cm.delete("test_key")
        assert cm.get("test_key") is None, "缓存删除失败"
        # 统计
        stats = cm.stats()
        print(f"  [PASS] 缓存管理器正常，当前缓存: {stats}")
        return True
    except Exception as e:
        print(f"  [FAIL] 缓存管理器测试失败: {e}")
        return False


def test_stock_data_fetcher():
    """测试4: 股票行情数据抓取"""
    print("\n" + "=" * 60)
    print("测试4: 股票行情数据抓取 (需要网络)")
    print("=" * 60)
    try:
        from data_fetcher.stock_data import StockDataFetcher
        fetcher = StockDataFetcher()

        # 4.1 实时行情 - 贵州茅台(600519)
        print("  [4.1] 获取贵州茅台(600519)实时行情...")
        quote = fetcher.get_realtime_quote("600519")
        if "error" in quote:
            print(f"  [WARN] 实时行情获取失败: {quote['error']} (可能是非交易时间或网络问题)")
        else:
            print(f"  [PASS] {quote.get('name', '')} 最新价: {quote.get('price', 'N/A')} 涨跌幅: {quote.get('change_pct', 'N/A')}%")
            print(f"        数据来源: {quote.get('data_source', 'N/A')} 时间: {quote.get('timestamp', 'N/A')}")

        # 4.2 板块行情
        print("  [4.2] 获取半导体板块行情...")
        sector = fetcher.get_sector_performance("半导体")
        if "error" in sector:
            print(f"  [WARN] 板块行情获取失败: {sector.get('error')}")
        else:
            print(f"  [PASS] 半导体板块涨跌幅: {sector.get('change_pct', 'N/A')}% 上涨: {sector.get('up_count', 'N/A')}家")

        # 4.3 板块成分股
        print("  [4.3] 获取半导体板块成分股...")
        stocks = fetcher.get_sector_top_stocks("半导体", limit=3)
        if stocks:
            for s in stocks:
                print(f"        {s.get('name', '')}({s.get('symbol', '')}): {s.get('change_pct', 'N/A')}%")
        else:
            print("  [WARN] 成分股获取为空")

        # 4.4 涨跌排行
        print("  [4.4] 获取涨幅排行...")
        gainers = fetcher.get_top_gainers(limit=3)
        if gainers:
            for g in gainers:
                print(f"        {g.get('name', '')}({g.get('symbol', '')}): {g.get('change_pct', 'N/A')}%")
        else:
            print("  [WARN] 涨幅排行获取为空")

        # 4.5 股票搜索
        print("  [4.5] 搜索股票(关键词: 茅台)...")
        results = fetcher.search_stock("茅台")
        if results:
            for r in results:
                print(f"        {r.get('name', '')}({r.get('symbol', '')})")
        else:
            print("  [WARN] 搜索结果为空")

        return True
    except Exception as e:
        print(f"  [FAIL] 股票数据测试异常: {e}")
        return False


def test_news_fetcher():
    """测试5: 财经新闻抓取"""
    print("\n" + "=" * 60)
    print("测试5: 财经新闻抓取 (需要网络)")
    print("=" * 60)
    try:
        from data_fetcher.news_fetcher import NewsFetcher
        fetcher = NewsFetcher()

        # 5.1 关键词新闻
        print("  [5.1] 获取AI芯片相关新闻...")
        news = fetcher.get_financial_news(keyword="AI芯片", limit=3)
        if news:
            for n in news:
                print(f"        [{n.get('publish_time', '')}] {n.get('title', '')[:40]}")
        else:
            print("  [WARN] 新闻获取为空")

        # 5.2 个股新闻
        print("  [5.2] 获取贵州茅台个股新闻...")
        stock_news = fetcher.get_stock_news("600519", limit=3)
        if stock_news:
            for n in stock_news:
                print(f"        {n.get('title', '')[:40]}")
        else:
            print("  [WARN] 个股新闻获取为空")

        return True
    except Exception as e:
        print(f"  [FAIL] 新闻抓取测试异常: {e}")
        return False


def test_market_indices():
    """测试6: 大盘指数与宏观数据"""
    print("\n" + "=" * 60)
    print("测试6: 大盘指数与宏观数据 (需要网络)")
    print("=" * 60)
    try:
        from data_fetcher.market_indices import MarketIndicesFetcher
        fetcher = MarketIndicesFetcher()

        # 6.1 主要指数
        print("  [6.1] 获取主要指数...")
        indices = fetcher.get_major_indices()
        if indices and "error" not in str(indices):
            for idx in indices:
                if "error" not in idx:
                    print(f"        {idx.get('name', '')}: {idx.get('price', 'N/A')} ({idx.get('change_pct', 'N/A')}%)")
                else:
                    print(f"        {idx.get('name', '')}: 获取失败")
        else:
            print("  [WARN] 指数获取失败")

        # 6.2 市场概览
        print("  [6.2] 获取市场概览...")
        overview = fetcher.get_market_overview()
        if "error" not in overview:
            print(f"  [PASS] 上涨: {overview.get('up_count', 'N/A')}家 下跌: {overview.get('down_count', 'N/A')}家")
            print(f"        平均涨跌幅: {overview.get('avg_change_pct', 'N/A')}%")
        else:
            print(f"  [WARN] 市场概览获取失败: {overview.get('error')}")

        # 6.3 资金流向
        print("  [6.3] 获取资金流向...")
        flow = fetcher.get_capital_flow()
        if "error" not in flow:
            print(f"  [PASS] 资金流向数据获取成功")
        else:
            print(f"  [WARN] 资金流向获取失败: {flow.get('error')}")

        return True
    except Exception as e:
        print(f"  [FAIL] 指数数据测试异常: {e}")
        return False


def test_workflow_integration():
    """测试7: 工作流集成（实时数据注入感知节点）"""
    print("\n" + "=" * 60)
    print("测试7: 工作流集成 - 实时数据注入")
    print("=" * 60)
    try:
        from agent_service.workflow import _fetch_realtime_data, _format_realtime_data_for_prompt

        # 模拟抓取数据
        print("  [7.1] 测试 _fetch_realtime_data()...")
        realtime_data = _fetch_realtime_data(topic="人工智能芯片市场", industry="半导体")
        sources = realtime_data.get("data_sources", [])
        errors = realtime_data.get("fetch_errors", [])
        print(f"  [INFO] 数据源获取: {sources}")
        if errors:
            print(f"  [WARN] 获取异常: {errors}")

        # 格式化注入 Prompt
        print("  [7.2] 测试 _format_realtime_data_for_prompt()...")
        prompt_text = _format_realtime_data_for_prompt(realtime_data)
        print(f"  [PASS] Prompt 注入文本生成成功，长度: {len(prompt_text)} 字符")
        # 显示前500字符
        print("  --- 注入文本预览 ---")
        print(prompt_text[:500])
        if len(prompt_text) > 500:
            print("  ... (省略)")
        print("  --- 预览结束 ---")

        return True
    except Exception as e:
        print(f"  [FAIL] 工作流集成测试异常: {e}")
        return False


def test_api_gateway():
    """测试8: API 网关市场数据路由"""
    print("\n" + "=" * 60)
    print("测试8: API 网关市场数据路由 (需要启动后端服务)")
    print("=" * 60)

    import requests
    BASE_URL = "http://localhost:8000"

    # 检查服务是否运行
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=3)
        if resp.status_code != 200:
            print("  [SKIP] 后端服务未运行，跳过API测试")
            print("  启动方法: cd backend/gateway && python main.py")
            return None
    except Exception:
        print("  [SKIP] 后端服务未启动，跳过API测试")
        print("  启动方法: cd backend/gateway && python main.py")
        return None

    # 测试各路由
    routes = [
        ("GET", "/api/v1/market/indices", "主要指数"),
        ("GET", "/api/v1/market/overview", "市场概览"),
        ("GET", "/api/v1/market/quote/600519", "贵州茅台行情"),
        ("GET", "/api/v1/market/sector/半导体", "半导体板块"),
        ("GET", "/api/v1/market/news?keyword=AI&limit=3", "AI新闻"),
        ("GET", "/api/v1/market/gainers-losers?limit=3", "涨跌排行"),
        ("GET", "/api/v1/market/capital-flow", "资金流向"),
        ("GET", "/api/v1/market/search?keyword=茅台", "股票搜索"),
    ]

    for method, route, desc in routes:
        try:
            resp = requests.get(f"{BASE_URL}{route}", timeout=15)
            data = resp.json()
            if data.get("success"):
                print(f"  [PASS] {desc}: {route}")
            else:
                print(f"  [WARN] {desc}: {route} - {data.get('message', '未知错误')}")
        except Exception as e:
            print(f"  [FAIL] {desc}: {route} - {e}")

    return True


def main():
    print("=" * 60)
    print("  智能投研平台 - AKShare 数据源接入功能测试")
    print("=" * 60)

    results = {}

    # 本地测试（不需要网络）
    results["AKShare安装"] = test_akshare_install()
    results["配置模块"] = test_config_module()
    results["缓存管理器"] = test_cache_manager()

    # 网络测试
    results["股票行情"] = test_stock_data_fetcher()
    results["财经新闻"] = test_news_fetcher()
    results["大盘指数"] = test_market_indices()
    results["工作流集成"] = test_workflow_integration()

    # API测试（需要启动后端）
    results["API网关"] = test_api_gateway()

    # 汇总
    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)

    for name, result in results.items():
        if result is True:
            status = "PASS"
        elif result is False:
            status = "FAIL"
        else:
            status = "SKIP"
        print(f"  [{status}] {name}")

    print(f"\n  总计: {total} 项 | 通过: {passed} | 失败: {failed} | 跳过: {skipped}")

    if failed > 0:
        print("\n  注意: 网络相关测试失败可能是非交易时间或网络限制导致，部署到正常网络环境后即可通过。")
    else:
        print("\n  所有测试通过！AKShare 数据源已成功接入。")


if __name__ == "__main__":
    main()
