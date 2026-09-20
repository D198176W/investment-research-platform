"""数据采集服务配置"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Dict
import os

# .env 固定定位到项目根目录（无论从哪个 cwd 启动都能加载）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_FILE = os.path.join(_PROJECT_ROOT, ".env")


class DataFetcherSettings(BaseSettings):
    """数据采集配置"""
    # 缓存过期时间（秒）
    QUOTE_CACHE_TTL: int = 30          # 实时行情缓存30秒
    KLINE_CACHE_TTL: int = 3600        # K线数据缓存1小时
    FINANCIAL_CACHE_TTL: int = 86400   # 财务数据缓存24小时
    NEWS_CACHE_TTL: int = 600          # 新闻缓存10分钟
    INDICES_CACHE_TTL: int = 60        # 指数缓存60秒

    # 请求限制
    MAX_REQUEST_RETRIES: int = 3       # 最大重试次数
    REQUEST_TIMEOUT: int = 15          # 请求超时（秒）

    # 行业-板块映射（AKShare 板块名称）
    INDUSTRY_SECTOR_MAP: Dict[str, str] = {
        "半导体": "半导体",
        "人工智能": "人工智能",
        "新能源": "新能源",
        "生物医药": "生物医药",
        "消费电子": "消费电子",
        "金融科技": "金融科技",
        "智能制造": "智能制造",
        "航空航天": "航空航天",
        "新材料": "新材料",
        "互联网": "互联网",
        "汽车": "汽车整车",
        "传媒娱乐": "文化传媒",
    }

    # 主题-关键词映射（用于新闻搜索）
    TOPIC_KEYWORD_MAP: Dict[str, str] = {
        "人工智能芯片市场": "AI芯片",
        "新能源汽车产业链": "新能源汽车",
        "创新药研发管线": "创新药",
        "跨境电商出海机遇": "跨境电商",
        "低空经济产业链": "低空经济",
        "储能技术商业化": "储能",
        "人形机器人产业": "人形机器人",
        "消费级AR/VR市场": "AR VR",
        "半导体设备国产化": "半导体设备",
        "金融科技监管趋势": "金融科技",
        "碳中和绿色投资": "碳中和",
        "量子计算商用化": "量子计算",
    }

    class Config:
        env_file = ENV_FILE
        env_file_encoding = "utf-8"
        extra = "ignore"  # 忽略 .env 中不属于本配置的字段


@lru_cache()
def get_data_fetcher_settings() -> DataFetcherSettings:
    """获取缓存的数据采集配置实例"""
    return DataFetcherSettings()
