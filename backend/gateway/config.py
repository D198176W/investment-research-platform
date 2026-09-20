"""FastAPI Gateway 配置"""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache
import os

# .env 固定定位到项目根目录（无论从哪个 cwd 启动都能加载）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_FILE = os.path.join(_PROJECT_ROOT, ".env")


class Settings(BaseSettings):
    """应用配置"""
    APP_NAME: str = "智能投研平台"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # API 网关配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # 内部服务地址（微服务架构预留）
    AGENT_SERVICE_URL: str = "http://localhost:8001"
    DATA_SERVICE_URL: str = "http://localhost:8002"
    REPORT_SERVICE_URL: str = "http://localhost:8003"

    # LLM 配置
    DASHSCOPE_API_KEY: str = ""
    DASHSCOPE_API_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DEFAULT_MODEL_NAME: str = "qwen-plus"

    # 安全配置
    CORS_ORIGINS: str = "http://localhost:8501,http://127.0.0.1:8501"  # 逗号分隔，*"放开
    RATE_LIMIT_PER_MINUTE: int = 120        # 每 IP 每分钟请求上限
    RATE_LIMIT_RESEARCH_PER_MINUTE: int = 10  # 创建研究任务（重资源）每 IP 每分钟上限

    class Config:
        env_file = ENV_FILE
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """获取缓存的配置实例"""
    return Settings()
