"""Agent Service 配置"""
from pydantic_settings import BaseSettings
from functools import lru_cache
import os

# .env 固定定位到项目根目录（无论从哪个 cwd 启动都能加载）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_FILE = os.path.join(_PROJECT_ROOT, ".env")


class AgentSettings(BaseSettings):
    """Agent 服务配置"""
    MAX_RETRIES: int = 3
    DEFAULT_MODEL_NAME: str = "qwen-plus"
    DEFAULT_API_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # v2: Agentic 能力开关
    USE_AGENTIC_PERCEPTION: bool = True    # 感知阶段是否启用工具调用循环
    AGENTIC_MAX_ROUNDS: int = 8            # 工具调用循环最大轮数
    MAX_REFLECTION_LOOPS: int = 1          # Reflector 最大反思回灌次数
    USE_MCP_EXECUTOR: bool = False         # 工具执行是否走 MCP Server（stdio 子进程），默认进程内执行

    # v2: 记忆与上下文
    MEMORY_MAX_ITEMS: int = 200            # 长期记忆容量上限
    CONTEXT_MAX_TOKENS: int = 30000        # 上下文窗口预算
    CONTEXT_RESERVE_OUTPUT: int = 2000     # 为输出预留的 token

    # v2: 多模型路由（按任务级别）
    HEAVY_MODEL: str = "qwen-max"          # 重推理任务
    LIGHT_MODEL: str = "qwen-turbo"        # 轻量任务（摘要/分类）
    FALLBACK_MODEL: str = "qwen-plus"      # 降级模型
    REVIEWER_MODEL: str = "qwen-max"       # 独立风控审核官模型（分析-审核双角色）

    class Config:
        env_file = ENV_FILE
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_agent_settings() -> AgentSettings:
    return AgentSettings()

