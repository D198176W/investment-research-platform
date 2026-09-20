"""Pydantic 校验工具 —— 从 workflow.py 抽取，供 workflow / reflector 等模块共用"""
import logging
from typing import Any, Dict

from pydantic import ValidationError

logger = logging.getLogger(__name__)


def validate_with_pydantic(model_class: type, data: Any) -> Dict[str, Any]:
    """用 Pydantic 模型校验 LLM 输出，校验失败返回原始数据并记录警告"""
    try:
        if isinstance(data, list):
            return [model_class(**item).model_dump() for item in data]
        return model_class(**data).model_dump()
    except (ValidationError, TypeError, KeyError) as e:
        logger.warning(f"Pydantic 校验失败({model_class.__name__}): {e}，使用原始数据")
        return data
