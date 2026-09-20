"""Data Service - 数据处理和分析"""
import json
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# 历史记录文件路径
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "analysis_history.json")


class DataService:
    """数据服务 - 处理历史记录的存储和查询"""
    
    def __init__(self, history_file: str = None):
        self.history_file = history_file or HISTORY_FILE
        # 确保目录存在
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
    
    def load_history(self) -> List[Dict[str, Any]]:
        """加载历史记录"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"加载历史记录失败: {e}")
        return []
    
    def save_history(self, history: List[Dict[str, Any]]) -> bool:
        """保存历史记录"""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"保存历史记录失败: {e}")
            return False
    
    def add_record(self, record: Dict[str, Any]) -> bool:
        """添加新记录"""
        history = self.load_history()
        history.insert(0, record)
        # 限制最多20条
        if len(history) > 20:
            history = history[:20]
        return self.save_history(history)
    
    def delete_record(self, index: int) -> bool:
        """删除指定索引的记录"""
        history = self.load_history()
        if 0 <= index < len(history):
            history.pop(index)
            return self.save_history(history)
        return False
    
    def clear_history(self) -> bool:
        """清空历史记录"""
        try:
            if os.path.exists(self.history_file):
                os.remove(self.history_file)
            return True
        except Exception as e:
            logger.error(f"清空历史记录失败: {e}")
            return False
    
    def get_record(self, index: int) -> Optional[Dict[str, Any]]:
        """获取指定索引的记录"""
        history = self.load_history()
        if 0 <= index < len(history):
            return history[index]
        return None
    
    def get_recent_records(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近的记录"""
        history = self.load_history()
        return history[:limit]
    
    def search_records(self, keyword: str) -> List[Dict[str, Any]]:
        """搜索历史记录"""
        history = self.load_history()
        keyword_lower = keyword.lower()
        return [
            record for record in history
            if keyword_lower in record.get("topic", "").lower()
            or keyword_lower in record.get("industry", "").lower()
        ]
