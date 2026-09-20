"""长期记忆 —— 历史任务结论 + Reflector 经验的写入/检索/淘汰

记忆写回与淘汰策略（对齐 JD「记忆窗口截断」「AI 自我进化」）：
- 写入：任务完成后，将「结论摘要 + Reflector 经验」向量化入库
- 检索：新任务开始时，按"主题+行业"检索 top_k 相似历史注入上下文
- 淘汰：超出容量上限时，按"重要性评分(置信度×时间衰减)"淘汰最低分
"""
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .embeddings import EmbeddingProvider
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# 项目根/data/memory_store（与 traces/checkpoints 同级，便于容器化挂载）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MEMORY_DIR = os.path.join(_PROJECT_ROOT, "data", "memory_store")
MEMORY_PATH = os.path.join(MEMORY_DIR, "long_term_memory.json")
MAX_MEMORY_ITEMS = 200  # 容量上限，超过则淘汰


@dataclass
class MemoryItem:
    """一条长期记忆"""
    id: str
    content: str                # 可检索的文本（结论摘要/经验）
    topic: str = ""
    industry: str = ""
    kind: str = "conclusion"    # conclusion（任务结论）/ reflection（反思经验）
    confidence: float = 0.5     # 重要性评分（0~1）
    created_at: float = field(default_factory=time.time)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "industry": self.industry,
            "kind": self.kind,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }


class LongTermMemory:
    """长期记忆管理器"""

    def __init__(self, embedding: Optional[EmbeddingProvider] = None,
                 store_path: str = MEMORY_PATH, max_items: int = MAX_MEMORY_ITEMS):
        self.embedding = embedding or EmbeddingProvider()
        self.store = VectorStore(store_path, dim=1024)
        self.max_items = max_items

    # ---------- 写入 ----------

    def write(self, item: MemoryItem) -> str:
        """写入一条记忆（自动生成 ID 与向量）"""
        if not item.id:
            item.id = str(uuid.uuid4())
        vector = self.embedding.embed(item.content)
        self.store.add(item.id, vector, item.content, item.to_metadata())
        self._evict_if_needed()
        logger.info(f"长期记忆写入: {item.kind} | {item.topic}/{item.industry} | {item.content[:60]}...")
        return item.id

    def write_conclusion(self, topic: str, industry: str, conclusion: str,
                         confidence: float = 0.5) -> str:
        """写入任务结论"""
        return self.write(MemoryItem(
            id="", content=conclusion, topic=topic, industry=industry,
            kind="conclusion", confidence=confidence,
        ))

    def write_reflection(self, topic: str, industry: str, lesson: str,
                         confidence: float = 0.6) -> str:
        """写入反思经验（Reflector 产出，权重略高于普通结论）"""
        return self.write(MemoryItem(
            id="", content=f"[经验教训] {lesson}", topic=topic, industry=industry,
            kind="reflection", confidence=confidence,
        ))

    # ---------- 检索 ----------

    def recall(self, query: str, top_k: int = 3,
               industry: Optional[str] = None,
               min_score: float = 0.3) -> List[Dict[str, Any]]:
        """检索相关记忆

        Args:
            query: 检索文本（通常是"主题+行业"）
            top_k: 返回条数
            industry: 可选，按行业过滤
            min_score: 相似度阈值（过滤低相关结果）
        """
        if self.store.count() == 0:
            return []
        vector = self.embedding.embed(query)
        where = {"industry": industry} if industry else None
        results = self.store.query(vector, top_k=top_k * 2, where=where)
        # 相似度过滤
        results = [r for r in results if r["score"] >= min_score][:top_k]
        logger.info(f"长期记忆检索: query={query[:40]}... 命中 {len(results)} 条")
        return results

    def recall_as_context(self, query: str, top_k: int = 3,
                          industry: Optional[str] = None) -> str:
        """检索并格式化为可注入 Prompt 的上下文"""
        results = self.recall(query, top_k=top_k, industry=industry)
        if not results:
            return ""
        lines = ["【历史分析记忆】（同一/相似标的的历史结论与经验，供参考）"]
        for r in results:
            meta = r["metadata"]
            lines.append(
                f"- [{meta.get('kind')}|相似度{r['score']}] {r['document']}"
            )
        return "\n".join(lines)

    # ---------- 淘汰 ----------

    def _evict_if_needed(self):
        """容量淘汰：按 置信度 × 时间衰减 淘汰最低分"""
        count = self.store.count()
        if count <= self.max_items:
            return
        # 计算每条的保留分
        now = time.time()
        scored = []
        for i, meta in enumerate(self.store._metadatas):
            age_days = (now - meta.get("created_at", now)) / 86400
            time_decay = max(0.1, 1.0 - age_days / 180)  # 180天半衰
            score = meta.get("confidence", 0.5) * time_decay
            scored.append((score, self.store._ids[i]))
        scored.sort(key=lambda x: x[0])
        to_remove = count - self.max_items
        for _, mid in scored[:to_remove]:
            self.store.delete(mid)
        logger.info(f"长期记忆淘汰: 移除 {to_remove} 条低分记忆")

    def stats(self) -> Dict[str, Any]:
        return {
            "count": self.store.count(),
            "max_items": self.max_items,
            "embedding_mode": "remote" if self.embedding._use_remote else "hash_fallback",
            "store_path": self.store.persist_path,
        }
