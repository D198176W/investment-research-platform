"""轻量向量库 —— numpy 余弦相似度 + JSON 持久化

为什么不用 Chroma/FAISS：本项目为单机研究场景，数据量 < 10k 条，
numpy 暴力检索足够（万级向量毫秒级），且零额外依赖、可解释、可调试。
接口设计与 Chroma 对齐（add/query/delete），后续可平滑替换。
"""
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """轻量向量存储（JSON 文件持久化）"""

    def __init__(self, persist_path: str, dim: int = 1024):
        self.persist_path = persist_path
        self.dim = dim
        self._ids: List[str] = []
        self._vectors: Optional[np.ndarray] = None  # shape (N, dim)
        self._metadatas: List[Dict[str, Any]] = []
        self._documents: List[str] = []
        self._lock = threading.Lock()
        self._load()

    # ---------- 持久化 ----------

    def _load(self):
        if not os.path.exists(self.persist_path):
            self._vectors = np.zeros((0, self.dim), dtype=np.float32)
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._ids = payload.get("ids", [])
            self._metadatas = payload.get("metadatas", [])
            self._documents = payload.get("documents", [])
            vecs = payload.get("vectors", [])
            self._vectors = np.array(vecs, dtype=np.float32) if vecs else np.zeros((0, self.dim), dtype=np.float32)
            logger.info(f"向量库已加载: {len(self._ids)} 条 ({self.persist_path})")
        except Exception as e:
            logger.error(f"向量库加载失败，重建空库: {e}")
            self._ids, self._metadatas, self._documents = [], [], []
            self._vectors = np.zeros((0, self.dim), dtype=np.float32)

    def _persist(self):
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        payload = {
            "ids": self._ids,
            "vectors": self._vectors.tolist() if self._vectors is not None else [],
            "metadatas": self._metadatas,
            "documents": self._documents,
        }
        tmp = self.persist_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, self.persist_path)  # 原子替换，避免写一半损坏

    # ---------- CRUD ----------

    def add(self, id: str, vector: List[float], document: str,
            metadata: Optional[Dict[str, Any]] = None):
        """新增/覆盖一条记忆"""
        with self._lock:
            vec = np.array(vector, dtype=np.float32)
            if vec.shape[0] != self.dim:
                # 维度不匹配时对齐（截断或补零）
                aligned = np.zeros(self.dim, dtype=np.float32)
                n = min(self.dim, vec.shape[0])
                aligned[:n] = vec[:n]
                vec = aligned
            if id in self._ids:
                idx = self._ids.index(id)
                self._vectors[idx] = vec
                self._documents[idx] = document
                self._metadatas[idx] = metadata or {}
            else:
                self._ids.append(id)
                self._vectors = np.vstack([self._vectors, vec[None, :]]) if self._vectors is not None and self._vectors.shape[0] > 0 else vec[None, :]
                self._documents.append(document)
                self._metadatas.append(metadata or {})
            self._persist()

    def query(self, vector: List[float], top_k: int = 5,
              where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """余弦相似度检索，支持 metadata 过滤

        Returns:
            [{"id", "document", "metadata", "score"}]，按 score 降序
        """
        with self._lock:
            if self._vectors is None or self._vectors.shape[0] == 0:
                return []

            q = np.array(vector, dtype=np.float32)
            q_norm = np.linalg.norm(q) or 1.0
            v_norm = np.linalg.norm(self._vectors, axis=1)
            v_norm[v_norm == 0] = 1.0
            scores = (self._vectors @ q) / (v_norm * q_norm)

            # metadata 过滤
            candidate_idx = list(range(len(self._ids)))
            if where:
                candidate_idx = [
                    i for i in candidate_idx
                    if all(self._metadatas[i].get(k) == v for k, v in where.items())
                ]
            if not candidate_idx:
                return []

            candidate_scores = [(i, float(scores[i])) for i in candidate_idx]
            candidate_scores.sort(key=lambda x: x[1], reverse=True)

            return [
                {
                    "id": self._ids[i],
                    "document": self._documents[i],
                    "metadata": self._metadatas[i],
                    "score": round(score, 4),
                }
                for i, score in candidate_scores[:top_k]
            ]

    def delete(self, id: str) -> bool:
        with self._lock:
            if id not in self._ids:
                return False
            idx = self._ids.index(id)
            self._ids.pop(idx)
            self._documents.pop(idx)
            self._metadatas.pop(idx)
            self._vectors = np.delete(self._vectors, idx, axis=0)
            self._persist()
            return True

    def count(self) -> int:
        return len(self._ids)
