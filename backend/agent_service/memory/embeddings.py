"""Embedding 提供方 —— DashScope text-embedding，失败降级为本地哈希向量

降级策略保证：无网络/无配额时记忆系统仍可运行（检索质量下降但不中断），
这也是面试中"降级与容错"的叙事点。
"""
import hashlib
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_DIM = 1024  # text-embedding-v3 维度


class EmbeddingProvider:
    """Embedding 提供方（DashScope 优先，哈希降级）"""

    def __init__(self, api_key: str = "", base_url: str = "",
                 model: str = DEFAULT_EMBEDDING_MODEL):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "DASHSCOPE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model
        self._use_remote = bool(self.api_key)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30)
        return self._client

    def embed(self, text: str) -> List[float]:
        """生成文本向量；远程失败时降级为哈希向量"""
        text = (text or "").strip()
        if not text:
            return [0.0] * EMBEDDING_DIM
        if self._use_remote:
            try:
                resp = self._get_client().embeddings.create(model=self.model, input=text[:2048])
                return resp.data[0].embedding
            except Exception as e:
                logger.warning(f"远程 Embedding 失败，降级为哈希向量: {e}")
                self._use_remote = False  # 本次会话内不再尝试远程
        return self._hash_embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成（远程支持批量接口，降级时逐个哈希）"""
        texts = [(t or "").strip() for t in texts]
        if self._use_remote:
            try:
                resp = self._get_client().embeddings.create(
                    model=self.model, input=[t[:2048] for t in texts])
                return [d.embedding for d in resp.data]
            except Exception as e:
                logger.warning(f"远程批量 Embedding 失败，降级为哈希向量: {e}")
                self._use_remote = False
        return [self._hash_embed(t) for t in texts]

    @staticmethod
    def _hash_embed(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
        """本地哈希向量（字符 n-gram 哈希 + L2 归一化）

        仅作为降级兜底：能捕捉字面重叠，无语义能力。
        """
        vec = [0.0] * dim
        for n in (2, 3):
            for i in range(len(text) - n + 1):
                gram = text[i:i + n]
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                vec[h % dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]
