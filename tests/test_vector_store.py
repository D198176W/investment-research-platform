"""轻量向量库单测：CRUD、余弦检索、metadata 过滤、持久化"""
import os

from agent_service.memory.vector_store import VectorStore


def _vec(dim, hot_idx):
    v = [0.0] * dim
    v[hot_idx] = 1.0
    return v


def test_add_and_query(tmp_path):
    store = VectorStore(str(tmp_path / "vs.json"), dim=8)
    store.add("a", _vec(8, 0), "文档A", {"kind": "conclusion"})
    store.add("b", _vec(8, 1), "文档B", {"kind": "reflection"})

    results = store.query(_vec(8, 0), top_k=2)
    assert results[0]["id"] == "a"
    assert results[0]["score"] > results[1]["score"]


def test_metadata_filter(tmp_path):
    store = VectorStore(str(tmp_path / "vs.json"), dim=8)
    store.add("a", _vec(8, 0), "半导体结论", {"industry": "半导体"})
    store.add("b", _vec(8, 0), "新能源结论", {"industry": "新能源"})

    results = store.query(_vec(8, 0), top_k=5, where={"industry": "新能源"})
    assert len(results) == 1
    assert results[0]["id"] == "b"


def test_upsert_same_id(tmp_path):
    store = VectorStore(str(tmp_path / "vs.json"), dim=8)
    store.add("a", _vec(8, 0), "旧内容")
    store.add("a", _vec(8, 1), "新内容")
    assert store.count() == 1
    assert store.query(_vec(8, 1), top_k=1)[0]["document"] == "新内容"


def test_delete(tmp_path):
    store = VectorStore(str(tmp_path / "vs.json"), dim=8)
    store.add("a", _vec(8, 0), "文档")
    assert store.delete("a") is True
    assert store.count() == 0
    assert store.delete("a") is False


def test_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "vs.json")
    store = VectorStore(path, dim=8)
    store.add("a", _vec(8, 0), "持久化文档", {"k": "v"})
    assert os.path.exists(path)

    store2 = VectorStore(path, dim=8)
    assert store2.count() == 1
    assert store2.query(_vec(8, 0), top_k=1)[0]["document"] == "持久化文档"


def test_dim_alignment(tmp_path):
    """维度不匹配时应对齐（截断或补零）而非报错"""
    store = VectorStore(str(tmp_path / "vs.json"), dim=8)
    store.add("a", [1.0, 2.0, 3.0], "短向量")
    store.add("b", [1.0] * 16, "长向量")
    assert store.count() == 2
