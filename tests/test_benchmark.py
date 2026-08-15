"""离线单测：BM25 检索器与 IR 评测指标（不依赖网络）。"""

from __future__ import annotations

import math

from paper_api.ir_metrics import mrr_at_k, ndcg_at_k, recall_at_k
from paper_api.retrieval import BM25Retriever, LocalHashingEmbedder, cosine_similarity


def test_bm25_ranks_relevant_doc_first() -> None:
    bm25 = BM25Retriever().fit(
        [
            ("a", "retrieval augmented generation survey for large language models"),
            ("b", "a simple pasta cooking recipe with tomato sauce"),
            ("c", "vector database indexing and approximate nearest neighbor search"),
        ]
    )
    ranked = bm25.search("retrieval augmented generation", top_n=3)
    assert ranked[0][0] == "a"


def test_bm25_handles_chinese_runs() -> None:
    bm25 = BM25Retriever().fit(
        [
            ("cn", "机器学习在推荐系统中的应用研究"),
            ("en", "recommender systems with deep learning"),
        ]
    )
    ranked = bm25.search("机器学习 推荐系统", top_n=2)
    assert ranked[0][0] == "cn"


def test_bm25_top_n_bounds_and_empty_query() -> None:
    bm25 = BM25Retriever().fit([("a", "hello world"), ("b", "foo bar")])
    assert len(bm25.search("hello", top_n=1)) == 1
    # 空查询不应报错，只返回分数全 0 的排序
    assert len(bm25.search("", top_n=5)) == 2


def test_recall_at_k() -> None:
    ranked = ["x", "y", "z"]
    assert recall_at_k(ranked, {"x"}, 1) == 1.0
    assert recall_at_k(ranked, {"x", "y"}, 3) == 1.0
    assert recall_at_k(ranked, {"z"}, 1) == 0.0
    assert recall_at_k(ranked, {"x", "y"}, 1) == 0.5  # 前 1 个只覆盖 1/2 相关项


def test_mrr_at_k() -> None:
    assert mrr_at_k(["a", "b"], {"b"}, 10) == 0.5
    assert mrr_at_k(["a", "b"], {"a"}, 10) == 1.0
    assert mrr_at_k(["a", "b"], {"c"}, 10) == 0.0
    assert mrr_at_k(["a", "b"], {"b"}, 1) == 0.0  # 相关项超出 k


def test_ndcg_at_k() -> None:
    # 相关项就在第一位：nDCG=1
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == 1.0
    # 相关项在第二位：理想位在第一位，折损比 = 1/log2(3)
    expected = 1.0 / math.log2(3)
    assert ndcg_at_k(["a", "b"], {"b"}, 2) == pytest.approx(expected)
    # 无相关项：0
    assert ndcg_at_k(["a", "b"], {"c"}, 2) == 0.0


def test_local_hashing_vector_is_unit_norm() -> None:
    vec = LocalHashingEmbedder().embed("retrieval augmented generation")
    norm = math.sqrt(sum(value * value for value in vec.vector))
    assert norm == pytest.approx(1.0)


def test_cosine_similarity_identical() -> None:
    embedder = LocalHashingEmbedder()
    a = embedder.embed("same text").vector
    b = embedder.embed("same text").vector
    assert cosine_similarity(a, b) == pytest.approx(1.0)


import pytest  # noqa: E402  (放在文件尾以保持测试函数先定义的阅读顺序)
