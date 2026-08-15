"""Offline tests for the SiliconFlow reranker client.

Network calls are replaced with a fake httpx client so the contract (payload
shape, response parsing, error handling) is verified without credentials or
outbound traffic. The live connectivity check lives in ``scripts/smoke_models.py``.
"""

from __future__ import annotations

import pytest

from paper_api.rerank import (
    RerankConfigurationError,
    RerankResponseError,
    SiliconFlowReranker,
)


def test_rerank_returns_hits_sorted_by_descending_score() -> None:
    reranker = SiliconFlowReranker(base_url="https://api.siliconflow.cn/v1", api_key="k", model="M")
    payload = {
        "results": [
            {"index": 2, "relevance_score": 0.10},
            {"index": 0, "relevance_score": 0.90},
            {"index": 1, "relevance_score": 0.50},
        ]
    }
    hits = reranker._parse_response(payload, doc_count=3)
    assert [hit.index for hit in hits] == [0, 1, 2]
    assert [hit.score for hit in hits] == pytest.approx([0.90, 0.50, 0.10])


def test_rerank_tolerates_short_score_key() -> None:
    reranker = SiliconFlowReranker(base_url="https://api.siliconflow.cn/v1", api_key="k", model="M")
    hits = reranker._parse_response({"results": [{"index": 0, "score": 0.7}]}, doc_count=1)
    assert hits[0].index == 0
    assert hits[0].score == pytest.approx(0.7)


def test_rerank_empty_documents_returns_no_hits_without_network() -> None:
    reranker = SiliconFlowReranker(base_url="https://api.siliconflow.cn/v1", api_key="k", model="M")
    assert reranker.rerank("query", []) == []


def test_rerank_rejects_missing_results() -> None:
    reranker = SiliconFlowReranker(base_url="https://api.siliconflow.cn/v1", api_key="k", model="M")
    with pytest.raises(RerankResponseError):
        reranker._parse_response({}, doc_count=1)


def test_rerank_rejects_hit_without_score() -> None:
    reranker = SiliconFlowReranker(base_url="https://api.siliconflow.cn/v1", api_key="k", model="M")
    with pytest.raises(RerankResponseError):
        reranker._parse_response({"results": [{"index": 0}]}, doc_count=1)


def test_rerank_rejects_out_of_range_index() -> None:
    reranker = SiliconFlowReranker(base_url="https://api.siliconflow.cn/v1", api_key="k", model="M")
    with pytest.raises(RerankResponseError):
        reranker._parse_response({"results": [{"index": 5, "relevance_score": 0.5}]}, doc_count=2)


def test_rerank_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeResponse:
        status_code: int = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"results": [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.4}]}

    class _FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def post(self, url: str, headers: object = None, json: object = None) -> _FakeResponse:
            captured["url"] = url
            captured["json"] = json
            return _FakeResponse()

    monkeypatch.setattr("paper_api.rerank.httpx.Client", _FakeClient)
    reranker = SiliconFlowReranker(
        base_url="https://api.siliconflow.cn/v1", api_key="sk", model="Qwen/Qwen3-Reranker-4B"
    )
    hits = reranker.rerank("what is RAG?", ["doc a", "doc b"], top_n=2)

    assert str(captured["url"]).endswith("/rerank")
    body = captured["json"]
    assert isinstance(body, dict)
    assert body["model"] == "Qwen/Qwen3-Reranker-4B"
    assert body["query"] == "what is RAG?"
    assert body["documents"] == ["doc a", "doc b"]
    assert body["return_documents"] is False
    assert body["top_n"] == 2
    assert hits[0].index == 0


def test_rerank_from_environment_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("RERANKER_MODEL", raising=False)
    monkeypatch.setattr("paper_api.rerank.load_local_env", lambda: None)
    with pytest.raises(RerankConfigurationError):
        SiliconFlowReranker.from_environment()


def test_rerank_from_environment_resolves_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk")
    monkeypatch.setenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
    monkeypatch.setattr("paper_api.rerank.load_local_env", lambda: None)
    reranker = SiliconFlowReranker.from_environment()
    assert reranker.model == "Qwen/Qwen3-Reranker-0.6B"
