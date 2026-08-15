"""Reranker clients for two-stage retrieval (embedding recall + rerank).

The retrieval service first recalls candidate chunks with an embedder, then a
reranker reorders those candidates by finer-grained relevance. This module
defines the :class:`Reranker` protocol and a SiliconFlow-backed implementation
that speaks the OpenAI-compatible ``/v1/rerank`` contract.

The same two-stage idea underpins most production RAG stacks: dense embedding
recall is cheap and high-coverage, while a cross-encoder reranker spends more
compute on a small candidate set to fix the ordering. Keeping the reranker
behind a protocol means the retrieval layer can be benchmarked with or without
it using one code path.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from .settings import load_local_env


@dataclass(frozen=True)
class RerankHit:
    """One reranked document, most relevant first."""

    index: int
    score: float
    document: str | None


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[RerankHit]:
        """Reorder ``documents`` by relevance to ``query`` (most relevant first)."""


class RerankConfigurationError(Exception):
    """Raised when required reranker settings are not configured."""


class RerankResponseError(Exception):
    """Raised when a rerank response cannot satisfy the output contract."""


class SiliconFlowReranker:
    """Rerank documents through a SiliconFlow ``/v1/rerank`` compatible endpoint.

    The response is expected to be a list of hits sorted by descending
    relevance, each carrying the original ``index`` into ``documents`` and a
    ``relevance_score``. Parsing tolerates both ``relevance_score`` and the
    shorter ``score`` key so the same client survives minor provider drift.
    """

    def __init__(self, base_url: str, api_key: str, model: str, endpoint: str = "/rerank", timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint or "/rerank"
        self.timeout = httpx.Timeout(timeout_seconds, connect=10.0, read=timeout_seconds, write=10.0, pool=10.0)

    @classmethod
    def from_environment(cls, model: str | None = None) -> "SiliconFlowReranker":
        load_local_env()
        base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").strip()
        api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
        resolved_model = model or os.getenv("RERANKER_MODEL", "").strip()
        if not api_key:
            raise RerankConfigurationError("Set SILICONFLOW_API_KEY before using a SiliconFlow reranker")
        if not resolved_model:
            raise RerankConfigurationError("Provide a reranker model name (or set RERANKER_MODEL)")
        return cls(base_url=base_url, api_key=api_key, model=resolved_model)

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[RerankHit]:
        if not documents:
            return []
        payload: dict[str, object] = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": False,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None

        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(f"{self.base_url}{self.endpoint}", headers=headers, json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable rerank response", request=response.request, response=response)
                response.raise_for_status()
                return self._parse_response(response.json(), len(documents))
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as error:
                last_error = error
                if attempt == 2:
                    break
                time.sleep(0.5 * (2**attempt))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise RerankResponseError("Rerank response did not contain usable hits") from error

        detail = _describe_http_error(last_error) if isinstance(last_error, httpx.HTTPStatusError) else str(last_error)
        raise RerankResponseError(f"Rerank request failed after 3 attempts: {detail}") from last_error

    def _parse_response(self, payload: object, doc_count: int) -> list[RerankHit]:
        if not isinstance(payload, dict):
            raise RerankResponseError("Rerank response root must be an object")
        results = payload.get("results")
        if not isinstance(results, list):
            raise RerankResponseError("Rerank response must contain a 'results' list")
        hits: list[RerankHit] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if not isinstance(index, int) or not isinstance(score, (int, float)):
                raise RerankResponseError("Each rerank hit needs an integer 'index' and a numeric score")
            if not 0 <= index < doc_count:
                raise RerankResponseError(f"Rerank hit index {index} out of range for {doc_count} documents")
            hits.append(RerankHit(index=index, score=float(score), document=item.get("document")))
        if not hits:
            raise RerankResponseError("Rerank response contained no usable hits")
        return sorted(hits, key=lambda hit: -hit.score)


# Re-export the protocol and hit type for callers that depend on the abstraction.
__all__ = [
    "RerankConfigurationError",
    "RerankHit",
    "RerankResponseError",
    "Reranker",
    "SiliconFlowReranker",
]


def _describe_http_error(error: httpx.HTTPStatusError) -> str:
    """Surface the real status code and provider message instead of a generic retry error."""
    response = error.response
    body = ""
    try:
        body = response.text[:300]
    except Exception:  # noqa: BLE001 - best-effort diagnostics only
        body = ""
    return f"HTTP {response.status_code}: {body}" if body else f"HTTP {response.status_code}"
