"""OpenAI-compatible embedding client and embedder selection.

The retrieval service depends only on the :class:`TextEmbedder` protocol from
``retrieval.py``. This module adds a real, network-backed embedder that speaks
the OpenAI ``/embeddings`` contract (and the Volcano Engine ``/embeddings/
multimodal`` variant), plus a selector that falls back to the
dependency-free local baseline when no embedding service is configured.
"""

from __future__ import annotations

import json
import os
import time
from typing import Protocol

import httpx

from .retrieval import EmbeddedText, LocalHashingEmbedder, TextEmbedder, normalize
from .settings import load_local_env


class EmbeddingConfigurationError(Exception):
    """Raised when required embedding settings are not configured."""


class EmbeddingResponseError(Exception):
    """Raised when an embedding response cannot satisfy the output contract."""


class OpenAICompatibleEmbedder:
    """Embed text through an OpenAI-compatible ``/embeddings`` endpoint.

    The ``endpoint`` argument targets providers whose contract differs from the
    vanilla OpenAI path. Volcano Engine serves its multimodal embedding model at
    ``/embeddings/multimodal``: it expects a structured ``input`` list and
    returns a single ``data`` object instead of a list. When ``endpoint``
    contains ``multimodal`` the payload and response parsing switch to that
    variant automatically, so the same client works for both providers.
    """

    def __init__(self, base_url: str, api_key: str, model: str, endpoint: str = "/embeddings", timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint or "/embeddings"
        self.multimodal = "multimodal" in self.endpoint
        self.timeout = httpx.Timeout(timeout_seconds, connect=10.0, read=timeout_seconds, write=10.0, pool=10.0)

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleEmbedder":
        load_local_env()
        base_url = os.getenv("EMBEDDING_BASE_URL", "").strip()
        api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
        model = os.getenv("EMBEDDING_MODEL", "").strip()
        endpoint = os.getenv("EMBEDDING_ENDPOINT", "/embeddings").strip()
        if not all((base_url, api_key, model)):
            raise EmbeddingConfigurationError(
                "Set EMBEDDING_BASE_URL, EMBEDDING_API_KEY, and EMBEDDING_MODEL before using a real embedder"
            )
        return cls(base_url=base_url, api_key=api_key, model=model, endpoint=endpoint or "/embeddings")

    def embed(self, text: str) -> EmbeddedText:
        payload = self._build_payload(text)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None

        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(f"{self.base_url}{self.endpoint}", headers=headers, json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable embedding response", request=response.request, response=response)
                response.raise_for_status()
                return self._parse_response(response.json())
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as error:
                last_error = error
                if attempt == 2:
                    break
                time.sleep(0.5 * (2**attempt))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise EmbeddingResponseError("Embedding response did not contain a usable vector") from error

        detail = _describe_http_error(last_error) if isinstance(last_error, httpx.HTTPStatusError) else str(last_error)
        raise EmbeddingResponseError(f"Embedding request failed after 3 attempts: {detail}") from last_error

    def _build_payload(self, text: str) -> dict[str, object]:
        if self.multimodal:
            return {
                "model": self.model,
                "encoding_format": "float",
                "input": [{"type": "text", "text": text}],
            }
        return {"model": self.model, "input": [text]}

    def _parse_response(self, payload: object) -> EmbeddedText:
        if not isinstance(payload, dict):
            raise EmbeddingResponseError("Embedding response root must be an object")
        data = payload.get("data")
        if isinstance(data, dict):
            vector = data.get("embedding")
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            vector = data[0].get("embedding")
        else:
            raise EmbeddingResponseError("Embedding response data must contain a usable vector")
        if not isinstance(vector, list) or not all(isinstance(value, (int, float)) for value in vector):
            raise EmbeddingResponseError("Embedding vector must be a list of numbers")
        return EmbeddedText(vector=normalize([float(value) for value in vector]), model=self.model)


def _describe_http_error(error: httpx.HTTPStatusError) -> str:
    """Surface the real status code and provider message instead of a generic retry error."""
    response = error.response
    body = ""
    try:
        body = response.text[:300]
    except Exception:  # noqa: BLE001 - best-effort diagnostics only
        body = ""
    return f"HTTP {response.status_code}: {body}" if body else f"HTTP {response.status_code}"


def get_default_embedder() -> TextEmbedder:
    """Return the configured real embedder, or the local baseline if unset.

    Selection is environment-driven so the same code path serves both a
    deterministic offline baseline and a production embedding service. The
    index and the queries for a single running app always share one embedder,
    which keeps stored vectors and query vectors comparable.
    """
    load_local_env()
    base_url = os.getenv("EMBEDDING_BASE_URL", "").strip()
    api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    model = os.getenv("EMBEDDING_MODEL", "").strip()
    endpoint = os.getenv("EMBEDDING_ENDPOINT", "/embeddings").strip()
    if all((base_url, api_key, model)):
        return OpenAICompatibleEmbedder(base_url=base_url, api_key=api_key, model=model, endpoint=endpoint or "/embeddings")
    return LocalHashingEmbedder()


# Re-export the protocol for callers that depend on the embedder abstraction.
__all__ = [
    "EmbeddingConfigurationError",
    "EmbeddingResponseError",
    "OpenAICompatibleEmbedder",
    "get_default_embedder",
    "TextEmbedder",
]
