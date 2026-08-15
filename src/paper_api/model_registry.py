"""Named model registry for multi-model retrieval and rerank benchmarking.

Centralizes the embedding and reranker endpoints so scripts and the corpus
benchmark can refer to models by short names instead of repeating URLs and API
keys. Secrets are read from the environment (``.env``) and never hardcoded; the
registry only ever hands back configured clients or spec objects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .embeddings import OpenAICompatibleEmbedder, EmbeddingConfigurationError
from .rerank import RerankConfigurationError, SiliconFlowReranker
from .settings import load_local_env

# Provider base URLs (public, safe to keep in code).
VOLCANO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"


@dataclass(frozen=True)
class EmbedderSpec:
    name: str
    base_url: str
    api_key: str
    model: str
    endpoint: str


@dataclass(frozen=True)
class RerankerSpec:
    name: str
    base_url: str
    api_key: str
    model: str


def get_embedder_specs() -> list[EmbedderSpec]:
    """Return every configured embedding model.

    Volcano keys come from ``EMBEDDING_API_KEY``; SiliconFlow keys come from
    ``SILICONFLOW_API_KEY``. A model whose key is missing is omitted so callers
    can run the offline subset without erroring.
    """
    load_local_env()
    volcano_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    silicon_key = os.getenv("SILICONFLOW_API_KEY", "").strip()

    specs: list[EmbedderSpec] = []
    if volcano_key:
        specs.append(
            EmbedderSpec(
                name="volcano-vision-251215",
                base_url=VOLCANO_BASE_URL,
                api_key=volcano_key,
                model="doubao-embedding-vision-251215",
                endpoint="/embeddings/multimodal",
            )
        )
        specs.append(
            EmbedderSpec(
                name="volcano-large-text-250515",
                base_url=VOLCANO_BASE_URL,
                api_key=volcano_key,
                model="doubao-embedding-large-text-250515",
                endpoint="/embeddings",
            )
        )
    if silicon_key:
        specs.extend(
            [
                EmbedderSpec(
                    name="siliconflow-qwen3vl-embed-8b",
                    base_url=SILICONFLOW_BASE_URL,
                    api_key=silicon_key,
                    model="Qwen/Qwen3-VL-Embedding-8B",
                    endpoint="/embeddings",
                ),
                EmbedderSpec(
                    name="siliconflow-qwen3-embed-0.6b",
                    base_url=SILICONFLOW_BASE_URL,
                    api_key=silicon_key,
                    model="Qwen/Qwen3-Embedding-0.6B",
                    endpoint="/embeddings",
                ),
                EmbedderSpec(
                    name="siliconflow-bge-m3",
                    base_url=SILICONFLOW_BASE_URL,
                    api_key=silicon_key,
                    model="BAAI/bge-m3",
                    endpoint="/embeddings",
                ),
            ]
        )
    return specs


def get_reranker_specs() -> list[RerankerSpec]:
    """Return every configured reranker (all SiliconFlow today)."""
    load_local_env()
    silicon_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    if not silicon_key:
        return []
    return [
        RerankerSpec("siliconflow-qwen3vl-rerank-8b", SILICONFLOW_BASE_URL, silicon_key, "Qwen/Qwen3-VL-Reranker-8B"),
        RerankerSpec("siliconflow-qwen3-rerank-4b", SILICONFLOW_BASE_URL, silicon_key, "Qwen/Qwen3-Reranker-4B"),
        RerankerSpec("siliconflow-qwen3-rerank-0.6b", SILICONFLOW_BASE_URL, silicon_key, "Qwen/Qwen3-Reranker-0.6B"),
    ]


def build_embedder(spec: EmbedderSpec) -> OpenAICompatibleEmbedder:
    return OpenAICompatibleEmbedder(
        base_url=spec.base_url, api_key=spec.api_key, model=spec.model, endpoint=spec.endpoint
    )


def build_reranker(spec: RerankerSpec) -> SiliconFlowReranker:
    return SiliconFlowReranker(base_url=spec.base_url, api_key=spec.api_key, model=spec.model)


__all__ = [
    "EmbedderSpec",
    "RerankerSpec",
    "build_embedder",
    "build_reranker",
    "get_embedder_specs",
    "get_reranker_specs",
]
