"""Named model registry for multi-model retrieval and rerank benchmarking.

Centralizes the embedding, reranker, and LLM endpoints so scripts and the
benchmark can refer to models by short names instead of repeating URLs and
API keys. Secrets are read from the environment (``.env``) and never
hardcoded; the registry only ever hands back configured clients or spec
objects.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .embeddings import OpenAICompatibleEmbedder, EmbeddingConfigurationError
from .rerank import RerankConfigurationError, SiliconFlowReranker
from .llm_client import OpenAICompatibleClient
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


@dataclass(frozen=True)
class LLMSpec:
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
    """Return every configured reranker (all SiliconFlow today).

    NOTE: ``Qwen/Qwen3-VL-Reranker-8B`` was removed on 2026-08-16 — it is a
    vision-language reranker whose gains on pure-text retrieval are not
    representative, and the benchmark should not spend calls on it.
    """
    load_local_env()
    silicon_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    if not silicon_key:
        return []
    return [
        RerankerSpec("siliconflow-qwen3-rerank-4b", SILICONFLOW_BASE_URL, silicon_key, "Qwen/Qwen3-Reranker-4B"),
        RerankerSpec("siliconflow-qwen3-rerank-0.6b", SILICONFLOW_BASE_URL, silicon_key, "Qwen/Qwen3-Reranker-0.6B"),
    ]


def get_llm_specs() -> list[LLMSpec]:
    """Return every configured LLM chat model for RAG answering / insight.

    Volcano chat keys come from ``VOLCANO_LLM_API_KEY``; SiliconFlow chat keys
    come from ``SILICONFLOW_LLM_API_KEY``. These are separate from the embedding
    keys because the two capabilities were provisioned under different tokens.
    """
    load_local_env()
    volcano_key = os.getenv("VOLCANO_LLM_API_KEY", "").strip()
    silicon_key = os.getenv("SILICONFLOW_LLM_API_KEY", "").strip()
    specs: list[LLMSpec] = []
    if volcano_key:
        specs.extend(
            [
                LLMSpec("volcano-glm-4-7", VOLCANO_BASE_URL, volcano_key, "glm-4-7-251222"),
                LLMSpec("volcano-doubao-seed-2-0-lite", VOLCANO_BASE_URL, volcano_key, "doubao-seed-2-0-lite-260428"),
                LLMSpec("volcano-deepseek-v4-flash", VOLCANO_BASE_URL, volcano_key, "deepseek-v4-flash-260425"),
            ]
        )
    if silicon_key:
        specs.extend(
            [
                LLMSpec("siliconflow-deepseek-v3.2", SILICONFLOW_BASE_URL, silicon_key, "deepseek-ai/DeepSeek-V3.2"),
                LLMSpec("siliconflow-qwen3.5-35b", SILICONFLOW_BASE_URL, silicon_key, "Qwen/Qwen3.5-35B-A3B"),
            ]
        )
    return specs


def build_embedder(spec: EmbedderSpec) -> OpenAICompatibleEmbedder:
    return OpenAICompatibleEmbedder(
        base_url=spec.base_url, api_key=spec.api_key, model=spec.model, endpoint=spec.endpoint
    )


def build_reranker(spec: RerankerSpec) -> SiliconFlowReranker:
    return SiliconFlowReranker(base_url=spec.base_url, api_key=spec.api_key, model=spec.model)


def build_llm_client(spec: LLMSpec) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(base_url=spec.base_url, api_key=spec.api_key, model=spec.model)


def get_reachable_embedder_specs(smoke_path: str = "data/model_smoke.json") -> list[EmbedderSpec]:
    """Only embedding models proven reachable by ``scripts/smoke_models.py``.

    The smoke run records live status per model name; this filter drops any
    model that returned an error (e.g. a retired endpoint) so the benchmark
    never spends time on calls that are known to fail.
    """
    specs = get_embedder_specs()
    try:
        with open(smoke_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    ok = {item["name"] for item in data.get("embedders", []) if item.get("status") == "ok"}
    return [spec for spec in specs if spec.name in ok]


def get_reachable_reranker_specs(smoke_path: str = "data/model_smoke.json") -> list[RerankerSpec]:
    """Only reranker models proven reachable by the smoke run."""
    specs = get_reranker_specs()
    try:
        with open(smoke_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    ok = {item["name"] for item in data.get("rerankers", []) if item.get("status") == "ok"}
    return [spec for spec in specs if spec.name in ok]


def get_reachable_llm_specs(smoke_path: str = "data/llm_smoke.json") -> list[LLMSpec]:
    """Only LLM chat models proven reachable by ``scripts/smoke_llms.py``."""
    specs = get_llm_specs()
    try:
        with open(smoke_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    ok = {item["name"] for item in data.get("models", []) if item.get("status") == "ok"}
    return [spec for spec in specs if spec.name in ok]


__all__ = [
    "EmbedderSpec",
    "LLMSpec",
    "RerankerSpec",
    "build_embedder",
    "build_llm_client",
    "build_reranker",
    "get_embedder_specs",
    "get_llm_specs",
    "get_reachable_embedder_specs",
    "get_reachable_llm_specs",
    "get_reachable_reranker_specs",
    "get_reranker_specs",
]
