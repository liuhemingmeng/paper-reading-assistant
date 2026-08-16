"""用选定的生产配置跑一次端到端 RAG demo。

配置（见第 11 周教程）：
  - Embedder: siliconflow-bge-m3（文本检索最强, R@1=0.953）
  - LLM:      volcano-deepseek-v4-flash-260425（延迟最低 3.1s, 中文强）
  - Chunking: 1000 字 / 重叠 120
  - 检索 top-N: 10（喂给 LLM 的证据段数）
  - Reranker: 默认不接

为了不触发「全量 embedding 13976 chunk」的慢/不稳问题，demo 只加载 QA 数据集里
少数几篇文档（默认 4 篇：2 arXiv + 2 行业）做检索索引，证明「选定配置端到端可用」。
embedding 缓存复用 data/embed_cache_qa（与基准同源、可续跑），SSL 抖动导致的中断
重跑即可补完，不会白跑。

用法：
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  .venv/Scripts/python.exe scripts/run_rag_demo.py --docs 4 --out data/rag_demo.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper_api.cache_io import load_json_cache, atomic_write_json
from paper_api.chunking import chunk_corpus
from paper_api.corpus import load_corpus
from paper_api.llm_client import OpenAICompatibleClient
from paper_api.model_registry import build_embedder, build_llm_client, get_embedder_specs, get_llm_specs
from paper_api.retrieval import cosine_similarity

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 120
TOP_K = 10
EMBED_CACHE_DIR = "data/embed_cache_qa"  # 与基准同源，可续跑
QA_PATH = "data/qa_dataset.json"
MANIFEST = "data/corpus/corpus_manifest.json"


def pick_subset_docs(docs: int) -> list[dict]:
    """从 QA 数据集取前 docs 个不同 doc_id（尽量 arXiv/行业各半）。"""
    qa = json.load(open(QA_PATH, encoding="utf-8"))["items"]
    chosen: list[dict] = []
    seen: set[str] = set()
    for it in qa:
        if it["doc_id"] in seen:
            continue
        seen.add(it["doc_id"])
        chosen.append(it)
        if len(chosen) >= docs:
            break
    return chosen


def embed_subset(embedder, chunk_ids: list[str], chunk_text: dict[str, str]) -> dict[str, list[float]]:
    """对子集 chunk 做 embedding，复用缓存、SSL 失败跳过续跑。"""
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(EMBED_CACHE_DIR, "siliconflow-bge-m3.json")
    cache = load_json_cache(cache_path)
    done = 0
    for cid in chunk_ids:
        if cid in cache:
            continue
        try:
            cache[cid] = embedder.embed(chunk_text[cid]).vector
        except Exception as error:  # noqa: BLE001 - 单 chunk 失败不应中断整轮 demo
            print(f"  [warn] embed failed, skip: {cid} ({error})", flush=True)
            continue
        done += 1
        if done % 50 == 0:
            atomic_write_json(cache_path, cache)
            print(f"  embedded {done} new chunks (total cached {len(cache)})", flush=True)
    atomic_write_json(cache_path, cache)
    print(f"  chunk vectors ready: {len(cache)} cached, {done} newly embedded", flush=True)
    return cache


def embed_query(embedder, question: str, cache: dict, cache_path: str) -> Optional[list[float]]:
    if question in cache:
        return cache[question]
    try:
        vec = embedder.embed(question).vector
    except Exception as error:  # noqa: BLE001
        print(f"  [warn] query embed failed: {error}", flush=True)
        return None
    cache[question] = vec
    atomic_write_json(cache_path, cache)
    return vec


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end RAG demo with the chosen config")
    parser.add_argument("--docs", type=int, default=4, help="number of source docs to load from QA set")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="evidence chunks sent to the LLM")
    parser.add_argument("--out", default="data/rag_demo.json", help="result json path")
    args = parser.parse_args()

    embed_spec = next(s for s in get_embedder_specs() if s.name == "siliconflow-bge-m3")
    embedder = build_embedder(embed_spec)
    llm_spec = next(s for s in get_llm_specs() if s.name == "volcano-deepseek-v4-flash")
    llm: OpenAICompatibleClient = build_llm_client(llm_spec)
    print(f"config: embedder={embed_spec.model}  llm={llm_spec.model}  chunk={CHUNK_SIZE}/{CHUNK_OVERLAP}  top_k={args.top_k}", flush=True)

    subset = pick_subset_docs(args.docs)
    doc_ids = {it["doc_id"] for it in subset}
    docs = [d for d in load_corpus(MANIFEST) if d.doc_id in doc_ids]
    chunks = chunk_corpus(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    chunk_ids = [c.chunk_id for c in chunks]
    chunk_text = {c.chunk_id: c.text for c in chunks}
    chunk_doc = {c.chunk_id: c.doc_id for c in chunks}
    print(f"loaded {len(docs)} docs -> {len(chunks)} chunks", flush=True)

    cache = embed_subset(embedder, chunk_ids, chunk_text)
    query_cache = load_json_cache(os.path.join(EMBED_CACHE_DIR, "siliconflow-bge-m3.json"))
    query_cache_path = os.path.join(EMBED_CACHE_DIR, "siliconflow-bge-m3.json")

    results = []
    for it in subset:
        q = it["question"]
        qv = embed_query(embedder, q, query_cache, query_cache_path)
        if qv is None:
            continue
        scored = sorted(
            ((cid, cosine_similarity(cache[cid], qv)) for cid in chunk_ids if cid in cache),
            key=lambda x: -x[1],
        )
        top = scored[: args.top_k]
        if not top:
            continue
        evidence = "\n\n".join(f"[doc_id={chunk_doc[cid]} | chunk={cid}]\n{chunk_text[cid]}" for cid, _ in top)
        ans = llm.answer(q, evidence).answer
        hit_doc = top[0][0]
        results.append({
            "question": q,
            "source": it.get("source"),
            "target_doc_id": it["doc_id"],
            "top1_chunk": hit_doc,
            "top1_doc_id": chunk_doc.get(hit_doc),
            "top1_score": round(top[0][1], 4),
            "top10_doc_ids": [chunk_doc[cid] for cid, _ in top],
            "answer": ans,
            "citations": [{"chunk_id": cid, "doc_id": chunk_doc[cid], "score": round(s, 4)} for cid, s in top],
        })
        print("\n" + "=" * 70)
        print(f"Q({it.get('source')}): {q}")
        print(f"top1 chunk={hit_doc} doc={chunk_doc.get(hit_doc)} score={top[0][1]:.4f}")
        print(f"top1 命中文档是否即问题来源文档: {'是' if chunk_doc.get(hit_doc) == it['doc_id'] else '否'}")
        print("-" * 70)
        print(f"A: {ans}")
        print("=" * 70, flush=True)

    json.dump({"config": {"embedder": embed_spec.model, "llm": llm_spec.model, "chunk": f"{CHUNK_SIZE}/{CHUNK_OVERLAP}", "top_k": args.top_k}, "results": results}, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {len(results)} demo answers -> {args.out}")


if __name__ == "__main__":
    main()
