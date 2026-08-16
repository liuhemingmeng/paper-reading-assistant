#!/usr/bin/env python
"""语料级真实问答检索基准（chunk 级）+ reranker 重排。

与 ``benchmark_retrieval.py``（已知项检索：标题找自己）互补：这里用 LLM 生成的
**自然语言问题**驱动检索，正样本 = 问题所属文档。更接近真实 RAG 的「用户提问 →
证据段落」场景，也是 reranker 真正发挥作用的舞台。

检索单元是 chunk（整篇文档切 1000 字滑窗），从而：
- 词法检索（BM25）不再享受「标题词 = 正文词」的送分题；
- 语义检索需要在段落粒度上匹配问题语义；
- reranker 在候选 chunk 上重排，增益更能体现。

指标：``Recall@K`` / ``MRR@10`` / ``nDCG@K``（K=1,3,5,10），按 chunk→doc_id 映射
判定正样本，arxiv/industry 分组。embedding chunk 向量按模型缓存；rerank 组合可
续跑，超时被杀重跑只补未完成的查询。

用法
----
    python scripts/benchmark_qa_retrieval.py --out data/benchmark_qa.json
    python scripts/benchmark_qa_retrieval.py --limit 20 --no-rerank   # 自测
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_api.chunking import chunk_corpus
from paper_api.corpus import load_corpus
from paper_api.ir_metrics import mean, mrr_at_k, ndcg_at_k, recall_at_k
from paper_api.model_registry import (
    build_embedder,
    build_reranker,
    get_reachable_embedder_specs,
    get_reachable_reranker_specs,
)
from paper_api.cache_io import atomic_write_json, load_json_cache
from paper_api.retrieval import BM25Retriever, LocalHashingEmbedder, cosine_similarity

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 120
QA_PATH = "data/qa_dataset.json"
EMBED_CACHE_DIR = "data/embed_cache_qa"
RERANK_CACHE_DIR = "data/rerank_cache_qa"
RERANK_PAUSE = 0.06
DOC_EMBED_CHARS = 4000
RERANK_DOC_CHARS = 1200
# 真实 QA 基准聚焦文本语义：只编码 2 个文本 embedding（强 + 中），BM25 / local-hashing
# 免费。视觉/多模态 embedder 在纯文本上偏弱，其结论已在第 9/10 周文档级基准论证过。
QA_EMBED_MODELS = ["siliconflow-bge-m3", "siliconflow-qwen3-embed-0.6b"]
# 每编码多少 chunk 落盘一次（瞬态锁不中断整轮，被杀重跑只补未完成片段）
FLUSH_EVERY = 300


class ChunkVectorRetriever:
    """Dense 检索器：chunk 向量与问题向量都预先算好，search 只做余弦排序。"""

    def __init__(self, name: str, chunk_vecs: dict[str, list[float]], query_vecs: dict[str, list[float]]) -> None:
        self.name = name
        self.chunk_vecs = chunk_vecs
        self.query_vecs = query_vecs

    def search(self, question: str, top_n: int | None = None) -> list[str]:
        qv = self.query_vecs[question]
        scored = [(cid, cosine_similarity(qv, vec)) for cid, vec in self.chunk_vecs.items()]
        scored.sort(key=lambda item: -item[1])
        ranked = [cid for cid, _ in scored]
        return ranked[:top_n] if top_n else ranked


class ChunkBM25:
    def __init__(self, name: str, bm25: BM25Retriever) -> None:
        self.name = name
        self.bm25 = bm25

    def search(self, question: str, top_n: int | None = None) -> list[str]:
        return [cid for cid, _ in self.bm25.search(question, top_n)]


def chunk_id_to_doc(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def embed_chunks_and_queries(spec, chunk_ids, chunk_texts, questions) -> tuple[dict, dict]:
    """对指定模型算 chunk 向量与问题向量，按模型缓存、可续跑。返回 (chunk_vecs, query_vecs)。

    缓存用原子写 + 周期性 flush：瞬态锁不中断整轮，被杀重跑只补未完成的片段。
    """
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(EMBED_CACHE_DIR, f"{spec.name}.json")
    cache: dict[str, list[float]] = load_json_cache(cache_file)

    client = build_embedder(spec)
    chunk_vecs: dict[str, list[float]] = {}
    for cid in chunk_ids:
        if cid in cache:
            chunk_vecs[cid] = cache[cid]
    done = 0
    for cid in chunk_ids:
        if cid in chunk_vecs:
            continue
        chunk_vecs[cid] = client.embed(chunk_texts[cid][:DOC_EMBED_CHARS]).vector
        cache[cid] = chunk_vecs[cid]
        done += 1
        if done % FLUSH_EVERY == 0:
            atomic_write_json(cache_file, cache)  # 周期性落盘，非致命

    query_vecs: dict[str, list[float]] = {}
    for q in questions:
        if q in cache:
            query_vecs[q] = cache[q]
    for q in questions:
        if q in query_vecs:
            continue
        query_vecs[q] = client.embed(q).vector
        cache[q] = query_vecs[q]

    atomic_write_json(cache_file, cache)
    return chunk_vecs, query_vecs


def evaluate(ranked_per_query: dict[str, list[str]], positives: dict[str, str], ks: list[int]) -> dict:
    out: dict[int, dict[str, float]] = {}
    for k in ks:
        recalls = [recall_at_k(ranked, {positives[q]}, k) for q, ranked in ranked_per_query.items()]
        mrrs = [mrr_at_k(ranked, {positives[q]}, k) for q, ranked in ranked_per_query.items()]
        ndcgs = [ndcg_at_k(ranked, {positives[q]}, k) for q, ranked in ranked_per_query.items()]
        out[k] = {"recall": mean(recalls), "mrr": mean(mrrs), "ndcg": mean(ndcgs)}
    return out


def evaluate_subset(ranked_per_query, positives, sources, source, ks) -> dict:
    subset = {q: ranked for q, ranked in ranked_per_query.items() if sources.get(q) == source}
    sub_pos = {q: positives[q] for q in subset}
    return evaluate(subset, sub_pos, ks)


def build_base_retrievers(chunk_ids, chunk_texts, questions) -> list[tuple[str, object]]:
    retrievers: list[tuple[str, object]] = []
    local = LocalHashingEmbedder()
    local_chunk = {cid: local.embed(chunk_texts[cid]).vector for cid in chunk_ids}
    local_query = {q: local.embed(q).vector for q in questions}
    retrievers.append(("local-hashing", ChunkVectorRetriever("local-hashing", local_chunk, local_query)))

    bm25 = BM25Retriever().fit([(cid, chunk_texts[cid]) for cid in chunk_ids])
    retrievers.append(("bm25", ChunkBM25("bm25", bm25)))

    for spec in get_reachable_embedder_specs():
        if spec.name not in QA_EMBED_MODELS:
            continue
        cv, qv = embed_chunks_and_queries(spec, chunk_ids, chunk_texts, questions)
        retrievers.append((spec.name, ChunkVectorRetriever(spec.name, cv, qv)))
        print(f"  [embed] {spec.name}: chunks={len(cv)} queries={len(qv)}", flush=True)
    return retrievers


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus-level real-QA retrieval benchmark (chunk-level + reranker)")
    parser.add_argument("--out", default="data/benchmark_qa.json")
    parser.add_argument("--qa", default=QA_PATH)
    parser.add_argument("--ks", default="1,3,5,10")
    parser.add_argument("--candidates", type=int, default=20, help="每个基础检索器喂给 reranker 的候选 chunk 数")
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 篇文档建索引（自测用）")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    max_k = max(ks)

    print("加载语料并切 chunk…", flush=True)
    docs = load_corpus()
    if args.limit:
        docs = docs[: args.limit]
    chunks = chunk_corpus(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    chunk_ids = [c.chunk_id for c in chunks]
    chunk_texts = {c.chunk_id: c.text for c in chunks}
    print(f"chunk 数：{len(chunk_ids)}（来自 {len(docs)} 篇文档）", flush=True)

    with open(args.qa, encoding="utf-8") as handle:
        qa = json.load(handle)
    items = qa["items"]
    questions = [it["question"] for it in items]
    positives = {it["question"]: it["doc_id"] for it in items}
    sources = {it["question"]: it["source"] for it in items}
    print(f"QA 问题数：{len(questions)}（arxiv={sum(1 for s in sources.values() if s=='arxiv')} "
          f"industry={sum(1 for s in sources.values() if s=='industry')}）", flush=True)

    print("构建基础检索器（local-hashing / BM25 / 各 embedding）…", flush=True)
    base_retrievers = build_base_retrievers(chunk_ids, chunk_texts, questions)

    overall: dict[str, dict] = {}
    by_source: dict[str, dict[str, dict]] = {"arxiv": {}, "industry": {}}

    for name, retriever in base_retrievers:
        ranked_doc = {q: [chunk_id_to_doc(cid) for cid in retriever.search(q)] for q in questions}
        overall[name] = evaluate(ranked_doc, positives, ks)
        by_source["arxiv"][name] = evaluate_subset(ranked_doc, positives, sources, "arxiv", ks)
        by_source["industry"][name] = evaluate_subset(ranked_doc, positives, sources, "industry", ks)
        top1 = overall[name][1]["recall"]
        print(f"  [base] {name}: Recall@1={top1:.3f} MRR@10={overall[name][max_k]['mrr']:.3f}", flush=True)

    reranker_section: dict[str, dict] = {}
    if not args.no_rerank:
        os.makedirs(RERANK_CACHE_DIR, exist_ok=True)
        rerank_specs = get_reachable_reranker_specs()
        print(f"reranker 重排（接在每个基础检索器后，候选={args.candidates}，可续跑）…", flush=True)
        for spec in rerank_specs:
            reranker = build_reranker(spec)
            combos: dict[str, dict] = {}
            total_errors = 0
            for base_name, retriever in base_retrievers:
                cache_file = os.path.join(RERANK_CACHE_DIR, f"{spec.name}__{base_name}.json")
                combo_cache: dict[str, list[str]] = {}
                if os.path.exists(cache_file):
                    with open(cache_file, encoding="utf-8") as handle:
                        combo_cache = json.load(handle)
                ranked_doc: dict[str, list[str]] = {}
                errs = 0
                pending = 0
                for idx, q in enumerate(questions):
                    if q in combo_cache:
                        ranked_doc[q] = combo_cache[q]
                        continue
                    pending += 1
                    pool = retriever.search(q, args.candidates)
                    if not pool:
                        ranked_doc[q] = []
                        combo_cache[q] = []
                        continue
                    docs_text = [chunk_texts[cid][:RERANK_DOC_CHARS] for cid in pool]
                    try:
                        hits = reranker.rerank(q, docs_text)
                        reranked = [pool[hit.index] for hit in hits]
                    except Exception:  # noqa: BLE001 - 单条失败不中断整体
                        errs += 1
                        reranked = pool
                    ranked_doc[q] = [chunk_id_to_doc(cid) for cid in reranked]
                    combo_cache[q] = ranked_doc[q]
                    if idx % 10 == 0:
                        atomic_write_json(cache_file, combo_cache)
                    time.sleep(RERANK_PAUSE)
                atomic_write_json(cache_file, combo_cache)
                total_errors += errs
                combos[base_name] = {
                    "overall": evaluate(ranked_doc, positives, ks),
                    "by_source": {
                        "arxiv": evaluate_subset(ranked_doc, positives, sources, "arxiv", ks),
                        "industry": evaluate_subset(ranked_doc, positives, sources, "industry", ks),
                    },
                }
                base_r1 = overall[base_name][1]["recall"]
                combo_r1 = combos[base_name]["overall"][1]["recall"]
                cached = len(questions) - pending
                print(f"    {spec.name} + {base_name}: R@1 {base_r1:.3f}→{combo_r1:.3f} "
                      f"(Δ{combo_r1 - base_r1:+.3f}) 缓存{cached} 新算{pending} errors={errs}", flush=True)
            reranker_section[spec.name] = {"combos": combos, "errors": total_errors}

    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "ks": ks,
            "candidates": args.candidates,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "doc_embed_chars": DOC_EMBED_CHARS,
            "rerank_doc_chars": RERANK_DOC_CHARS,
            "n_documents": len(docs),
            "n_chunks": len(chunk_ids),
            "n_questions": len(questions),
        },
        "overall": overall,
        "by_source": by_source,
        "rerankers": reranker_section,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {args.out}")
    _print_table(results, ks, max_k)


def _short(name: str) -> str:
    return name.replace("siliconflow-", "").replace("volcano-", "")[:10]


def _print_table(results: dict, ks: list[int], max_k: int) -> None:
    base_names = list(results["overall"].keys())
    col_w = 13
    header = f"{'retriever':<26}" + "".join(f" R@{k:<3}" for k in ks) + f"  MRR@{max_k:<3} nDCG@{max_k}"
    print("\n[基础检索器]")
    print(header)
    print("-" * len(header))
    for name in base_names:
        sub = results["overall"][name]
        row = f"{name:<26}"
        for k in ks:
            row += f" {sub[k]['recall']:<5.3f}"
        row += f"  {sub[max_k]['mrr']:<6.3f} {sub[max_k]['ndcg']:.3f}"
        print(row)

    if results["rerankers"]:
        print("\n[reranker 接在基础检索器后：R@1/MRR@10]")
        hdr = f"{'reranker \\ base':<26}" + "".join(f"{_short(b):<{col_w}}" for b in base_names)
        print(hdr)
        print("-" * len(hdr))
        for rname, rsec in results["rerankers"].items():
            row = f"{rname:<26}"
            for b in base_names:
                combo = rsec.get("combos", {}).get(b, {}).get("overall", {})
                if combo:
                    row += f"{combo[1]['recall']:.3f}/{combo[max_k]['mrr']:.3f}".ljust(col_w)
                else:
                    row += "—".ljust(col_w)
            print(row)


if __name__ == "__main__":
    main()
