#!/usr/bin/env python
"""语料级多模型检索基准（known-item search）。

设计
----
对每篇文档 ``d``，以其**标题**作为查询，正样本集合 = ``{d}``。任务即：
给定一段标题描述，检索器能否把正确的文档排到前面。这是一种标准的「已知项
检索」，能公平比较词法检索（BM25、本地哈希）与语义向量检索（各 embedding
模型），并检验 reranker 在候选池上的重排增益。

检索器
------
- ``local-hashing``：最弱离线基线（哈希词袋，无语义）
- ``bm25``：词法金标准（离线；中文整段不切词，对中文子集天然吃亏）
- 每个 smoke 验证可达的 embedding 模型（dense 向量）
- 每个 reranker 接在「每个基础检索器」后面：取该检索器自己的 top-N 候选做
  cross-encoder 重排，报告「X 检索器 + reranker_Y」组合指标，并与 X 单独对比
  （增益 = ΔRecall@1 / ΔMRR），避免混合候选池覆盖不足导致 reranker 无从选起。

指标
----
``Recall@K`` (K=1,3,5,10)、``MRR@10``、``nDCG@K`` (K=5,10)，语料级取均值，
并按 ``arxiv`` / ``industry`` 两组分别聚合。embedding 向量按模型缓存到
``data/embed_cache/``，重跑可续跑、不重复花钱。

用法
----
    python scripts/benchmark_retrieval.py --out data/benchmark_results.json
    python scripts/benchmark_retrieval.py --limit 20      # 先小批量自测
    python scripts/benchmark_retrieval.py --no-rerank     # 只看基础检索器
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

from paper_api.corpus import load_corpus
from paper_api.ir_metrics import mean, mrr_at_k, ndcg_at_k, recall_at_k
from paper_api.model_registry import (
    build_embedder,
    build_reranker,
    get_reachable_embedder_specs,
    get_reachable_reranker_specs,
)
from paper_api.retrieval import BM25Retriever, LocalHashingEmbedder, cosine_similarity

DOC_EMBED_CHARS = 6000
RERANK_DOC_CHARS = 1200
EMBED_CACHE_DIR = "data/embed_cache"
RERANK_CACHE_DIR = "data/rerank_cache"
RERANK_PAUSE = 0.06  # 两次 rerank 调用之间稍作停顿，避免触发限流


class VectorRetrieverById:
    """Dense 检索器：语料向量与查询向量都预先算好，search 只做余弦排序。"""

    def __init__(self, name: str, corpus_vecs: dict[str, list[float]], query_vecs: dict[str, list[float]]) -> None:
        self.name = name
        self.corpus_vecs = corpus_vecs
        self.query_vecs = query_vecs

    def search(self, query_doc_id: str, top_n: int | None = None) -> list[str]:
        qv = self.query_vecs[query_doc_id]
        scored = [(doc_id, cosine_similarity(qv, vec)) for doc_id, vec in self.corpus_vecs.items()]
        scored.sort(key=lambda item: -item[1])
        ranked = [doc_id for doc_id, _ in scored]
        return ranked[:top_n] if top_n else ranked


class BM25ById:
    """BM25 检索器包装：查询来自文档标题。"""

    def __init__(self, name: str, bm25: BM25Retriever, titles: dict[str, str]) -> None:
        self.name = name
        self.bm25 = bm25
        self.titles = titles

    def search(self, query_doc_id: str, top_n: int | None = None) -> list[str]:
        return [doc_id for doc_id, _ in self.bm25.search(self.titles[query_doc_id], top_n)]


def embed_corpus_and_queries(spec, doc_ids: list[str], texts: dict[str, str], titles: dict[str, str]) -> tuple[dict, dict]:
    """对指定模型算语料向量与查询向量，按模型缓存、可续跑。返回 (corpus_vecs, query_vecs)。"""
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(EMBED_CACHE_DIR, f"{spec.name}.json")
    cache: dict[str, list[float]] = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as handle:
            cache = json.load(handle)

    client = build_embedder(spec)
    corpus_vecs: dict[str, list[float]] = {}
    query_vecs: dict[str, list[float]] = {}

    for doc_id in doc_ids:
        if doc_id in cache:
            corpus_vecs[doc_id] = cache[doc_id]
    for doc_id in doc_ids:
        if doc_id in corpus_vecs:
            continue
        corpus_vecs[doc_id] = client.embed(texts[doc_id][:DOC_EMBED_CHARS]).vector
        cache[doc_id] = corpus_vecs[doc_id]

    for doc_id in doc_ids:
        qkey = f"q::{doc_id}"
        if qkey in cache:
            query_vecs[doc_id] = cache[qkey]
    for doc_id in doc_ids:
        if doc_id in query_vecs:
            continue
        query_vecs[doc_id] = client.embed(titles[doc_id]).vector
        cache[qkey] = query_vecs[doc_id]

    with open(cache_file, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False)
    return corpus_vecs, query_vecs


def evaluate(ranked_per_query: dict[str, list[str]], ks: list[int]) -> dict:
    out: dict[int, dict[str, float]] = {}
    for k in ks:
        recalls = [recall_at_k(ranked, {qid}, k) for qid, ranked in ranked_per_query.items()]
        mrrs = [mrr_at_k(ranked, {qid}, k) for qid, ranked in ranked_per_query.items()]
        ndcgs = [ndcg_at_k(ranked, {qid}, k) for qid, ranked in ranked_per_query.items()]
        out[k] = {"recall": mean(recalls), "mrr": mean(mrrs), "ndcg": mean(ndcgs)}
    return out


def evaluate_subset(ranked_per_query: dict[str, list[str]], sources: dict[str, str], source: str, ks: list[int]) -> dict:
    subset = {qid: ranked for qid, ranked in ranked_per_query.items() if sources.get(qid) == source}
    return evaluate(subset, ks)


def build_base_retrievers(doc_ids, texts, titles) -> list[tuple[str, object]]:
    """返回 [(name, retriever)]，retriever.search(query_doc_id, top_n) -> [doc_id]。"""
    retrievers: list[tuple[str, object]] = []

    # 离线：本地哈希基线
    local = LocalHashingEmbedder()
    local_corpus = {doc_id: local.embed(texts[doc_id]).vector for doc_id in doc_ids}
    local_query = {doc_id: local.embed(titles[doc_id]).vector for doc_id in doc_ids}
    retrievers.append(("local-hashing", VectorRetrieverById("local-hashing", local_corpus, local_query)))

    # 离线：BM25
    bm25 = BM25Retriever().fit([(doc_id, texts[doc_id]) for doc_id in doc_ids])
    retrievers.append(("bm25", BM25ById("bm25", bm25, titles)))

    # 联网：各可达 embedding 模型
    for spec in get_reachable_embedder_specs():
        corpus_vecs, query_vecs = embed_corpus_and_queries(spec, doc_ids, texts, titles)
        retrievers.append((spec.name, VectorRetrieverById(spec.name, corpus_vecs, query_vecs)))
        print(f"  [embed] {spec.name}: corpus={len(corpus_vecs)} queries={len(query_vecs)}", flush=True)

    return retrievers


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus-level multi-model retrieval benchmark")
    parser.add_argument("--out", default="data/benchmark_results.json")
    parser.add_argument("--ks", default="1,3,5,10", help="Recall/MRR/nDCG 的 K 列表")
    parser.add_argument("--candidates", type=int, default=30, help="每个基础检索器喂给 reranker 的候选数")
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 篇文档（自测用）")
    parser.add_argument("--no-rerank", action="store_true", help="跳过 reranker 评测")
    args = parser.parse_args()

    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    max_k = max(ks)

    print("加载语料…", flush=True)
    docs = load_corpus()
    if args.limit:
        docs = docs[: args.limit]
    doc_ids = [d.doc_id for d in docs]
    texts = {d.doc_id: d.text for d in docs}
    titles = {d.doc_id: d.title for d in docs}
    sources = {d.doc_id: d.source for d in docs}
    print(f"语料文档数：{len(doc_ids)}（arxiv={sum(1 for s in sources.values() if s=='arxiv')}，"
          f"industry={sum(1 for s in sources.values() if s=='industry')}）", flush=True)

    print("构建基础检索器（本地哈希 / BM25 / 各 embedding）…", flush=True)
    base_retrievers = build_base_retrievers(doc_ids, texts, titles)

    overall: dict[str, dict] = {}
    by_source: dict[str, dict[str, dict]] = {"arxiv": {}, "industry": {}}

    for name, retriever in base_retrievers:
        ranked = {qid: retriever.search(qid) for qid in doc_ids}
        overall[name] = evaluate(ranked, ks)
        by_source["arxiv"][name] = evaluate_subset(ranked, sources, "arxiv", ks)
        by_source["industry"][name] = evaluate_subset(ranked, sources, "industry", ks)
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
                ranked: dict[str, list[str]] = {}
                errs = 0
                pending = 0
                for idx, qid in enumerate(doc_ids):
                    if qid in combo_cache:
                        ranked[qid] = combo_cache[qid]
                        continue
                    pending += 1
                    pool_list = retriever.search(qid, args.candidates)
                    if not pool_list:
                        ranked[qid] = []
                        combo_cache[qid] = []
                        continue
                    docs_text = [texts[pid][:RERANK_DOC_CHARS] for pid in pool_list]
                    try:
                        hits = reranker.rerank(titles[qid], docs_text)
                        ranked[qid] = [pool_list[hit.index] for hit in hits]
                    except Exception:  # noqa: BLE001 - 单条失败不中断整体
                        errs += 1
                        ranked[qid] = pool_list  # 退回候选池顺序
                    combo_cache[qid] = ranked[qid]
                    if idx % 10 == 0:  # 周期落盘，超时被杀也能续跑
                        with open(cache_file, "w", encoding="utf-8") as handle:
                            json.dump(combo_cache, handle, ensure_ascii=False)
                    time.sleep(RERANK_PAUSE)
                with open(cache_file, "w", encoding="utf-8") as handle:
                    json.dump(combo_cache, handle, ensure_ascii=False)
                total_errors += errs
                combos[base_name] = {
                    "overall": evaluate(ranked, ks),
                    "by_source": {
                        "arxiv": evaluate_subset(ranked, sources, "arxiv", ks),
                        "industry": evaluate_subset(ranked, sources, "industry", ks),
                    },
                }
                base_r1 = overall[base_name][1]["recall"]
                combo_r1 = combos[base_name]["overall"][1]["recall"]
                cached = len(doc_ids) - pending
                print(f"    {spec.name} + {base_name}: R@1 {base_r1:.3f}→{combo_r1:.3f} "
                      f"(Δ{combo_r1 - base_r1:+.3f}) 缓存{cached} 新算{pending} errors={errs}", flush=True)
            reranker_section[spec.name] = {"combos": combos, "errors": total_errors}

    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "ks": ks,
            "candidates": args.candidates,
            "doc_embed_chars": DOC_EMBED_CHARS,
            "rerank_doc_chars": RERANK_DOC_CHARS,
            "n_documents": len(doc_ids),
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
