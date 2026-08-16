#!/usr/bin/env python
"""语料级端到端答案质量闭环 + 4 LLM 横评。

覆盖两个相辅相成的目标：

* 任务A（真实 LLM 答案质量闭环）：用强 embedding 检索到 evidence，让 LLM 只依据
  evidence 回答，再交给独立 faithfulness 裁判判断答案是否「只来自 evidence、无幻觉」。
* 任务C（4 LLM 选型对比）：对同一个固定 QA 集，用 4 个可达 LLM 各生成答案，对比
  延迟、答案长度（成本代理）与 faithfulness 忠实率，形成可写进简历的选型报告。

流程：QA 问题 → 强 embedding（bge-m3）检索 top-k chunk 作 evidence → 各 LLM 基于
evidence 回答 → 独立 judge 评忠实度。embedding 向量与答案/judge 结果均缓存可续跑。

用法
----
    python scripts/evaluate_answers_corpus.py --out data/answer_eval.json
    python scripts/evaluate_answers_corpus.py --limit 10           # 先小批量
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
from paper_api.ir_metrics import mean, mrr_at_k, recall_at_k
from paper_api.model_registry import (
    build_embedder,
    build_llm_client,
    get_reachable_embedder_specs,
    get_reachable_llm_specs,
)
from paper_api.retrieval import cosine_similarity
from paper_api.answer_evaluation import OpenAICompatibleFaithfulnessJudge
from paper_api.cache_io import atomic_write_json, load_json_cache

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 120
QA_PATH = "data/qa_dataset.json"
EMBED_CACHE_DIR = "data/embed_cache_qa"
ANSWER_CACHE_DIR = "data/answer_cache"
JUDGE_CACHE_DIR = "data/judge_cache"
RETRIEVE_K = 5
EVIDENCE_CHARS = 3000
INDEX_MODEL = "siliconflow-bge-m3"  # 用最强文本 embedding 建索引
JUDGE_MODEL = "siliconflow-qwen3.5-35b"  # 不同家族的裁判，避免同模型自评偏差
PAUSE = 0.05


def chunk_id_to_doc(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def build_index(spec, chunk_ids, chunk_texts, questions) -> tuple[dict, dict]:
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(EMBED_CACHE_DIR, f"{spec.name}.json")
    cache: dict = load_json_cache(cache_file)
    client = build_embedder(spec)
    chunk_vecs: dict[str, list[float]] = {}
    for cid in chunk_ids:
        if cid in cache:
            chunk_vecs[cid] = cache[cid]
    done = 0
    for cid in chunk_ids:
        if cid in chunk_vecs:
            continue
        chunk_vecs[cid] = client.embed(chunk_texts[cid][:4000]).vector
        cache[cid] = chunk_vecs[cid]
        done += 1
        if done % 300 == 0:
            atomic_write_json(cache_file, cache)
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


def pick_judge() -> tuple[object, str]:
    for spec in get_reachable_llm_specs():
        if spec.name == JUDGE_MODEL:
            return OpenAICompatibleFaithfulnessJudge(spec.base_url, spec.api_key, spec.model), spec.name
    judge = OpenAICompatibleFaithfulnessJudge.from_environment()
    return judge, "default-env"


def _safe(name: str) -> str:
    return name.replace("/", "__")


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus-level answer-quality loop + 4-LLM comparison")
    parser.add_argument("--out", default="data/answer_eval.json")
    parser.add_argument("--qa", default=QA_PATH)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--k", type=int, default=RETRIEVE_K)
    args = parser.parse_args()

    docs = load_corpus()
    if args.limit:
        docs = docs[: args.limit]
    chunks = chunk_corpus(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    chunk_ids = [c.chunk_id for c in chunks]
    chunk_texts = {c.chunk_id: c.text for c in chunks}
    print(f"chunk 数 {len(chunk_ids)} / 文档 {len(docs)}", flush=True)

    with open(args.qa, encoding="utf-8") as handle:
        qa = json.load(handle)
    items = qa["items"]
    if args.limit:
        items = items[: args.limit]
    questions = [it["question"] for it in items]
    positives = {it["question"]: it["doc_id"] for it in items}
    sources = {it["question"]: it["source"] for it in items}

    idx_spec = next((s for s in get_reachable_embedder_specs() if s.name == INDEX_MODEL), None)
    if idx_spec is None:
        raise SystemExit(f"索引模型 {INDEX_MODEL} 不可达（先跑 scripts/smoke_models.py）")
    chunk_vecs, query_vecs = build_index(idx_spec, chunk_ids, chunk_texts, questions)
    print(f"索引模型 {idx_spec.name}：chunks={len(chunk_vecs)} queries={len(query_vecs)}", flush=True)

    def search(question: str) -> list[str]:
        qv = query_vecs[question]
        scored = sorted(
            ((cid, cosine_similarity(qv, chunk_vecs[cid])) for cid in chunk_ids),
            key=lambda item: -item[1],
        )
        return [cid for cid, _ in scored]

    llm_specs = get_reachable_llm_specs()
    llm_names = [s.name for s in llm_specs]
    clients = [build_llm_client(s) for s in llm_specs]
    judge, judge_name = pick_judge()
    print(f"回答 LLM：{llm_names}\njudge：{judge_name}", flush=True)

    os.makedirs(ANSWER_CACHE_DIR, exist_ok=True)
    os.makedirs(JUDGE_CACHE_DIR, exist_ok=True)
    answer_cache: dict[str, dict] = {}
    judge_cache: dict[str, dict] = {}
    for n in llm_names:
        p_a = f"{ANSWER_CACHE_DIR}/{_safe(n)}.json"
        answer_cache[n] = load_json_cache(p_a)
        p_j = f"{JUDGE_CACHE_DIR}/{_safe(n)}.json"
        judge_cache[n] = load_json_cache(p_j)

    per_llm: dict[str, dict] = {
        n: {
            "latency": [],
            "chars": [],
            "faithful": 0,
            "n": 0,
            "by_source": {"arxiv": {"faithful": 0, "n": 0}, "industry": {"faithful": 0, "n": 0}},
        }
        for n in llm_names
    }
    retr_recall: list[float] = []
    retr_mrr: list[float] = []
    retr_by_source: dict[str, list[float]] = {"arxiv": [], "industry": []}
    samples: list[dict] = []

    for i, it in enumerate(items, 1):
        q = it["question"]
        gold = it["doc_id"]
        src = it["source"]
        ranked = [chunk_id_to_doc(c) for c in search(q)]
        r1 = recall_at_k(ranked, {gold}, 1)
        mrr = mrr_at_k(ranked, {gold}, 10)
        retr_recall.append(r1)
        retr_mrr.append(mrr)
        retr_by_source[src].append(r1)

        top = search(q)[: args.k]
        evidence = "\n\n".join(chunk_texts[c][:1200] for c in top)[:EVIDENCE_CHARS]
        sample = {"question": q, "gold": gold, "source": src, "evidence_preview": evidence[:300], "answers": {}}

        for n, client in zip(llm_names, clients):
            if q in answer_cache[n]:
                ans_rec = answer_cache[n][q]
            else:
                t0 = time.time()
                try:
                    answer_text = client.answer(q, evidence).answer
                except Exception as error:  # noqa: BLE001 - 单条失败不中断
                    answer_text = f"[ERROR] {error}"
                ans_rec = {"answer": answer_text, "latency": round(time.time() - t0, 2), "chars": len(answer_text)}
                answer_cache[n][q] = ans_rec
                atomic_write_json(f"{ANSWER_CACHE_DIR}/{_safe(n)}.json", answer_cache[n])

            if q in judge_cache[n]:
                jv = judge_cache[n][q]
            else:
                try:
                    verdict = judge.judge(q, evidence, ans_rec["answer"])
                    jv = {"faithful": verdict.faithful, "reason": verdict.reason}
                except Exception as error:  # noqa: BLE001
                    jv = {"faithful": None, "reason": f"[judge error] {error}"}
                judge_cache[n][q] = jv
                atomic_write_json(f"{JUDGE_CACHE_DIR}/{_safe(n)}.json", judge_cache[n])

            per_llm[n]["latency"].append(ans_rec["latency"])
            per_llm[n]["chars"].append(ans_rec["chars"])
            if jv["faithful"]:
                per_llm[n]["faithful"] += 1
                per_llm[n]["by_source"][src]["faithful"] += 1
            per_llm[n]["n"] += 1
            per_llm[n]["by_source"][src]["n"] += 1
            sample["answers"][n] = {"answer": ans_rec["answer"][:400], "faithful": jv["faithful"], "reason": jv.get("reason", "")}

        if i <= 6:
            samples.append(sample)
        print(f"  {i}/{len(items)} {src} R@1={r1:.2f} | " + " ".join(
            f"{n.split('-')[-1]}:{per_llm[n]['faithful']}/{per_llm[n]['n']}" for n in llm_names
        ), flush=True)
        time.sleep(PAUSE)

    def _rate(d: dict) -> float | None:
        return round(d["faithful"] / d["n"], 3) if d["n"] else None

    def _rate_src(d: dict, s: str) -> float | None:
        return round(d["by_source"][s]["faithful"] / d["by_source"][s]["n"], 3) if d["by_source"][s]["n"] else None

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "index_model": INDEX_MODEL,
            "judge_model": judge_name,
            "retrieve_k": args.k,
            "evidence_chars": EVIDENCE_CHARS,
            "n_questions": len(items),
        },
        "retrieval": {
            "recall_at_1": round(mean(retr_recall), 3),
            "mrr_at_10": round(mean(retr_mrr), 3),
            "by_source": {
                "arxiv": round(mean(retr_by_source["arxiv"]), 3),
                "industry": round(mean(retr_by_source["industry"]), 3),
            },
        },
        "llms": {
            n: {
                "n": d["n"],
                "avg_latency_s": round(mean(d["latency"]), 2),
                "avg_answer_chars": round(mean(d["chars"])),
                "faithful_rate": _rate(d),
                "by_source": {"arxiv": _rate_src(d, "arxiv"), "industry": _rate_src(d, "industry")},
            }
            for n, d in per_llm.items()
        },
        "samples": samples,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n写入 {args.out}")
    _print(report)


def _print(report: dict) -> None:
    r = report["retrieval"]
    print(f"\n[检索] Recall@1={r['recall_at_1']:.3f}  MRR@10={r['mrr_at_10']:.3f}  "
          f"(arxiv={r['by_source']['arxiv']:.3f} industry={r['by_source']['industry']:.3f})")
    hdr = f"{'LLM':<30}{'avg_lat_s':>11}{'avg_chars':>11}{'faithful%':>11}"
    print("\n[LLM 横评]")
    print(hdr)
    print("-" * len(hdr))
    for n, d in report["llms"].items():
        fr = d["faithful_rate"]
        print(f"{n:<30}{d['avg_latency_s']:>11}{d['avg_answer_chars']:>11}"
              f"{(f'{fr:.3f}' if fr is not None else 'n/a'):>11}")


if __name__ == "__main__":
    main()
