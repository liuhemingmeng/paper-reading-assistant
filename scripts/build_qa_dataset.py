#!/usr/bin/env python
"""用 LLM 把语料证据段落改写成自然语言问题，构造真实问答数据集。

这个数据集是后续两个评测的共同输入：

* ``benchmark_qa_retrieval.py``：自然语言问题驱动 chunk 级检索 + reranker 基准，
  正样本 = 问题所属文档。
* ``evaluate_answers_corpus.py``：端到端答案质量闭环，问题 → 检索证据 → LLM 回答
  → faithfulness 裁判。

设计要点
--------
对每篇文档取开头一段作为「证据」，让 LLM 生成**只能由该证据回答**的自然语言问题
（不直接抄句子、用领域术语改写）。这样得到的问题是真实的「用户提问」分布，而非
已知项检索里"用标题找自己"的词汇泄漏题——reranker 的增益在这里才会显现。

联网生成一次，结果缓存到 ``data/qa_dataset.json``，重跑不重复花钱。

用法
----
    python scripts/build_qa_dataset.py
    python scripts/build_qa_dataset.py --per-source 10 --limit 20   # 小批量自测
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_api.corpus import load_corpus
from paper_api.model_registry import build_llm_client, get_reachable_llm_specs

OUT = "data/qa_dataset.json"
EVIDENCE_CHARS = 1500
PER_SOURCE = 15  # arxiv / industry 各取 15 篇 → 30 个 QA 对
GEN_MODEL = "volcano-deepseek-v4-flash"
PAUSE = 0.2


def detect_lang(text: str) -> str:
    cn = len(re.findall(r"[\u4e00-\u9fff]", text))
    return "zh" if cn / max(1, len(text)) > 0.25 else "en"


def pick_generator() -> tuple[object, str]:
    specs = get_reachable_llm_specs()
    if not specs:
        raise SystemExit("没有可达 LLM（先跑 scripts/smoke_llms.py 确认 key 有效）")
    for spec in specs:
        if spec.name == GEN_MODEL:
            return build_llm_client(spec), spec.name
    return build_llm_client(specs[0]), specs[0].name


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real-QA dataset from corpus via LLM")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--per-source", type=int, default=PER_SOURCE)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 篇（自测用）")
    args = parser.parse_args()

    docs = load_corpus()
    arxiv = [d for d in docs if d.source == "arxiv" and len(d.text) >= 600]
    industry = [d for d in docs if d.source == "industry" and len(d.text) >= 600]
    selected = arxiv[: args.per_source] + industry[: args.per_source]
    if args.limit:
        selected = selected[: args.limit]

    client, gen_name = pick_generator()
    items: list[dict] = []
    print(
        f"生成 QA：选 {len(selected)} 篇（arxiv={len(arxiv[:args.per_source])} "
        f"industry={len(industry[:args.per_source])}），生成模型={gen_name}",
        flush=True,
    )

    for i, doc in enumerate(selected, 1):
        evidence = doc.text[:EVIDENCE_CHARS]
        lang = detect_lang(evidence)
        question = ""
        try:
            insight = client.generate(evidence)
            if insight.questions:
                question = insight.questions[0].strip()
        except Exception as error:  # noqa: BLE001 - 单篇失败不中断整体
            print(f"  [{doc.doc_id}] 生成失败：{error}", flush=True)
        if not question:
            continue
        items.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "source": doc.source,
                "lang": lang,
                "evidence": evidence,
                "question": question,
            }
        )
        print(f"  {i}/{len(selected)} {doc.doc_id} [{lang}] {question[:60]}", flush=True)
        time.sleep(PAUSE)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(
        {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generator": gen_name,
            "evidence_chars": EVIDENCE_CHARS,
            "items": items,
        },
        open(args.out, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    print(f"\n写入 {args.out}：{len(items)} 条 QA（{sum(1 for it in items if it['lang']=='zh')} 中文 / "
          f"{sum(1 for it in items if it['lang']=='en')} 英文）")


if __name__ == "__main__":
    main()
