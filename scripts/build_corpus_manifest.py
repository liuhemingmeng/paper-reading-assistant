"""Aggregate the per-subcorpus manifests into one corpus_manifest.json.

Run after the corpus-gathering subagents finish. Reads the five
``manifest.json`` files under ``data/corpus/<subcorpus>/`` and merges them
into ``data/corpus/corpus_manifest.json`` plus a human-readable
``data/corpus/README.md``. The merged manifest is the input for the
multi-model / reranker retrieval benchmark.
"""

from __future__ import annotations

import json
import os
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
SUBCORPORA = [
    "arxiv_retrieval",
    "arxiv_llm",
    "arxiv_embedding",
    "industry_fin",
    "industry_tech",
]


def main() -> None:
    documents: list[dict] = []
    per_subcorpus: dict[str, int] = {}

    for sub in SUBCORPORA:
        path = os.path.join(BASE, sub, "manifest.json")
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        items = manifest.get("documents", [])
        source = manifest.get("source", "unknown")
        group = manifest.get("field") or manifest.get("category") or "unknown"
        for item in items:
            item["source"] = source
            item["group"] = group
        per_subcorpus[sub] = len(items)
        documents.extend(items)

    total = len(documents)
    arxiv = sum(1 for d in documents if d.get("source") == "arxiv")
    industry = total - arxiv

    by_group = Counter(d.get("group", "unknown") for d in documents)
    by_status = Counter(d.get("status", "unknown") for d in documents)
    downloaded = sum(1 for d in documents if d.get("status") == "downloaded")

    merged = {
        "generated_at": "2026-08-15",
        "total": total,
        "arxiv_total": arxiv,
        "industry_total": industry,
        "downloaded": downloaded,
        "link_only": total - downloaded,
        "per_subcorpus": per_subcorpus,
        "by_group": dict(by_group),
        "by_status": dict(by_status),
        "documents": documents,
    }

    out_manifest = os.path.join(BASE, "corpus_manifest.json")
    with open(out_manifest, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=2)

    readme = os.path.join(BASE, "README.md")
    lines = [
        "# 评测语料库 Corpus",
        "",
        f"总计 **{total}** 篇真实公开文档（arXiv 学术 {arxiv} + 行业资料 {industry}），"
        "用于 RAG 检索的多模型 / reranker 评测基准。",
        "",
        "## 子语料分布",
        "",
    ]
    for sub, n in per_subcorpus.items():
        lines.append(f"- `{sub}`: {n} 篇")
    lines += [
        "",
        "## 分组（arXiv field / 行业类别）",
        "",
    ]
    for cat, n in sorted(by_group.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {cat}: {n}")
    lines += [
        "",
        f"## 下载状态：downloaded {downloaded} / link_only {total - downloaded}",
        "",
        "## 用途",
        "",
        "配合 `scripts/compare_embeddings.py` 扩展为多模型注册表，对 "
        "local-hashing / BM25 / 火山 doubao-embedding / 其他 embedding 模型 "
        "× 是否接 reranker 跑 Recall@K / MRR / nDCG@K，并聚合到语料级。",
        "",
        "## 说明",
        "",
        "- PDF/HTML 已存于各子目录（被 .gitignore 忽略，不进仓库）；"
        "`link_only` 条目仅记录 URL，需手动或经代理获取原文。",
        "- 所有来源均为真实公开链接，未编造；部分券商研报、运营商招标文件、"
        "上交所招股书因登录墙 / 免责页拦截记为 link_only。",
        "- 每条文档的 `local_path` 为相对 `data/corpus/` 的路径，供解析与切分使用。",
        "",
    ]
    with open(readme, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print("TOTAL", total, "| arxiv", arxiv, "| industry", industry)
    print("per_subcorpus", per_subcorpus)
    print("by_status", dict(by_status))
    print("by_group", dict(by_group))
    print("wrote", out_manifest, "and", readme)


if __name__ == "__main__":
    main()
