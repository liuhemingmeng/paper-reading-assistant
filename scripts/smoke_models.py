"""Smoke-test every configured embedding and reranker model against the live API.

This is a connectivity + contract check, not a quality benchmark. For each
model it performs one real call and reports the essentials (dimension for
embedders, ranked scores for rerankers) so we can confirm keys, endpoints, and
response shapes before wiring the models into the corpus benchmark.

Run with the proxy unset if the sandbox intercepts outbound traffic::

    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    python scripts/smoke_models.py --out data/model_smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_api.model_registry import get_embedder_specs, get_reranker_specs, build_embedder, build_reranker

SAMPLE_TEXT = (
    "Retrieval-augmented generation grounds an answer in retrieved documents so the "
    "model cites evidence instead of guessing from parametric memory."
)
SAMPLE_QUERY = "How does retrieval-augmented generation reduce hallucination?"
SAMPLE_DOCS = [
    "RAG grounds answers in retrieved documents to reduce hallucination and add citations.",
    "Dense vector search finds nearest neighbors with cosine similarity over embeddings.",
    "Convolutional neural networks dominate image classification benchmarks.",
    "A reranker reorders a small candidate set by finer-grained relevance than the retriever.",
]


def smoke_embedders() -> list[dict]:
    rows: list[dict] = []
    for spec in get_embedder_specs():
        entry = {"name": spec.name, "model": spec.model, "endpoint": spec.endpoint, "status": "error", "detail": ""}
        try:
            embedded = build_embedder(spec).embed(SAMPLE_TEXT)
            dims = len(embedded.vector)
            entry.update(
                status="ok",
                dimensions=dims,
                model_returned=embedded.model,
                first_values=[round(value, 5) for value in embedded.vector[:3]],
            )
        except Exception as error:  # noqa: BLE001 - smoke must report, not crash
            entry["detail"] = f"{type(error).__name__}: {error}"
        rows.append(entry)
        print(f"[embedder] {entry['name']:<32} {entry['status']:<5} "
              f"{('dim=' + str(entry.get('dimensions'))) if entry['status'] == 'ok' else entry['detail']}")
    return rows


def smoke_rerankers() -> list[dict]:
    rows: list[dict] = []
    for spec in get_reranker_specs():
        entry = {"name": spec.name, "model": spec.model, "status": "error", "detail": ""}
        try:
            hits = build_reranker(spec).rerank(SAMPLE_QUERY, SAMPLE_DOCS)
            entry.update(
                status="ok",
                ranked=[{"index": hit.index, "score": round(hit.score, 5)} for hit in hits],
                top_doc_index=hits[0].index if hits else None,
            )
        except Exception as error:  # noqa: BLE001 - smoke must report, not crash
            entry["detail"] = f"{type(error).__name__}: {error}"
        rows.append(entry)
        if entry["status"] == "ok":
            ranked = ", ".join(f"#{h['index']}:{h['score']}" for h in entry["ranked"])
            print(f"[reranker] {entry['name']:<32} ok      top_doc={entry['top_doc_index']}  {ranked}")
        else:
            print(f"[reranker] {entry['name']:<32} ERROR   {entry['detail']}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test configured embedding and reranker models.")
    parser.add_argument("--out", default=None, help="Write the full JSON result to this path.")
    args = parser.parse_args()

    embedder_rows = smoke_embedders()
    reranker_rows = smoke_rerankers()

    ok = sum(1 for row in embedder_rows + reranker_rows if row["status"] == "ok")
    total = len(embedder_rows) + len(reranker_rows)
    print(f"\nSummary: {ok}/{total} models reachable")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {"ok": ok, "total": total},
        "embedders": embedder_rows,
        "rerankers": reranker_rows,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Full JSON written to {args.out}")

    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()
