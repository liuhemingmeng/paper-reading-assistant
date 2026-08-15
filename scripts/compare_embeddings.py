"""Compare the local hashing baseline against a real semantic embedder.

This script runs a controlled retrieval benchmark so the difference between a
dependency-free lexical baseline (``LocalHashingEmbedder``) and a production
semantic embedding model (Volcano Engine ``doubao-embedding-vision`` by
default) is measurable instead of anecdotal.

It builds a synthetic multi-topic document, indexes it with each embedder, then
evaluates source-aware retrieval with Recall@K and MRR. Two question families
are used:

* ``lexical``  -- questions that reuse the source vocabulary (easy for both).
* ``semantic`` -- questions paraphrased with synonyms absent from the source
                  (only a semantic model should localize reliably).

The document text and the evaluation cases are defined inline so the experiment
is fully reproducible. Results are written as JSON (``--out``) and a human
summary is printed to stdout.

Usage::

    # offline baseline only (no embedding keys needed)
    python scripts/compare_embeddings.py

    # real semantic model (needs EMBEDDING_* in .env; unset proxy if blocked)
    python scripts/compare_embeddings.py --out data/embedding_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_api.database import create_engine_for_url, create_session_factory, create_tables
from paper_api.evaluation import EvaluationCase, evaluate_retrieval
from paper_api.embeddings import EmbeddingConfigurationError, OpenAICompatibleEmbedder
from paper_api.models import Paper
from paper_api.pdf_processing import ExtractedPage, chunk_pages, extract_pdf_pages
from paper_api.retrieval import LocalHashingEmbedder, TextEmbedder
from paper_api.services import (
    NoRelevantEvidenceError,
    build_retrieval_index,
    retrieve_chunks,
    save_processed_document,
)

# --------------------------------------------------------------------------- #
# Controlled benchmark document: six pages, one distinct topic per page.
# --------------------------------------------------------------------------- #
DOCUMENT_PAGES: list[str] = [
    """1 Background

Retrieval-augmented generation reduces hallucination by grounding answers in an external corpus instead of relying only on the model's parametric memory. When a language model invents facts not supported by its training data, the output becomes unreliable. Adding a retriever lets the system cite evidence from source documents.

Large language models store knowledge implicitly in weights. This parametric memory is convenient but cannot be updated without retraining and may confidently state incorrect information. Grounding with retrieved passages mitigates these failure modes.""",
    """2 Vector Index

Dense vector search stores each passage as an embedding in a high-dimensional space. To answer a query, the system finds the nearest neighbor to the query vector using approximate nearest neighbor algorithms such as HNSW. Cosine similarity then ranks candidate passages.

An embedding index must support fast lookup at scale. Hierarchical navigable small world graphs organize vectors so that top-k retrieval stays sub-linear. Without an index, scanning every stored vector is too slow for large corpora.""",
    """3 Chunking

A sliding window splits a long document into overlapping segments so that no sentence is cut at a boundary. The token overlap between adjacent chunks preserves local context that would otherwise be lost when a paragraph is divided.

Chunk size trades recall against precision. Small chunks localize evidence but increase the number of vectors to search; large chunks keep context but bury the relevant sentence. Tuning the segment length is a key retrieval hyperparameter.""",
    """4 Evaluation

Ranking quality is measured with recall at k, the fraction of queries whose relevant passage appears in the top k results. Mean reciprocal rank rewards systems that place the correct passage as early as possible in the ranked list.

Ground truth labels map each question to the pages that contain its answer. A retrieval benchmark reports recall and MRR so that different embedding models can be compared on the same queries.""",
    """5 Fine-tuning

Embedding models are trained with a contrastive loss that pulls a query and its positive passage close while pushing negative samples away. A positive pair consists of a question and the passage that answers it.

Hard negative mining selects misleading passages that are similar but irrelevant, strengthening the embedding space. During training, in-batch negatives treat other queries' passages as contrastive examples, improving efficiency.""",
    """6 Deployment

Production retrieval needs high throughput so that many queries are served per second. A dedicated inference server batches embedding requests and runs them on a GPU to keep latency low.

Online serving must balance cost and speed. Caching frequent query embeddings avoids recomputation, while quantization shrinks vector size so the index fits in memory.""",
]

# question, expected page number, question family
EVALUATION_CASES: list[tuple[str, int, str]] = [
    ("What is hallucination in language models?", 1, "lexical"),
    ("What is an approximate nearest neighbor algorithm such as HNSW?", 2, "lexical"),
    ("What is a sliding window in chunking?", 3, "lexical"),
    ("What does recall at k measure?", 4, "lexical"),
    ("What is a contrastive loss in embedding training?", 5, "lexical"),
    ("What is throughput in production retrieval?", 6, "lexical"),
    ("Why does a model sometimes state things that were never in its training data?", 1, "semantic"),
    ("How does a system quickly locate the most similar stored passage to a user question?", 2, "semantic"),
    ("How can text be divided so that context is not lost at the cut point?", 3, "semantic"),
    ("Which metric gives credit when the right document is ranked first?", 4, "semantic"),
    ("During embedding training, what objective brings matching pairs together?", 5, "semantic"),
    ("How can a service answer more requests each second?", 6, "semantic"),
]

KS = [1, 3, 5]


def build_chunks(pdf_path: str | None) -> tuple[list[ExtractedPage], list]:
    if pdf_path:
        pages = extract_pdf_pages(Path(pdf_path))
    else:
        pages = [ExtractedPage(page_number=i, text=text) for i, text in enumerate(DOCUMENT_PAGES, start=1)]
    return pages, chunk_pages(pages)


def insert_paper(engine, pages, chunks) -> int:
    session_factory = create_session_factory(engine)
    with session_factory() as session:
        paper = Paper(
            title="Controlled Retrieval Benchmark",
            authors="WorkBuddy",
            abstract="Synthetic multi-topic document used to compare embedders.",
        )
        session.add(paper)
        session.commit()
        session.refresh(paper)
        save_processed_document(
            session=session,
            paper_id=paper.id,
            original_filename="benchmark.pdf",
            storage_path=Path("/tmp/benchmark.pdf"),
            file_size=0,
            pages=pages,
            chunks=chunks,
        )
        return paper.id


def make_retrieve_pages(session_factory, paper_id, embedder: TextEmbedder):
    def retrieve_pages(question: str, limit: int) -> list[int]:
        with session_factory() as session:
            try:
                return [
                    chunk.page_number
                    for chunk, _ in retrieve_chunks(session, paper_id, question, limit=limit, embedder=embedder)
                ]
            except NoRelevantEvidenceError:
                return []

    return retrieve_pages


def evaluate_embedder(embedder: TextEmbedder, pages, chunks) -> dict:
    engine = create_engine_for_url("sqlite://")
    create_tables(engine)
    session_factory = create_session_factory(engine)
    paper_id = insert_paper(engine, pages, chunks)

    with session_factory() as session:
        model, indexed = build_retrieval_index(session, paper_id, embedder=embedder)

    all_cases = [EvaluationCase(question=q, expected_page_numbers=[p]) for q, p, _ in EVALUATION_CASES]
    lexical_cases = [c for c, (_, _, kind) in zip(all_cases, EVALUATION_CASES) if kind == "lexical"]
    semantic_cases = [c for c, (_, _, kind) in zip(all_cases, EVALUATION_CASES) if kind == "semantic"]

    def summarize(cases):
        return {k: evaluate_retrieval(cases, make_retrieve_pages(session_factory, paper_id, embedder), k) for k in KS}

    overall = summarize(all_cases)
    lexical = summarize(lexical_cases)
    semantic = summarize(semantic_cases)

    per_case = []
    for (question, expected_page, kind) in EVALUATION_CASES:
        case = EvaluationCase(question=question, expected_page_numbers=[expected_page])
        row = {"question": question, "expected_page": expected_page, "kind": kind}
        for k in KS:
            report = evaluate_retrieval([case], make_retrieve_pages(session_factory, paper_id, embedder), k)
            result = report.results[0]
            row[f"k{k}_hit"] = result.hit
            row[f"k{k}_pages"] = result.retrieved_page_numbers
        per_case.append(row)

    return {
        "model": model,
        "indexed_chunks": indexed,
        "metrics": {
            "overall": {k: {"recall": overall[k].recall_at_k, "mrr": overall[k].mean_reciprocal_rank} for k in KS},
            "lexical": {k: {"recall": lexical[k].recall_at_k, "mrr": lexical[k].mean_reciprocal_rank} for k in KS},
            "semantic": {k: {"recall": semantic[k].recall_at_k, "mrr": semantic[k].mean_reciprocal_rank} for k in KS},
        },
        "per_case": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare local vs semantic embedder retrieval quality.")
    parser.add_argument("--pdf", default=None, help="Optional real PDF path; defaults to the synthetic benchmark.")
    parser.add_argument("--out", default=None, help="Write the full JSON result to this path.")
    args = parser.parse_args()

    pages, chunks = build_chunks(args.pdf)
    print(f"Benchmark document: {len(pages)} pages, {len(chunks)} chunks")

    embedders: list[tuple[str, TextEmbedder]] = [("local-hashing", LocalHashingEmbedder())]
    try:
        embedders.append(("volcano-semantic", OpenAICompatibleEmbedder.from_environment()))
    except EmbeddingConfigurationError as error:
        print(f"[warn] real semantic embedder skipped: {error}")

    results: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document": {"pages": len(pages), "chunks": len(chunks)},
        "embedders": {},
    }

    for name, embedder in embedders:
        print(f"\n=== Evaluating embedder: {name} ===")
        report = evaluate_embedder(embedder, pages, chunks)
        results["embedders"][name] = report
        for k in KS:
            m = report["metrics"]["overall"][k]
            print(f"  k={k}: recall={m['recall']:.3f}  mrr={m['mrr']:.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nFull JSON written to {args.out}")


if __name__ == "__main__":
    main()
