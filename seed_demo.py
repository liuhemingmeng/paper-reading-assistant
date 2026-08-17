"""Seed the local demo corpus: 3 themed papers, indexed with bge-m3.

Run while the API is up (python -m paper_api). Builds papers + uploads PDFs
+ indexes once, so the demo is ready with zero extra config.
"""
from __future__ import annotations

import io

import fitz
import httpx

BASE = "http://127.0.0.1:8011"


PAPERS = [
    {
        "title": "Retrieval-Augmented Generation: Grounding LLMs with External Knowledge",
        "authors": "Demo Author",
        "abstract": "RAG couples a retriever with a generator to ground answers in retrieved evidence.",
        "sections": [
            ("1. What is RAG",
             "Retrieval-Augmented Generation combines an information retriever with a text generator. "
             "Instead of answering from parameters alone, the model first fetches relevant passages from a corpus "
             "and then conditions its answer on that evidence, which reduces hallucination."),
            ("2. System architecture",
             "A RAG pipeline has two stages. The retriever embeds the query and searches a vector store for the "
             "nearest chunks, and the generator is a language model that receives the retrieved passages inside the "
             "prompt. Citations link each claim back to the source chunk and page."),
            ("3. Why grounding matters",
             "Grounding answers in retrieved text gives provenance and makes responses verifiable. Users can open the "
             "cited paper section to check the claim, which is essential for research and enterprise assistants."),
            ("4. Chunking strategy",
             "Documents are split with a sliding window of about 1000 characters and a small overlap so that sentences "
             "are not cut across chunk boundaries. Each chunk is embedded independently and stored with its metadata."),
        ],
    },
    {
        "title": "Attention Mechanism and the Transformer Architecture",
        "authors": "Demo Author",
        "abstract": "Self-attention lets every token weigh the importance of every other token in the sequence.",
        "sections": [
            ("1. Scaled dot-product attention",
             "Scaled dot-product attention computes scores as softmax(Q K^T / sqrt(d)) and uses them to take a weighted "
             "sum of the value vectors V. The scaling by sqrt(d) keeps gradients stable for large dimensions."),
            ("2. Self-attention",
             "In self-attention the query, key and value all come from the same sequence, so each token can attend to "
             "every other token. This captures long-range dependencies without recurrence or convolution."),
            ("3. Multi-head attention",
             "Multi-head attention runs several attention computations in parallel, each with its own learned projection. "
             "Different heads specialize in different kinds of relationships, such as syntax or coreference."),
            ("4. The Transformer",
             "A Transformer stacks multi-head attention with feed-forward layers and residual connections. Removing "
             "recurrence allows full parallel training and became the backbone of modern language models."),
        ],
    },
    {
        "title": "Vector Databases and Approximate Nearest Neighbor Search",
        "authors": "Demo Author",
        "abstract": "Dense embeddings enable semantic search backed by ANN indexes such as HNSW.",
        "sections": [
            ("1. Dense embeddings",
             "An embedding model maps text to a fixed-length dense vector that captures semantic meaning. Similar "
             "sentences land close together in the vector space, enabling semantic rather than keyword search."),
            ("2. Cosine similarity",
             "Cosine similarity is the dot product of two L2-normalized vectors and ranges from -1 to 1. Retrieval "
             "ranks chunks by cosine similarity between the query vector and each stored chunk vector."),
            ("3. Approximate nearest neighbor indexes",
             "Exact search is too slow at scale, so ANN indexes like HNSW and IVF trade a little recall for large "
             "speedups. They organize vectors into graphs or coarse clusters for fast candidate retrieval."),
            ("4. pgvector",
             "pgvector is a Postgres extension that stores embeddings and supports cosine and L2 distance operators. "
             "With an HNSW index it serves milli-second nearest-neighbor queries inside an existing relational database."),
        ],
    },
]


def make_pdf(sections) -> bytes:
    doc = fitz.open()
    for title, body in sections:
        page = doc.new_page()
        page.insert_text((72, 72), title, fontsize=14)
        page.insert_text((72, 104), body, fontsize=11)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def main() -> None:
    for paper in PAPERS:
        created = httpx.post(
            f"{BASE}/papers",
            json={"title": paper["title"], "authors": paper["authors"], "abstract": paper["abstract"]},
        )
        created.raise_for_status()
        pid = created.json()["id"]
        pdf_bytes = make_pdf(paper["sections"])
        up = httpx.post(
            f"{BASE}/papers/{pid}/document",
            files={"file": ("paper.pdf", pdf_bytes, "application/pdf")},
        )
        up.raise_for_status()
        idx = httpx.post(f"{BASE}/papers/{pid}/retrieval:index")
        idx.raise_for_status()
        print(f"paper {pid}: indexed {idx.json()}")


if __name__ == "__main__":
    main()
