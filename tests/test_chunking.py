"""Offline tests for the corpus chunking helper."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_api.chunking import CorpusChunk, chunk_corpus, chunk_document
from paper_api.corpus import CorpusDoc


def test_short_text_is_single_chunk() -> None:
    chunks = chunk_document("d1", "短文本", size=900, overlap=120)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "d1#0"
    assert chunks[0].doc_id == "d1"
    assert chunks[0].text == "短文本"


def test_long_text_overlap_split() -> None:
    text = "x" * 2500
    chunks = chunk_document("d2", text, size=900, overlap=120)
    # step = 900 - 120 = 780; windows: 0-900, 780-1680, 1560-2460, 2340-2500 -> 4 chunks
    assert len(chunks) == 4
    assert all(isinstance(c, CorpusChunk) and c.doc_id == "d2" for c in chunks)
    assert {c.chunk_id for c in chunks} == {"d2#0", "d2#1", "d2#2", "d2#3"}
    # overlapping windows still cover the full original text as a substring
    assert text in "".join(c.text for c in chunks)


def test_chunk_ids_are_unique_and_ordered() -> None:
    text = "y" * 5000
    chunks = chunk_document("d3", text, size=1000, overlap=150)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_chunk_corpus_annotates_doc_id_and_source() -> None:
    docs = [
        CorpusDoc("a", "Title A", "this is a fairly long document body " * 30, "arxiv", "g1"),
        CorpusDoc("b", "Title B", "短", "industry", "g2"),
    ]
    chunks = chunk_corpus(docs, size=900, overlap=120)
    assert any(c.doc_id == "a" for c in chunks)
    assert any(c.doc_id == "b" for c in chunks)
    assert all(c.source in ("arxiv", "industry") for c in chunks)
    # the short doc stays one chunk; the long one is split
    assert sum(1 for c in chunks if c.doc_id == "b") == 1
    assert sum(1 for c in chunks if c.doc_id == "a") > 1
