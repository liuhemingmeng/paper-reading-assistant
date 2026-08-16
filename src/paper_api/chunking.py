"""把语料文档切成带 doc_id 标注的 chunk，供 chunk 级检索基准使用。

chunk 级检索比「整篇文档」检索更接近真实 RAG：用户问一个具体问题时，命中的
通常是一段而非整篇。这里只做纯文本滑窗切分（与 ``pdf_processing`` 的 PDF 分块
相互独立），每个 chunk 携带所属 ``doc_id``，便于检索命中后回判正样本文档。
"""

from __future__ import annotations

from dataclasses import dataclass

from .corpus import CorpusDoc


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    doc_id: str
    text: str
    source: str
    group: str


def chunk_document(
    doc_id: str,
    text: str,
    size: int = 900,
    overlap: int = 120,
    source: str = "",
    group: str = "",
) -> list[CorpusChunk]:
    """把一篇文档切成等长滑窗 chunk，带 doc_id 标注。

    文本短于 ``size`` 时返回单个 chunk；否则按 ``size - overlap`` 步长滑窗，
    重叠区保证句子不被切断导致语义丢失。每个 chunk_id 形如 ``<doc_id>#<idx>``。
    """
    text = (text or "").strip()
    if len(text) <= size:
        return [CorpusChunk(f"{doc_id}#0", doc_id, text, source, group)]

    chunks: list[CorpusChunk] = []
    step = max(1, size - overlap)
    start = 0
    idx = 0
    while start < len(text):
        end = min(len(text), start + size)
        segment = text[start:end].strip()
        if segment:
            chunks.append(CorpusChunk(f"{doc_id}#{idx}", doc_id, segment, source, group))
            idx += 1
        if end >= len(text):
            break
        start += step
    return chunks


def chunk_corpus(docs: list[CorpusDoc], size: int = 900, overlap: int = 120) -> list[CorpusChunk]:
    """对一批文档批量切分，扁平化为 chunk 列表。"""
    out: list[CorpusChunk] = []
    for doc in docs:
        out.extend(chunk_document(doc.doc_id, doc.text, size, overlap, doc.source, doc.group))
    return out


__all__ = ["CorpusChunk", "chunk_document", "chunk_corpus"]
