"""把收集到的评测语料变成可检索的纯文本文档。

基准需要文档的真实正文来构造索引条目和查询描述。这里统一处理三种来源：
PDF 用 PyMuPDF 抽文字、HTML 用标准库 ``html.parser`` 剥标签、txt/md 直接读。
抽取结果会缓存到 ``data/corpus_text_cache.json``，重跑时跳过重复的解析。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from html.parser import HTMLParser

CORPUS_MANIFEST = "data/corpus/corpus_manifest.json"
TEXT_CACHE = "data/corpus_text_cache.json"

# 与 retrieval.py 一致：英文/数字词整体，中文整段（不拆字）。
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


@dataclass
class CorpusDoc:
    doc_id: str
    title: str
    text: str
    source: str
    group: str


class _HTMLTextExtractor(HTMLParser):
    """只收集标签之间的可见文字，丢弃标签与脚本。"""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._parts.append(data.strip())

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(html: str) -> str:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
        text = parser.get_text()
    except Exception:  # noqa: BLE001 - 解析失败退回朴素去标签
        text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def _extract_pdf(path: str) -> str:
    import pymupdf  # 延迟导入，离线路径不必依赖它

    doc = pymupdf.open(path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return handle.read()


def extract_document_text(path: str) -> str:
    """按扩展名分发到对应抽取器，未知类型按纯文本兜底。"""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext == "pdf":
        return _extract_pdf(path)
    if ext in ("html", "htm"):
        return _strip_html(_read_text(path))
    # txt / md / 其它一律当纯文本读
    return _read_text(path)


def load_corpus(manifest_path: str = CORPUS_MANIFEST, use_cache: bool = True) -> list[CorpusDoc]:
    """读取聚合 manifest，返回已下载且抽取到正文的文档列表。"""
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    cache: dict[str, str] = {}
    if use_cache and os.path.exists(TEXT_CACHE):
        with open(TEXT_CACHE, "r", encoding="utf-8") as handle:
            cache = json.load(handle)

    docs: list[CorpusDoc] = []
    updated = False
    for entry in manifest.get("documents", []):
        if entry.get("status") != "downloaded":
            continue
        local_path = entry.get("local_path")
        if not local_path or not os.path.exists(local_path):
            continue
        doc_id = entry["id"]
        if use_cache and doc_id in cache:
            text = cache[doc_id]
        else:
            text = extract_document_text(local_path)
            cache[doc_id] = text
            updated = True
        text = (text or "").strip()
        # 过短的正文对检索没有意义，直接跳过
        if len(text) < 80:
            continue
        docs.append(
            CorpusDoc(
                doc_id=doc_id,
                title=(entry.get("title") or "").strip(),
                text=text,
                source=entry.get("source", ""),
                group=entry.get("group", ""),
            )
        )

    if use_cache and updated:
        with open(TEXT_CACHE, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=False)

    return docs


__all__ = ["CorpusDoc", "extract_document_text", "load_corpus", "TOKEN_PATTERN"]
