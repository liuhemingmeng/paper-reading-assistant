"""PDF storage, text extraction, and source-aware chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pymupdf
from fastapi import UploadFile

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UPLOAD_ROOT = Path("data/uploads")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 160


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be safely stored or parsed."""


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    sequence: int
    page_number: int
    section_title: str | None
    content: str


def validate_pdf_upload(upload: UploadFile) -> None:
    filename = upload.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise PDFProcessingError("Only .pdf files are accepted")
    if upload.content_type and upload.content_type not in {"application/pdf", "application/octet-stream"}:
        raise PDFProcessingError("Upload content type must be application/pdf")


async def save_upload(upload: UploadFile, paper_id: int, root: Path = UPLOAD_ROOT) -> tuple[Path, int]:
    validate_pdf_upload(upload)
    target_dir = root / f"paper-{paper_id}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{uuid4().hex}.pdf"
    size = 0

    try:
        with target_path.open("wb") as destination:
            while content := await upload.read(1024 * 1024):
                size += len(content)
                if size > MAX_UPLOAD_BYTES:
                    raise PDFProcessingError("PDF exceeds the 10 MiB upload limit")
                destination.write(content)
    except Exception:
        target_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    return target_path, size


def normalize_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_pdf_pages(path: Path) -> list[ExtractedPage]:
    try:
        with pymupdf.open(path) as document:
            pages = [
                ExtractedPage(page_number=index, text=normalize_text(page.get_text("text")))
                for index, page in enumerate(document, start=1)
            ]
    except (pymupdf.FileDataError, RuntimeError, OSError) as error:
        raise PDFProcessingError("The uploaded file is not a readable PDF") from error

    if not pages or not any(page.text for page in pages):
        raise PDFProcessingError("The PDF contains no extractable text; OCR is not supported yet")
    return pages


def looks_like_heading(text: str) -> bool:
    return bool(re.match(r"^(\d+(?:\.\d+)*[.、]?\s+|[A-Z][A-Z\s]{4,}$)", text)) and len(text) < 180


def split_text(text: str, max_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= max_size:
        return [text]

    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_size, len(text))
        if end < len(text):
            boundary = max(text.rfind("。", start, end), text.rfind(". ", start, end), text.rfind("\n", start, end))
            if boundary > start + max_size // 2:
                end = boundary + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [part for part in parts if part]


def chunk_pages(pages: list[ExtractedPage]) -> list[TextChunk]:
    """Split page text into paragraph-level chunks with source metadata.

    PyMuPDF's ``get_text`` returns one text line per row, so real papers would
    otherwise be cut into one chunk per line (sentence fragments). Instead we
    join consecutive non-blank lines into paragraphs (blank line = paragraph
    boundary, heading line = new section) and only then split long paragraphs
    with :func:`split_text`, which keeps sentence boundaries intact.
    """
    chunks: list[TextChunk] = []
    section_title: str | None = None
    sequence = 1

    def emit(page_number: int, buffer: list[str]) -> None:
        nonlocal sequence
        if not buffer:
            return
        paragraph = " ".join(buffer)
        for content in split_text(paragraph):
            chunks.append(
                TextChunk(
                    sequence=sequence,
                    page_number=page_number,
                    section_title=section_title,
                    content=content,
                )
            )
            sequence += 1

    for page in pages:
        buffer: list[str] = []
        for raw_line in page.text.split("\n"):
            line = raw_line.strip()
            if not line:
                emit(page.page_number, buffer)
                buffer = []
                continue
            if looks_like_heading(line):
                emit(page.page_number, buffer)
                buffer = []
                section_title = line
                continue
            buffer.append(line)
        emit(page.page_number, buffer)
    if not chunks:
        raise PDFProcessingError("The PDF has text but no usable paragraphs")
    return chunks
