"""Paper CRUD and document processing business logic."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .llm_client import InsightGenerator, ReadingInsight
from .models import Paper, PaperChunk, PaperDocument, PaperInsight
from .pdf_processing import ExtractedPage, TextChunk
from .schemas import PaperCreate, PaperUpdate


class PaperNotFoundError(Exception):
    """Raised when a requested paper does not exist."""


class DocumentNotFoundError(Exception):
    """Raised when a paper has no successfully processed PDF document."""


class InsightNotFoundError(Exception):
    """Raised when a paper has no generated reading insight."""


def create_paper(session: Session, data: PaperCreate) -> Paper:
    paper = Paper(**data.model_dump())
    session.add(paper)
    session.commit()
    session.refresh(paper)
    return paper


def list_papers(session: Session, offset: int, limit: int) -> list[Paper]:
    statement = select(Paper).order_by(Paper.id.desc()).offset(offset).limit(limit)
    return list(session.scalars(statement))


def get_paper(session: Session, paper_id: int) -> Paper:
    paper = session.get(Paper, paper_id)
    if paper is None:
        raise PaperNotFoundError(f"Paper not found: {paper_id}")
    return paper


def update_paper(session: Session, paper_id: int, data: PaperUpdate) -> Paper:
    paper = get_paper(session, paper_id)
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(paper, field, value)
    session.commit()
    session.refresh(paper)
    return paper


def delete_paper(session: Session, paper_id: int) -> None:
    paper = get_paper(session, paper_id)
    if paper.document is not None:
        Path(paper.document.storage_path).unlink(missing_ok=True)
    session.delete(paper)
    session.commit()


def save_processed_document(
    session: Session,
    paper_id: int,
    original_filename: str,
    storage_path: Path,
    file_size: int,
    pages: list[ExtractedPage],
    chunks: list[TextChunk],
) -> PaperDocument:
    paper = get_paper(session, paper_id)
    if paper.document is not None:
        previous_path = Path(paper.document.storage_path)
        session.delete(paper.document)
        session.flush()
        previous_path.unlink(missing_ok=True)

    document = PaperDocument(
        paper_id=paper.id,
        original_filename=original_filename,
        storage_path=str(storage_path),
        file_size=file_size,
        page_count=len(pages),
        extracted_text="\n\n".join(page.text for page in pages),
    )
    session.add(document)
    session.flush()
    session.add_all(
        [
            PaperChunk(
                document_id=document.id,
                sequence=chunk.sequence,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                content=chunk.content,
                char_count=len(chunk.content),
            )
            for chunk in chunks
        ]
    )
    paper.file_path = str(storage_path)
    session.commit()
    session.refresh(document)
    return document


def get_document(session: Session, paper_id: int) -> PaperDocument:
    paper = get_paper(session, paper_id)
    if paper.document is None:
        raise DocumentNotFoundError(f"Paper {paper_id} has no processed document")
    return paper.document


def list_chunks(session: Session, paper_id: int, offset: int, limit: int) -> list[PaperChunk]:
    document = get_document(session, paper_id)
    statement = (
        select(PaperChunk)
        .where(PaperChunk.document_id == document.id)
        .order_by(PaperChunk.sequence)
        .offset(offset)
        .limit(limit)
    )
    return list(session.scalars(statement))


def generate_insight(session: Session, paper_id: int, generator: InsightGenerator) -> PaperInsight:
    document = get_document(session, paper_id)
    insight: ReadingInsight = generator.generate(document.extracted_text)
    record = PaperInsight(
        paper_id=paper_id,
        summary=insight.summary,
        questions_json=json.dumps(insight.questions, ensure_ascii=False),
        model=insight.model,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def get_latest_insight(session: Session, paper_id: int) -> PaperInsight:
    get_paper(session, paper_id)
    statement = select(PaperInsight).where(PaperInsight.paper_id == paper_id).order_by(PaperInsight.id.desc())
    insight = session.scalars(statement).first()
    if insight is None:
        raise InsightNotFoundError(f"Paper {paper_id} has no generated insight")
    return insight
