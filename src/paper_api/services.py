"""Paper CRUD business logic."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Paper
from .schemas import PaperCreate, PaperUpdate


class PaperNotFoundError(Exception):
    """Raised when a requested paper does not exist."""


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
    session.delete(paper)
    session.commit()
