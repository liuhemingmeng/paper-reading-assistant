"""FastAPI application factory and HTTP routes."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Response, status
from sqlalchemy.orm import Session

from .database import (
    DEFAULT_DATABASE_URL,
    create_engine_for_url,
    create_session_factory,
    create_tables,
    get_session,
)
from .schemas import PaperCreate, PaperRead, PaperUpdate
from .services import PaperNotFoundError, create_paper, delete_paper, get_paper, list_papers, update_paper


def create_app(database_url: str = DEFAULT_DATABASE_URL) -> FastAPI:
    engine = create_engine_for_url(database_url)
    session_factory = create_session_factory(engine)
    create_tables(engine)

    app = FastAPI(
        title="Paper Reading Assistant API",
        version="0.2.0",
        description="Manage paper metadata before PDF parsing and RAG are added.",
    )

    def get_db_session():
        yield from get_session(session_factory)

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/papers", response_model=PaperRead, status_code=status.HTTP_201_CREATED)
    def create_paper_route(data: PaperCreate, session: Session = Depends(get_db_session)) -> PaperRead:
        return create_paper(session, data)

    @app.get("/papers", response_model=list[PaperRead])
    def list_papers_route(
        offset: int = 0,
        limit: int = 20,
        session: Session = Depends(get_db_session),
    ) -> list[PaperRead]:
        if offset < 0:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="offset must be >= 0")
        if not 1 <= limit <= 100:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="limit must be between 1 and 100")
        return list_papers(session, offset, limit)

    @app.get("/papers/{paper_id}", response_model=PaperRead)
    def get_paper_route(paper_id: int, session: Session = Depends(get_db_session)) -> PaperRead:
        try:
            return get_paper(session, paper_id)
        except PaperNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    @app.patch("/papers/{paper_id}", response_model=PaperRead)
    def update_paper_route(
        paper_id: int,
        data: PaperUpdate,
        session: Session = Depends(get_db_session),
    ) -> PaperRead:
        try:
            return update_paper(session, paper_id, data)
        except PaperNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    @app.delete("/papers/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_paper_route(paper_id: int, session: Session = Depends(get_db_session)) -> Response:
        try:
            delete_paper(session, paper_id)
        except PaperNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


app = create_app()
