"""FastAPI application factory and HTTP routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile, status
from sqlalchemy.orm import Session

from .database import (
    DEFAULT_DATABASE_URL,
    create_engine_for_url,
    create_session_factory,
    create_tables,
    get_session,
)
from .answer_evaluation import (
    AnswerEvaluationCase,
    LLMNotConfiguredForJudgingError,
    OpenAICompatibleFaithfulnessJudge,
    evaluate_answers,
)
from .embeddings import EmbeddingConfigurationError, EmbeddingResponseError, get_default_embedder
from .evaluation import EvaluationCase, evaluate_retrieval
from .llm_client import LLMConfigurationError, LLMResponseError, OpenAICompatibleClient
from .pdf_processing import PDFProcessingError, UPLOAD_ROOT, chunk_pages, extract_pdf_pages, save_upload
from .retrieval import TextEmbedder
from .schemas import (
    AnswerEvaluationCaseRequest,
    AnswerEvaluationReportRead,
    EvaluationCaseRequest,
    EvaluationCaseResultRead,
    GroundedAnswerRead,
    PaperChunkRead,
    PaperCreate,
    PaperDocumentRead,
    PaperInsightRead,
    PaperRead,
    PaperUpdate,
    RetrievalEvaluationRead,
    RetrievalIndexRead,
    RetrievalResultRead,
)
from .services import (
    DocumentNotFoundError,
    InsightNotFoundError,
    NoRelevantEvidenceError,
    PaperNotFoundError,
    RetrievalNotReadyError,
    answer_question,
    build_retrieval_index,
    create_paper,
    delete_paper,
    generate_insight,
    get_document,
    get_latest_insight,
    get_paper,
    list_chunks,
    list_papers,
    retrieve_chunks,
    save_processed_document,
    update_paper,
)


def create_app(
    database_url: str = DEFAULT_DATABASE_URL,
    upload_root: Path = UPLOAD_ROOT,
    embedder: TextEmbedder | None = None,
) -> FastAPI:
    engine = create_engine_for_url(database_url)
    session_factory = create_session_factory(engine)
    create_tables(engine)

    app = FastAPI(
        title="Paper Reading Assistant API",
        version="0.6.0",
        description="Manage papers, parse PDFs, retrieve source-aware evidence with a configurable embedder, answer with citations, and evaluate retrieval.",
    )
    app.state.session_factory = session_factory
    app.state.embedder = embedder or get_default_embedder()

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
        validate_pagination(offset, limit)
        return list_papers(session, offset, limit)

    @app.get("/papers/{paper_id}", response_model=PaperRead)
    def get_paper_route(paper_id: int, session: Session = Depends(get_db_session)) -> PaperRead:
        try:
            return get_paper(session, paper_id)
        except PaperNotFoundError as error:
            raise not_found(error) from error

    @app.patch("/papers/{paper_id}", response_model=PaperRead)
    def update_paper_route(
        paper_id: int,
        data: PaperUpdate,
        session: Session = Depends(get_db_session),
    ) -> PaperRead:
        try:
            return update_paper(session, paper_id, data)
        except PaperNotFoundError as error:
            raise not_found(error) from error

    @app.delete("/papers/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_paper_route(paper_id: int, session: Session = Depends(get_db_session)) -> Response:
        try:
            delete_paper(session, paper_id)
        except PaperNotFoundError as error:
            raise not_found(error) from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/papers/{paper_id}/document", response_model=PaperDocumentRead, status_code=status.HTTP_201_CREATED)
    async def upload_document_route(
        paper_id: int,
        file: UploadFile = File(...),
        session: Session = Depends(get_db_session),
    ) -> PaperDocumentRead:
        try:
            get_paper(session, paper_id)
            storage_path, file_size = await save_upload(file, paper_id, root=upload_root)
            try:
                pages = extract_pdf_pages(storage_path)
                chunks = chunk_pages(pages)
                return save_processed_document(
                    session=session,
                    paper_id=paper_id,
                    original_filename=file.filename or "uploaded.pdf",
                    storage_path=storage_path,
                    file_size=file_size,
                    pages=pages,
                    chunks=chunks,
                )
            except PDFProcessingError:
                storage_path.unlink(missing_ok=True)
                raise
        except PaperNotFoundError as error:
            raise not_found(error) from error
        except PDFProcessingError as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error

    @app.get("/papers/{paper_id}/document", response_model=PaperDocumentRead)
    def get_document_route(paper_id: int, session: Session = Depends(get_db_session)) -> PaperDocumentRead:
        try:
            return get_document(session, paper_id)
        except (PaperNotFoundError, DocumentNotFoundError) as error:
            raise not_found(error) from error

    @app.get("/papers/{paper_id}/chunks", response_model=list[PaperChunkRead])
    def list_chunks_route(
        paper_id: int,
        offset: int = 0,
        limit: int = 100,
        session: Session = Depends(get_db_session),
    ) -> list[PaperChunkRead]:
        validate_pagination(offset, limit)
        try:
            return list_chunks(session, paper_id, offset, limit)
        except (PaperNotFoundError, DocumentNotFoundError) as error:
            raise not_found(error) from error

    @app.post("/papers/{paper_id}/retrieval:index", response_model=RetrievalIndexRead)
    def index_document_route(paper_id: int, session: Session = Depends(get_db_session)) -> RetrievalIndexRead:
        try:
            model, indexed_chunks = build_retrieval_index(session, paper_id, embedder=app.state.embedder)
            return RetrievalIndexRead(paper_id=paper_id, model=model, indexed_chunks=indexed_chunks)
        except (PaperNotFoundError, DocumentNotFoundError) as error:
            raise not_found(error) from error
        except EmbeddingResponseError as error:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error

    @app.get("/papers/{paper_id}/search", response_model=list[RetrievalResultRead])
    def search_paper_route(
        paper_id: int,
        query: str,
        limit: int = 3,
        session: Session = Depends(get_db_session),
    ) -> list[RetrievalResultRead]:
        validate_retrieval_limit(limit)
        try:
            return [
                RetrievalResultRead(
                    chunk_id=chunk.id,
                    sequence=chunk.sequence,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    content=chunk.content,
                    score=round(score, 6),
                )
                for chunk, score in retrieve_chunks(session, paper_id, query, limit=limit, embedder=app.state.embedder)
            ]
        except (PaperNotFoundError, DocumentNotFoundError) as error:
            raise not_found(error) from error
        except RetrievalNotReadyError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except EmbeddingResponseError as error:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
        except (NoRelevantEvidenceError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error

    @app.post("/papers/{paper_id}/retrieval:evaluate", response_model=RetrievalEvaluationRead)
    def evaluate_retrieval_route(
        paper_id: int,
        cases: list[EvaluationCaseRequest],
        k: int = 3,
        session: Session = Depends(get_db_session),
    ) -> RetrievalEvaluationRead:
        validate_retrieval_limit(k)
        try:
            report = evaluate_retrieval(
                [
                    EvaluationCase(question=case.question, expected_page_numbers=case.expected_page_numbers)
                    for case in cases
                ],
                lambda question, limit: retrieve_page_numbers_for_evaluation(
                    session, paper_id, question, limit, app.state.embedder
                ),
                k,
            )
            return RetrievalEvaluationRead(
                k=report.k,
                case_count=report.case_count,
                recall_at_k=round(report.recall_at_k, 6),
                mean_reciprocal_rank=round(report.mean_reciprocal_rank, 6),
                results=[
                    EvaluationCaseResultRead(
                        question=result.question,
                        expected_page_numbers=result.expected_page_numbers,
                        retrieved_page_numbers=result.retrieved_page_numbers,
                        hit=result.hit,
                        reciprocal_rank=result.reciprocal_rank,
                    )
                    for result in report.results
                ],
            )
        except (PaperNotFoundError, DocumentNotFoundError) as error:
            raise not_found(error) from error
        except RetrievalNotReadyError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except EmbeddingResponseError as error:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
        except (NoRelevantEvidenceError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error

    @app.post("/papers/{paper_id}/answers:evaluate", response_model=AnswerEvaluationReportRead)
    def evaluate_answers_route(
        paper_id: int,
        cases: list[AnswerEvaluationCaseRequest],
        run_faithfulness: bool = False,
        k: int = 3,
        session: Session = Depends(get_db_session),
    ) -> AnswerEvaluationReportRead:
        validate_retrieval_limit(k)
        try:
            generator = OpenAICompatibleClient.from_environment()
            judge = OpenAICompatibleFaithfulnessJudge.from_environment() if run_faithfulness else None
            report = evaluate_answers(
                session,
                paper_id,
                [
                    AnswerEvaluationCase(
                        question=case.question,
                        expected_page_numbers=case.expected_page_numbers,
                        expected_answer_pages=case.expected_answer_pages,
                    )
                    for case in cases
                ],
                generator,
                judge=judge,
                k=k,
                embedder=app.state.embedder,
            )
            return AnswerEvaluationReportRead(
                k=report.k,
                case_count=report.case_count,
                recall_at_k=round(report.recall_at_k, 6),
                mean_reciprocal_rank=round(report.mean_reciprocal_rank, 6),
                citation_correct_rate=round(report.citation_correct_rate, 6),
                faithfulness_run=report.faithfulness_run,
                faithful_rate=round(report.faithful_rate, 6) if report.faithful_rate is not None else None,
                results=[
                    AnswerEvaluationCaseResultRead(
                        question=result.question,
                        expected_page_numbers=result.expected_page_numbers,
                        answer=result.answer,
                        cited_pages=result.cited_pages,
                        evidence_pages=result.evidence_pages,
                        citation_consistent=result.citation_consistent,
                        faithful=result.faithful,
                        faithfulness_reason=result.faithfulness_reason,
                    )
                    for result in report.results
                ],
            )
        except (PaperNotFoundError, DocumentNotFoundError) as error:
            raise not_found(error) from error
        except RetrievalNotReadyError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except (NoRelevantEvidenceError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error
        except LLMConfigurationError as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
        except LLMNotConfiguredForJudgingError as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
        except (LLMResponseError, EmbeddingResponseError) as error:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error

    @app.post("/papers/{paper_id}/questions:answer", response_model=GroundedAnswerRead)
    def answer_question_route(
        paper_id: int,
        question: str,
        limit: int = 3,
        session: Session = Depends(get_db_session),
    ) -> GroundedAnswerRead:
        validate_retrieval_limit(limit)
        try:
            answer, evidence = answer_question(
                session,
                paper_id,
                question,
                OpenAICompatibleClient.from_environment(),
                limit=limit,
                embedder=app.state.embedder,
            )
            return GroundedAnswerRead(
                answer=answer.answer,
                model=answer.model,
                citations=[
                    RetrievalResultRead(
                        chunk_id=chunk.id,
                        sequence=chunk.sequence,
                        page_number=chunk.page_number,
                        section_title=chunk.section_title,
                        content=chunk.content,
                        score=round(score, 6),
                    )
                    for chunk, score in evidence
                ],
            )
        except (PaperNotFoundError, DocumentNotFoundError) as error:
            raise not_found(error) from error
        except RetrievalNotReadyError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except (NoRelevantEvidenceError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error
        except LLMConfigurationError as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
        except (LLMResponseError, EmbeddingResponseError) as error:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error

    @app.post("/papers/{paper_id}/insights:generate", response_model=PaperInsightRead, status_code=status.HTTP_201_CREATED)
    def generate_insight_route(paper_id: int, session: Session = Depends(get_db_session)) -> PaperInsightRead:
        try:
            record = generate_insight(session, paper_id, OpenAICompatibleClient.from_environment())
            return PaperInsightRead.from_record(record)
        except (PaperNotFoundError, DocumentNotFoundError) as error:
            raise not_found(error) from error
        except LLMConfigurationError as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
        except LLMResponseError as error:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error

    @app.get("/papers/{paper_id}/insight", response_model=PaperInsightRead)
    def get_insight_route(paper_id: int, session: Session = Depends(get_db_session)) -> PaperInsightRead:
        try:
            return PaperInsightRead.from_record(get_latest_insight(session, paper_id))
        except (PaperNotFoundError, InsightNotFoundError) as error:
            raise not_found(error) from error

    return app


def validate_pagination(offset: int, limit: int) -> None:
    if offset < 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="offset must be >= 0")
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="limit must be between 1 and 100")


def retrieve_page_numbers_for_evaluation(
    session: Session, paper_id: int, question: str, limit: int, embedder: TextEmbedder
) -> list[int]:
    try:
        return [chunk.page_number for chunk, _ in retrieve_chunks(session, paper_id, question, limit=limit, embedder=embedder)]
    except NoRelevantEvidenceError:
        return []


def validate_retrieval_limit(limit: int) -> None:
    if not 1 <= limit <= 10:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="limit must be between 1 and 10")


def not_found(error: Exception) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))


app = create_app()
