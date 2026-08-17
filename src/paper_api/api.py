"""FastAPI application factory and HTTP routes."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import (
    DEFAULT_DATABASE_URL,
    create_engine_for_url,
    create_session_factory,
    create_tables,
    get_session,
)
from .vector_store import ensure_vector_table
from .answer_evaluation import (
    AnswerEvaluationCase,
    LLMNotConfiguredForJudgingError,
    OpenAICompatibleFaithfulnessJudge,
    evaluate_answers,
)
from .auth import verify_api_key
from .embeddings import EmbeddingConfigurationError, EmbeddingResponseError, get_default_embedder
from .rerank import RerankConfigurationError, RerankResponseError, SiliconFlowReranker
from .settings import write_env_values
from .evaluation import EvaluationCase, evaluate_retrieval
from .llm_client import LLMConfigurationError, LLMResponseError, OpenAICompatibleClient
from .models import Paper, PaperChunk, PaperDocument
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
    SettingsStatus,
    SettingsUpdate,
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
    retrieve_corpus,
    answer_question_corpus,
    save_processed_document,
    update_paper,
)


def _maybe_rerank(
    query: str, results: list[tuple[PaperChunk, float]]
) -> list[tuple[PaperChunk, float]]:
    """Reorder retrieval hits with a cross-encoder reranker when enabled.

    Gated by ``RERANKER_ENABLED`` (set to 1/true/yes). When disabled or the
    reranker is misconfigured the original embedding ranking is returned
    unchanged, so the default demo path never depends on the reranker.
    """
    flag = os.getenv("RERANKER_ENABLED", "").strip().lower()
    if flag not in ("1", "true", "yes") or not results:
        return results
    try:
        model = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B").strip() or "Qwen/Qwen3-Reranker-0.6B"
        reranker = SiliconFlowReranker.from_environment(model=model)
    except (RerankConfigurationError, RerankResponseError):
        return results
    documents = [chunk.content for chunk, _ in results]
    try:
        hits = reranker.rerank(query, documents, top_n=len(results))
    except RerankResponseError:
        return results
    order = {hit.index: hit.score for hit in hits}
    return [(results[i][0], order.get(i, results[i][1])) for i in sorted(order)]


def create_app(
    database_url: str = DEFAULT_DATABASE_URL,
    upload_root: Path = UPLOAD_ROOT,
    embedder: TextEmbedder | None = None,
) -> FastAPI:
    # Deployment overrides the database backend via DATABASE_URL (e.g. a
    # pgvector instance). Tests keep passing an explicit sqlite URL, so this
    # only takes effect when the variable is actually set.
    database_url = os.getenv("DATABASE_URL", database_url)
    engine = create_engine_for_url(database_url)
    session_factory = create_session_factory(engine)
    create_tables(engine)
    if engine.dialect.name == "postgresql":
        with session_factory() as init_session:
            ensure_vector_table(init_session)

    app = FastAPI(
        title="Paper Reading Assistant API",
        version="0.7.0",
        description="Manage papers, parse PDFs, retrieve source-aware evidence with a configurable embedder, answer with citations, and evaluate retrieval.",
        dependencies=[Depends(verify_api_key)],
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

    @app.get("/corpus/search", response_model=list[RetrievalResultRead])
    def search_corpus_route(
        query: str,
        limit: int = 3,
        session: Session = Depends(get_db_session),
    ) -> list[RetrievalResultRead]:
        validate_retrieval_limit(limit)
        try:
            results = retrieve_corpus(session, query, limit=limit, embedder=app.state.embedder)
            results = _maybe_rerank(query, results)
            chunk_ids = [chunk.id for chunk, _ in results]
            paper_map = {
                chunk.id: paper
                for chunk, paper in session.execute(
                    select(PaperChunk, Paper).join(PaperDocument, PaperDocument.id == PaperChunk.document_id).join(Paper, Paper.id == PaperDocument.paper_id).where(PaperChunk.id.in_(chunk_ids))
                ).all()
            }
            return [
                RetrievalResultRead(
                    chunk_id=chunk.id,
                    sequence=chunk.sequence,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    content=chunk.content,
                    score=round(score, 6),
                    paper_id=paper_map[chunk.id].id if chunk.id in paper_map else None,
                    paper_title=paper_map[chunk.id].title if chunk.id in paper_map else None,
                )
                for chunk, score in results
            ]
        except RetrievalNotReadyError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except EmbeddingResponseError as error:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
        except (NoRelevantEvidenceError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error

    @app.post("/corpus/questions:answer", response_model=GroundedAnswerRead)
    def answer_corpus_route(
        question: str,
        limit: int = 5,
        session: Session = Depends(get_db_session),
    ) -> GroundedAnswerRead:
        validate_retrieval_limit(limit)
        try:
            answer, evidence = answer_question_corpus(
                session,
                question,
                OpenAICompatibleClient.from_environment(),
                limit=limit,
                embedder=app.state.embedder,
            )
            evidence = _maybe_rerank(question, evidence)
            chunk_ids = [chunk.id for chunk, _ in evidence]
            paper_map = {
                chunk.id: paper
                for chunk, paper in session.execute(
                    select(PaperChunk, Paper)
                    .join(PaperDocument, PaperDocument.id == PaperChunk.document_id)
                    .join(Paper, Paper.id == PaperDocument.paper_id)
                    .where(PaperChunk.id.in_(chunk_ids))
                ).all()
            }
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
                        paper_id=paper_map[chunk.id].id if chunk.id in paper_map else None,
                        paper_title=paper_map[chunk.id].title if chunk.id in paper_map else None,
                    )
                    for chunk, score in evidence
                ],
            )
        except RetrievalNotReadyError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except (NoRelevantEvidenceError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error
        except LLMConfigurationError as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
        except (LLMResponseError, EmbeddingResponseError) as error:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error

    @app.get("/settings/status", response_model=SettingsStatus)
    def settings_status_route() -> SettingsStatus:
        """Report the active provider models without exposing any secrets."""
        emb_model = app.state.embedder.model if app.state.embedder is not None else "local-hashing-v1"
        llm_configured = all(
            os.getenv(var, "").strip() for var in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
        )
        llm_model = os.getenv("LLM_MODEL", "").strip() if llm_configured else None
        return SettingsStatus(embedding_model=emb_model, llm_model=llm_model)

    @app.post("/settings", response_model=SettingsStatus)
    def update_settings_route(body: SettingsUpdate) -> SettingsStatus:
        """Persist provider keys supplied from the UI and re-wire the embedder.

        Only the fields the caller actually sends are changed (``None`` means
        "leave as-is"). Values are applied to the process environment immediately
        and, when a ``.env`` file exists next to the working directory, written
        back so they survive a restart. The retrieval embedder is rebuilt so new
        indexes/queries use the updated model without a server restart. The LLM
        client is created per request from the environment, so it picks the new
        key up automatically. ``RAG_API_KEY`` is intentionally not touched here.
        """
        updates: dict[str, str] = {}
        mapping = {
            "embedding_base_url": "EMBEDDING_BASE_URL",
            "embedding_api_key": "EMBEDDING_API_KEY",
            "embedding_model": "EMBEDDING_MODEL",
            "embedding_endpoint": "EMBEDDING_ENDPOINT",
            "llm_base_url": "LLM_BASE_URL",
            "llm_api_key": "LLM_API_KEY",
            "llm_model": "LLM_MODEL",
        }
        for field, env_key in mapping.items():
            value = getattr(body, field)
            if value is not None:
                updates[env_key] = value
        if not updates:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="no settings provided")
        for env_key, value in updates.items():
            os.environ[env_key] = value
        write_env_values(updates)
        app.state.embedder = get_default_embedder()
        return settings_status_route()

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

    @app.get("/insight", response_class=FileResponse, include_in_schema=False)
    def frontend_index() -> FileResponse:
        # When installed as a package, __file__ lives in site-packages, so the
        # frontend is located via FRONTEND_DIR (set to /app/frontend in the
        # image) and falls back to the repo layout for local development.
        frontend_dir = Path(os.getenv("FRONTEND_DIR", str(Path(__file__).resolve().parent.parent.parent / "frontend"))).resolve()
        index_path = frontend_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="frontend not bundled")

    frontend_dir = Path(os.getenv("FRONTEND_DIR", str(Path(__file__).resolve().parent.parent.parent / "frontend"))).resolve()
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

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
