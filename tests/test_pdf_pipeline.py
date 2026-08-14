from collections.abc import Generator
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from paper_api.api import create_app
from paper_api.answer_evaluation import (
    AnswerEvaluationCase,
    FaithfulVerdict,
    LLMNotConfiguredForJudgingError,
    OpenAICompatibleFaithfulnessJudge,
    check_citation_correctness,
    evaluate_answers,
    extract_cited_pages,
)
from paper_api.evaluation import EvaluationCase, evaluate_retrieval
from paper_api.embeddings import OpenAICompatibleEmbedder, get_default_embedder
from paper_api.llm_client import GroundedAnswer, ReadingInsight
from paper_api.models import PaperInsight
from paper_api.retrieval import EmbeddedText, LocalHashingEmbedder, TextEmbedder
from paper_api.services import (
    RetrievalNotReadyError,
    answer_question,
    build_retrieval_index,
    generate_insight,
    retrieve_chunks,
)


@pytest.fixture
def client(tmp_path: Path) -> Generator[TestClient, None, None]:
    with TestClient(create_app("sqlite://", upload_root=tmp_path / "uploads")) as test_client:
        yield test_client


def create_paper(client: TestClient) -> int:
    response = client.post(
        "/papers",
        json={
            "title": "PDF Pipeline Test",
            "authors": "Test Author",
            "abstract": "Testing PDF extraction and persistence.",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def pdf_bytes(*pages: str) -> bytes:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text)
    result = document.tobytes()
    document.close()
    return result


def upload_pdf(client: TestClient, paper_id: int, content: bytes, filename: str = "sample.pdf"):
    return client.post(
        f"/papers/{paper_id}/document",
        files={"file": (filename, content, "application/pdf")},
    )


def test_upload_pdf_extracts_and_persists_chunks(client: TestClient) -> None:
    paper_id = create_paper(client)
    content = pdf_bytes("1 Introduction\nRetrieval augmented generation combines retrieval with generation.", "2 Method\nThe method retrieves evidence.")

    uploaded = upload_pdf(client, paper_id, content)

    assert uploaded.status_code == 201, uploaded.text
    document = uploaded.json()
    assert document["page_count"] == 2
    assert document["original_filename"] == "sample.pdf"
    assert Path(document["storage_path"]).exists()

    chunks = client.get(f"/papers/{paper_id}/chunks")
    assert chunks.status_code == 200
    assert len(chunks.json()) == 2
    assert chunks.json()[0]["page_number"] == 1
    assert chunks.json()[0]["section_title"] == "1 Introduction"
    assert "retrieval augmented generation" in chunks.json()[0]["content"].lower()
    assert chunks.json()[1]["page_number"] == 2


def test_reupload_replaces_previous_document_and_chunks(client: TestClient) -> None:
    paper_id = create_paper(client)
    first = upload_pdf(client, paper_id, pdf_bytes("Old text."))
    second = upload_pdf(client, paper_id, pdf_bytes("New text."), "replacement.pdf")

    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["original_filename"] == "replacement.pdf"
    assert not Path(first.json()["storage_path"]).exists()
    chunks = client.get(f"/papers/{paper_id}/chunks").json()
    assert len(chunks) == 1
    assert chunks[0]["content"] == "New text."


def test_delete_paper_removes_its_document_file(client: TestClient) -> None:
    paper_id = create_paper(client)
    uploaded = upload_pdf(client, paper_id, pdf_bytes("Text that will be deleted."))
    assert uploaded.status_code == 201
    stored_path = Path(uploaded.json()["storage_path"])
    assert stored_path.exists()

    deleted = client.delete(f"/papers/{paper_id}")

    assert deleted.status_code == 204
    assert not stored_path.exists()
    assert client.get(f"/papers/{paper_id}/document").status_code == 404


def test_rejects_non_pdf_and_empty_text_pdf(client: TestClient) -> None:
    paper_id = create_paper(client)

    non_pdf = upload_pdf(client, paper_id, b"not a PDF", "notes.txt")
    assert non_pdf.status_code == 422
    assert "Only .pdf files" in non_pdf.json()["detail"]

    empty_pdf = upload_pdf(client, paper_id, pdf_bytes(""))
    assert empty_pdf.status_code == 422
    assert "no extractable text" in empty_pdf.json()["detail"]


def test_document_routes_return_not_found_when_missing(client: TestClient) -> None:
    paper_id = create_paper(client)
    assert client.get(f"/papers/{paper_id}/document").status_code == 404
    assert client.get(f"/papers/{paper_id}/chunks").status_code == 404
    assert upload_pdf(client, 404, pdf_bytes("Text.")).status_code == 404


def test_retrieval_indexes_chunks_and_returns_source_aware_evidence(client: TestClient) -> None:
    paper_id = create_paper(client)
    assert upload_pdf(
        client,
        paper_id,
        pdf_bytes(
            "1 Retrieval\nRetrieval finds relevant evidence before generation.",
            "2 Evaluation\nEvaluation measures whether cited evidence supports an answer.",
        ),
    ).status_code == 201

    not_indexed = client.get(f"/papers/{paper_id}/search", params={"query": "retrieval evidence"})
    assert not_indexed.status_code == 409

    indexed = client.post(f"/papers/{paper_id}/retrieval:index")
    assert indexed.status_code == 200
    assert indexed.json()["indexed_chunks"] == 2
    assert indexed.json()["model"] == "local-hashing-v1"

    results = client.get(f"/papers/{paper_id}/search", params={"query": "retrieval evidence", "limit": 1})
    assert results.status_code == 200
    assert len(results.json()) == 1
    result = results.json()[0]
    assert result["page_number"] == 1
    assert result["section_title"] == "1 Retrieval"
    assert "Retrieval finds" in result["content"]
    assert result["score"] > 0


def test_answer_question_uses_only_retrieved_evidence(client: TestClient) -> None:
    paper_id = create_paper(client)
    assert upload_pdf(
        client,
        paper_id,
        pdf_bytes(
            "1 Method\nThe method retrieves evidence before generating an answer.",
            "2 Results\nThe experiment reports a separate result.",
        ),
    ).status_code == 201
    assert client.post(f"/papers/{paper_id}/retrieval:index").status_code == 200

    class FakeAnswerGenerator:
        def answer(self, question: str, evidence: str) -> GroundedAnswer:
            assert question == "What does the method retrieve?"
            assert "retrieves evidence" in evidence
            assert "separate result" not in evidence
            return GroundedAnswer(answer="It retrieves evidence.", model="fake-answer-model")

    session = client.app.state.session_factory()
    try:
        answer, citations = answer_question(
            session,
            paper_id,
            "What does the method retrieve?",
            FakeAnswerGenerator(),
            limit=1,
        )
    finally:
        session.close()

    assert answer.answer == "It retrieves evidence."
    assert len(citations) == 1
    assert citations[0][0].page_number == 1


def test_retrieval_rejects_irrelevant_or_blank_query(client: TestClient) -> None:
    paper_id = create_paper(client)
    assert upload_pdf(client, paper_id, pdf_bytes("Known terms appear here.")).status_code == 201
    assert client.post(f"/papers/{paper_id}/retrieval:index").status_code == 200

    blank = client.get(f"/papers/{paper_id}/search", params={"query": "   "})
    assert blank.status_code == 422
    unrelated = client.get(f"/papers/{paper_id}/search", params={"query": "quantum banana"})
    assert unrelated.status_code == 422


def test_retrieval_evaluation_reports_recall_and_mrr(client: TestClient) -> None:
    paper_id = create_paper(client)
    assert upload_pdf(
        client,
        paper_id,
        pdf_bytes(
            "1 Retrieval\nRetrieval finds relevant evidence before generation.",
            "2 Evaluation\nEvaluation measures ranking quality with recall and MRR.",
        ),
    ).status_code == 201
    assert client.post(f"/papers/{paper_id}/retrieval:index").status_code == 200

    response = client.post(
        f"/papers/{paper_id}/retrieval:evaluate?k=1",
        json=[
            {"question": "What finds relevant evidence?", "expected_page_numbers": [1]},
            {"question": "What measures ranking quality?", "expected_page_numbers": [2]},
        ],
    )

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["k"] == 1
    assert report["case_count"] == 2
    assert report["recall_at_k"] == 1.0
    assert report["mean_reciprocal_rank"] == 1.0
    assert [item["retrieved_page_numbers"] for item in report["results"]] == [[1], [2]]


def test_evaluation_rejects_unindexed_and_invalid_cases(client: TestClient) -> None:
    paper_id = create_paper(client)
    assert upload_pdf(client, paper_id, pdf_bytes("Known evidence appears here.")).status_code == 201

    unindexed = client.post(
        f"/papers/{paper_id}/retrieval:evaluate",
        json=[{"question": "What appears?", "expected_page_numbers": [1]}],
    )
    assert unindexed.status_code == 409

    invalid = client.post(
        f"/papers/{paper_id}/retrieval:evaluate",
        json=[{"question": "", "expected_page_numbers": [0]}],
    )
    assert invalid.status_code == 422


def test_evaluation_metric_math_uses_first_relevant_rank() -> None:
    report = evaluate_retrieval(
        [
            EvaluationCase(question="first", expected_page_numbers=[2]),
            EvaluationCase(question="second", expected_page_numbers=[4]),
            EvaluationCase(question="miss", expected_page_numbers=[9]),
        ],
        lambda question, k: {"first": [1, 2, 3], "second": [4, 1, 2], "miss": [1, 2, 3]}[question][:k],
        k=3,
    )

    assert report.recall_at_k == pytest.approx(2 / 3)
    assert report.mean_reciprocal_rank == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert report.results[0].reciprocal_rank == 0.5


def test_answer_route_requires_llm_configuration_after_indexing(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    paper_id = create_paper(client)
    assert upload_pdf(client, paper_id, pdf_bytes("The method retrieves evidence.")).status_code == 201
    assert client.post(f"/papers/{paper_id}/retrieval:index").status_code == 200
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    response = client.post(
        f"/papers/{paper_id}/questions:answer",
        params={"question": "What does the method retrieve?"},
    )
    assert response.status_code == 503


def test_generate_insight_persists_structured_output(client: TestClient) -> None:
    paper_id = create_paper(client)
    assert upload_pdf(client, paper_id, pdf_bytes("A concise paper body.")).status_code == 201

    class FakeGenerator:
        def generate(self, text: str) -> ReadingInsight:
            assert "concise paper body" in text.lower()
            return ReadingInsight(
                summary="A concise summary.",
                questions=[f"Question {index}?" for index in range(1, 6)],
                model="fake-model",
            )

    app = client.app
    session_factory = app.state.session_factory
    session = session_factory()
    try:
        generated = generate_insight(session, paper_id, FakeGenerator())
        assert generated.summary == "A concise summary."
        assert session.query(PaperInsight).count() == 1
    finally:
        session.close()

    response = client.get(f"/papers/{paper_id}/insight")
    assert response.status_code == 200
    assert response.json()["questions"] == [f"Question {index}?" for index in range(1, 6)]


def test_insight_route_requires_llm_configuration(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    paper_id = create_paper(client)
    assert upload_pdf(client, paper_id, pdf_bytes("Text for insight.")).status_code == 201
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    response = client.post(f"/papers/{paper_id}/insights:generate")

    assert response.status_code == 503
    assert "LLM_BASE_URL" in response.json()["detail"]


class LabeledFakeEmbedder:
    """Deterministic embedder used to prove the TextEmbedder protocol is pluggable."""

    def __init__(self, model: str = "fake-v1") -> None:
        self.model = model

    def embed(self, text: str) -> EmbeddedText:
        hits = 1.0 if "evidence" in text.lower() else 0.0
        return EmbeddedText(vector=[hits, 1.0 - hits], model=self.model)


def test_openai_compatible_embedder_returns_normalized_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    import paper_api.embeddings as embeddings_mod

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": [{"embedding": [3.0, 4.0]}]}

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *exc) -> bool:
            return False

        def post(self, *args, **kwargs) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(embeddings_mod.httpx, "Client", FakeClient)
    embedder = OpenAICompatibleEmbedder(base_url="https://api.example.com/v1", api_key="k", model="emb-model")
    embedded = embedder.embed("hello")

    assert embedded.model == "emb-model"
    assert embedded.vector == pytest.approx([0.6, 0.8])


def test_get_default_embedder_falls_back_to_local_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    assert isinstance(get_default_embedder(), LocalHashingEmbedder)


def test_get_default_embedder_uses_real_embedder_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.setenv("EMBEDDING_MODEL", "emb-model")

    embedder = get_default_embedder()
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder.model == "emb-model"


def test_injected_fake_embedder_indexes_and_retrieves(tmp_path: Path) -> None:
    with TestClient(
        create_app("sqlite://", upload_root=tmp_path / "uploads", embedder=LabeledFakeEmbedder("fake-v1"))
    ) as client:
        paper_id = create_paper(client)
        assert upload_pdf(
            client,
            paper_id,
            pdf_bytes("1 Evidence\nThis passage contains the evidence.", "2 Other\nUnrelated content."),
        ).status_code == 201
        assert client.post(f"/papers/{paper_id}/retrieval:index").status_code == 200

        search = client.get(f"/papers/{paper_id}/search", params={"query": "where is the evidence"})
        assert search.status_code == 200, search.text
        assert search.json()[0]["page_number"] == 1


def test_retrieval_guards_against_mixed_embedder_models(tmp_path: Path) -> None:
    with TestClient(
        create_app("sqlite://", upload_root=tmp_path / "uploads", embedder=LabeledFakeEmbedder("fake-v1"))
    ) as client:
        paper_id = create_paper(client)
        assert upload_pdf(client, paper_id, pdf_bytes("Evidence only here.")).status_code == 201

        session = client.app.state.session_factory()
        try:
            build_retrieval_index(session, paper_id, embedder=LabeledFakeEmbedder("fake-v1"))
            with pytest.raises(RetrievalNotReadyError):
                retrieve_chunks(session, paper_id, "evidence", embedder=LabeledFakeEmbedder("fake-v2"))
        finally:
            session.close()


def test_extract_cited_pages_handles_english_and_chinese() -> None:
    assert extract_cited_pages("See page 3 and p.5 for details.") == [3, 5]
    assert extract_cited_pages("依据第 7 页与第 8 页的结论。") == [7, 8]
    assert extract_cited_pages("An answer with no page reference.") == []
    assert extract_cited_pages("page 4 page 4 p.4") == [4]


def test_check_citation_correctness_flags_unsupported_pages() -> None:
    consistent = check_citation_correctness("See page 1 and page 2.", {1, 2})
    assert consistent.consistent is True
    assert consistent.cited_pages == [1, 2]

    hallucinated = check_citation_correctness("See page 9 for the result.", {1, 2})
    assert hallucinated.consistent is False
    assert hallucinated.unsupported_pages == [9]


class EchoPageAnswerGenerator:
    """Offline generator that cites the evidence pages it received."""

    def answer(self, question: str, evidence: str) -> GroundedAnswer:
        pages: list[int] = []
        for part in evidence.split("[chunk:"):
            if "page:" in part:
                try:
                    pages.append(int(part.split("page:")[1].split(";")[0]))
                except ValueError:
                    pass
        cited = " ".join(f"See page {page}." for page in sorted(set(pages)))
        return GroundedAnswer(answer=f"Answer. {cited}".strip(), model="fake-answer")


class AlwaysFaithfulJudge:
    def judge(self, question: str, evidence: str, answer: str) -> FaithfulVerdict:
        return FaithfulVerdict(faithful=True, reason="fake judge")


def test_evaluate_answers_runs_offline_with_fake_generator(client: TestClient) -> None:
    paper_id = create_paper(client)
    assert upload_pdf(
        client,
        paper_id,
        pdf_bytes(
            "1 Retrieval\nRetrieval finds relevant evidence before generation.",
            "2 Evaluation\nEvaluation measures whether cited evidence supports an answer.",
        ),
    ).status_code == 201
    assert client.post(f"/papers/{paper_id}/retrieval:index").status_code == 200

    session = client.app.state.session_factory()
    try:
        report = evaluate_answers(
            session,
            paper_id,
            [
                AnswerEvaluationCase(
                    question="What finds relevant evidence?",
                    expected_page_numbers=[1],
                    expected_answer_pages=[1],
                ),
                AnswerEvaluationCase(
                    question="What measures citation support?",
                    expected_page_numbers=[2],
                    expected_answer_pages=[2],
                ),
            ],
            EchoPageAnswerGenerator(),
            judge=None,
            k=1,
            embedder=client.app.state.embedder,
        )
    finally:
        session.close()

    assert report.case_count == 2
    assert report.recall_at_k == 1.0
    assert report.citation_correct_rate == 1.0
    assert report.faithfulness_run is False
    assert report.faithful_rate is None
    assert all(result.citation_consistent for result in report.results)
    assert report.results[0].cited_pages == [1]


def test_evaluate_answers_with_fake_judge_sets_faithful_rate(client: TestClient) -> None:
    paper_id = create_paper(client)
    assert upload_pdf(
        client,
        paper_id,
        pdf_bytes(
            "1 Retrieval\nRetrieval finds relevant evidence before generation.",
            "2 Evaluation\nEvaluation measures citation support.",
        ),
    ).status_code == 201
    assert client.post(f"/papers/{paper_id}/retrieval:index").status_code == 200

    session = client.app.state.session_factory()
    try:
        report = evaluate_answers(
            session,
            paper_id,
            [AnswerEvaluationCase(question="What finds evidence?", expected_page_numbers=[1])],
            EchoPageAnswerGenerator(),
            judge=AlwaysFaithfulJudge(),
            k=1,
            embedder=client.app.state.embedder,
        )
    finally:
        session.close()

    assert report.faithfulness_run is True
    assert report.faithful_rate == 1.0
    assert report.results[0].faithful is True


def test_answers_evaluate_route_requires_llm_configuration(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    paper_id = create_paper(client)
    assert upload_pdf(client, paper_id, pdf_bytes("The method retrieves evidence.")).status_code == 201
    assert client.post(f"/papers/{paper_id}/retrieval:index").status_code == 200
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    response = client.post(
        f"/papers/{paper_id}/answers:evaluate",
        json=[{"question": "What retrieves evidence?", "expected_page_numbers": [1]}],
    )
    assert response.status_code == 503


def test_answers_evaluate_route_rejects_unindexed_and_invalid_cases(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper_id = create_paper(client)
    assert upload_pdf(client, paper_id, pdf_bytes("Known evidence here.")).status_code == 201

    monkeypatch.setattr(
        "paper_api.llm_client.OpenAICompatibleClient.from_environment",
        lambda: EchoPageAnswerGenerator(),
    )

    unindexed = client.post(
        f"/papers/{paper_id}/answers:evaluate",
        json=[{"question": "What appears?", "expected_page_numbers": [1]}],
    )
    assert unindexed.status_code == 409

    invalid = client.post(
        f"/papers/{paper_id}/answers:evaluate",
        json=[{"question": "", "expected_page_numbers": [0]}],
    )
    assert invalid.status_code == 422


def test_faithfulness_judge_from_environment_requires_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with pytest.raises(LLMNotConfiguredForJudgingError):
        OpenAICompatibleFaithfulnessJudge.from_environment()
