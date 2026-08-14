from collections.abc import Generator
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from paper_api.api import create_app
from paper_api.llm_client import GroundedAnswer, ReadingInsight
from paper_api.models import PaperInsight
from paper_api.services import answer_question, build_retrieval_index, generate_insight, retrieve_chunks


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
