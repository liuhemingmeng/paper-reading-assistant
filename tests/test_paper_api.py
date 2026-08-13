from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from paper_api.api import create_app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(create_app("sqlite://")) as test_client:
        yield test_client


def paper_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "title": "Retrieval-Augmented Generation Survey",
        "authors": "Lewis, Patrick",
        "abstract": "A survey of retrieval-augmented generation systems.",
        "file_path": "data/papers/rag-survey.pdf",
    }
    payload.update(overrides)
    return payload


def create_paper(client: TestClient, **overrides: object) -> dict[str, object]:
    response = client.post("/papers", json=paper_payload(**overrides))
    assert response.status_code == 201, response.text
    return response.json()


def test_health_check(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_create_and_get_paper(client: TestClient) -> None:
    created = create_paper(client, title="  RAG Survey  ")

    assert created["id"] == 1
    assert created["title"] == "RAG Survey"
    assert created["created_at"]
    assert created["updated_at"]

    fetched = client.get("/papers/1")
    assert fetched.status_code == 200
    assert fetched.json()["authors"] == "Lewis, Patrick"


def test_list_papers_supports_pagination(client: TestClient) -> None:
    first = create_paper(client, title="First paper")
    second = create_paper(client, title="Second paper")

    response = client.get("/papers", params={"offset": 0, "limit": 1})

    assert response.status_code == 200
    assert response.json() == [second]
    assert first["id"] == 1


def test_update_paper_and_clear_optional_file_path(client: TestClient) -> None:
    created = create_paper(client)

    response = client.patch(
        f"/papers/{created['id']}",
        json={"title": "Updated RAG Survey", "file_path": None},
    )

    assert response.status_code == 200
    assert response.json()["title"] == "Updated RAG Survey"
    assert response.json()["file_path"] is None


def test_delete_paper_returns_no_content(client: TestClient) -> None:
    created = create_paper(client)

    deleted = client.delete(f"/papers/{created['id']}")

    assert deleted.status_code == 204
    assert deleted.content == b""
    assert client.get(f"/papers/{created['id']}").status_code == 404


@pytest.mark.parametrize(
    ("method", "url", "payload"),
    [
        ("get", "/papers/404", None),
        ("patch", "/papers/404", {"title": "No paper"}),
        ("delete", "/papers/404", None),
    ],
)
def test_unknown_paper_returns_not_found(
    client: TestClient,
    method: str,
    url: str,
    payload: dict[str, str] | None,
) -> None:
    response = getattr(client, method)(url, json=payload) if payload else getattr(client, method)(url)

    assert response.status_code == 404
    assert response.json()["detail"] == "Paper not found: 404"


def test_invalid_create_request_returns_validation_error(client: TestClient) -> None:
    response = client.post("/papers", json=paper_payload(title="   "))

    assert response.status_code == 422


def test_invalid_pagination_returns_validation_error(client: TestClient) -> None:
    assert client.get("/papers", params={"offset": -1}).status_code == 422
    assert client.get("/papers", params={"limit": 0}).status_code == 422
    assert client.get("/papers", params={"limit": 101}).status_code == 422
