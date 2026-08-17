"""Optional PostgreSQL/pgvector storage for corpus-wide ANN retrieval.

The application keeps SQLite + JSON vectors as an offline development fallback.
When the SQLAlchemy session uses PostgreSQL, this module stores vectors in a
small pgvector table and queries it with cosine distance (``<=>``).
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rag_chunk_vectors (
    chunk_id INTEGER PRIMARY KEY REFERENCES paper_chunks(id) ON DELETE CASCADE,
    model VARCHAR(200) NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding vector NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def is_postgresql(session: Session) -> bool:
    bind = session.get_bind()
    return bool(bind is not None and bind.dialect.name == "postgresql")


def ensure_vector_table(session: Session) -> None:
    """Create the pgvector table and extension on a PostgreSQL database."""
    if not is_postgresql(session):
        return
    session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    session.execute(text(TABLE_SQL))
    session.commit()


def replace_vectors(session: Session, rows: list[dict[str, Any]]) -> None:
    """Upsert chunk vectors into pgvector; no-op for non-PostgreSQL sessions."""
    if not is_postgresql(session):
        return
    ensure_vector_table(session)
    for row in rows:
        vector_literal = json.dumps(row["vector"], separators=(",", ":"))
        session.execute(
            text(
                """
                INSERT INTO rag_chunk_vectors(chunk_id, model, dimensions, embedding)
                VALUES (:chunk_id, :model, :dimensions, CAST(:embedding AS vector))
                ON CONFLICT (chunk_id) DO UPDATE SET
                    model = EXCLUDED.model,
                    dimensions = EXCLUDED.dimensions,
                    embedding = EXCLUDED.embedding,
                    created_at = NOW()
                """
            ),
            {
                "chunk_id": row["chunk_id"],
                "model": row["model"],
                "dimensions": len(row["vector"]),
                "embedding": vector_literal,
            },
        )
    session.commit()


def search_vectors(
    session: Session,
    query_vector: list[float],
    model: str,
    limit: int,
) -> list[tuple[int, float]]:
    """Return ``(chunk_id, cosine_similarity)`` ordered by pgvector ANN."""
    if not is_postgresql(session):
        return []
    vector_literal = json.dumps(query_vector, separators=(",", ":"))
    rows = session.execute(
        text(
            """
            SELECT chunk_id, 1 - (embedding <=> CAST(:embedding AS vector)) AS score
            FROM rag_chunk_vectors
            WHERE model = :model
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
            """
        ),
        {"embedding": vector_literal, "model": model, "limit": limit},
    ).all()
    return [(int(row[0]), float(row[1])) for row in rows]


def delete_vectors(session: Session, chunk_ids: list[int]) -> None:
    if not is_postgresql(session) or not chunk_ids:
        return
    session.execute(text("DELETE FROM rag_chunk_vectors WHERE chunk_id = ANY(:chunk_ids)"), {"chunk_ids": chunk_ids})
    session.commit()


__all__ = ["delete_vectors", "ensure_vector_table", "is_postgresql", "replace_vectors", "search_vectors"]
