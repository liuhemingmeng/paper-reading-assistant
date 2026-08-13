"""Database setup and session lifecycle."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

DEFAULT_DATABASE_URL = "sqlite:///data/papers.db"


class Base(DeclarativeBase):
    """Base class shared by all ORM models."""


def create_engine_for_url(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    if database_url.startswith("sqlite:///"):
        database_path = database_url.removeprefix("sqlite:///")
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    if database_url == "sqlite://":
        return create_engine(database_url, connect_args=connect_args, poolclass=StaticPool)
    return create_engine(database_url, connect_args=connect_args)


def create_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def create_tables(engine) -> None:
    Base.metadata.create_all(bind=engine)


def get_session(session_factory) -> Generator[Session, None, None]:
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
