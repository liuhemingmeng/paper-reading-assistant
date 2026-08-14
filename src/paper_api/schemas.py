"""Pydantic schemas used at the HTTP boundary."""

from __future__ import annotations

import json
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PaperCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    authors: str = Field(min_length=1, max_length=500)
    abstract: str = Field(min_length=1, max_length=10_000)
    file_path: str | None = Field(default=None, max_length=1000)

    @field_validator("title", "authors", "abstract")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("file_path")
    @classmethod
    def strip_optional_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class PaperUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    authors: str | None = Field(default=None, min_length=1, max_length=500)
    abstract: str | None = Field(default=None, min_length=1, max_length=10_000)
    file_path: str | None = Field(default=None, max_length=1000)

    @field_validator("title", "authors", "abstract")
    @classmethod
    def strip_optional_required_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("file_path")
    @classmethod
    def strip_optional_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class PaperRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    authors: str
    abstract: str
    file_path: str | None
    created_at: datetime
    updated_at: datetime


class PaperDocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    paper_id: int
    original_filename: str
    storage_path: str
    file_size: int
    page_count: int
    status: str
    created_at: datetime


class PaperChunkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    sequence: int
    page_number: int
    section_title: str | None
    content: str
    char_count: int


class EvaluationCaseRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1_000)
    expected_page_numbers: list[int] = Field(min_length=1, max_length=10)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("expected_page_numbers")
    @classmethod
    def validate_expected_pages(cls, value: list[int]) -> list[int]:
        if any(page < 1 for page in value):
            raise ValueError("page numbers must be positive")
        return value


class EvaluationCaseResultRead(BaseModel):
    question: str
    expected_page_numbers: list[int]
    retrieved_page_numbers: list[int]
    hit: bool
    reciprocal_rank: float


class RetrievalEvaluationRead(BaseModel):
    k: int
    case_count: int
    recall_at_k: float
    mean_reciprocal_rank: float
    results: list[EvaluationCaseResultRead]


class AnswerEvaluationCaseRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1_000)
    expected_page_numbers: list[int] = Field(min_length=1, max_length=10)
    expected_answer_pages: list[int] = Field(default_factory=list, max_length=10)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("expected_page_numbers", "expected_answer_pages")
    @classmethod
    def validate_page_numbers(cls, value: list[int]) -> list[int]:
        if any(page < 1 for page in value):
            raise ValueError("page numbers must be positive")
        return value


class AnswerEvaluationCaseResultRead(BaseModel):
    question: str
    expected_page_numbers: list[int]
    answer: str
    cited_pages: list[int]
    evidence_pages: list[int]
    citation_consistent: bool
    faithful: bool | None
    faithfulness_reason: str | None


class AnswerEvaluationReportRead(BaseModel):
    k: int
    case_count: int
    recall_at_k: float
    mean_reciprocal_rank: float
    citation_correct_rate: float
    faithfulness_run: bool
    faithful_rate: float | None
    results: list[AnswerEvaluationCaseResultRead]


class RetrievalResultRead(BaseModel):
    chunk_id: int
    sequence: int
    page_number: int
    section_title: str | None
    content: str
    score: float


class RetrievalIndexRead(BaseModel):
    paper_id: int
    model: str
    indexed_chunks: int


class GroundedAnswerRead(BaseModel):
    answer: str
    model: str
    citations: list[RetrievalResultRead]


class PaperInsightRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    paper_id: int
    summary: str
    questions: list[str]
    model: str
    status: str
    error_message: str | None
    created_at: datetime

    @classmethod
    def from_record(cls, record: object) -> "PaperInsightRead":
        questions = json.loads(str(getattr(record, "questions_json")))
        return cls(
            id=int(getattr(record, "id")),
            paper_id=int(getattr(record, "paper_id")),
            summary=str(getattr(record, "summary")),
            questions=questions,
            model=str(getattr(record, "model")),
            status=str(getattr(record, "status")),
            error_message=getattr(record, "error_message"),
            created_at=getattr(record, "created_at"),
        )
