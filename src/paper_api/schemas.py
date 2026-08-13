"""Pydantic schemas used at the HTTP boundary."""

from __future__ import annotations

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
