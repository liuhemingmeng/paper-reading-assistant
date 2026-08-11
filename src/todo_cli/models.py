"""Data models for the todo CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class Todo:
    id: int
    title: str
    completed: bool = False
    created_at: str = ""

    def __post_init__(self) -> None:
        self.title = self.title.strip()
        if not self.title:
            raise ValueError("Todo title cannot be empty")
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Todo":
        return cls(
            id=int(data["id"]),
            title=str(data["title"]),
            completed=bool(data.get("completed", False)),
            created_at=str(data.get("created_at", "")),
        )
