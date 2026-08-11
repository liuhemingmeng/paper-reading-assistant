"""JSON persistence for todo items."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import Todo

LOGGER = logging.getLogger(__name__)


class TodoStorageError(Exception):
    """Raised when the todo data file cannot be read or written."""


def load_todos(path: Path) -> list[Todo]:
    if not path.exists():
        return []

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_data, list):
            raise ValueError("root JSON value must be a list")
        return [Todo.from_dict(item) for item in raw_data]
    except (OSError, json.JSONDecodeError, TypeError, KeyError, ValueError) as error:
        LOGGER.exception("Could not load todo data from %s", path)
        raise TodoStorageError(f"Invalid todo data: {path}") from error


def save_todos(path: Path, todos: list[Todo]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [todo.to_dict() for todo in todos]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as error:
        LOGGER.exception("Could not save todo data to %s", path)
        raise TodoStorageError(f"Could not write todo data: {path}") from error
