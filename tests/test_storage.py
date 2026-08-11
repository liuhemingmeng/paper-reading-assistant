from pathlib import Path

import pytest

from todo_cli.models import Todo
from todo_cli.storage import TodoStorageError, load_todos, save_todos


def test_load_missing_file_returns_empty_list(tmp_path: Path) -> None:
    assert load_todos(tmp_path / "missing.json") == []


def test_save_and_load_todos_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "todos.json"
    todos = [Todo(id=1, title="Read a paper")]

    save_todos(path, todos)
    loaded = load_todos(path)

    assert loaded[0].id == 1
    assert loaded[0].title == "Read a paper"
    assert loaded[0].completed is False
    assert loaded[0].created_at == todos[0].created_at


def test_load_invalid_json_raises_storage_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(TodoStorageError):
        load_todos(path)
