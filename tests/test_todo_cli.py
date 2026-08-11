from argparse import Namespace
from pathlib import Path

import pytest

from todo_cli.cli import run_command
from todo_cli.storage import TodoStorageError


def args(data_file: Path, command: str, **values: object) -> Namespace:
    return Namespace(data_file=data_file, command=command, **values)


def test_todo_lifecycle(tmp_path: Path) -> None:
    data_file = tmp_path / "todos.json"

    assert run_command(args(data_file, "add", title="Read RAG paper")) == "Added todo 1: Read RAG paper"
    assert run_command(args(data_file, "list")) == "  1. [ ] Read RAG paper"
    assert run_command(args(data_file, "done", id=1)) == "Completed todo 1: Read RAG paper"
    assert run_command(args(data_file, "list")) == "  1. [x] Read RAG paper"
    assert run_command(args(data_file, "delete", id=1)) == "Deleted todo 1: Read RAG paper"
    assert run_command(args(data_file, "list")) == "No todos yet."


def test_empty_title_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        run_command(args(tmp_path / "todos.json", "add", title="   "))


def test_unknown_todo_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Todo not found: 99"):
        run_command(args(tmp_path / "todos.json", "done", id=99))


def test_corrupt_data_is_reported(tmp_path: Path) -> None:
    data_file = tmp_path / "todos.json"
    data_file.write_text("{bad", encoding="utf-8")

    with pytest.raises(TodoStorageError):
        run_command(args(data_file, "list"))
