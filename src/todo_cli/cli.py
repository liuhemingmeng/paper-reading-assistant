"""Command-line interface and todo operations."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .models import Todo
from .storage import TodoStorageError, load_todos, save_todos

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_PATH = Path("data/todos.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage a local JSON todo list.")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to JSON data file (default: {DEFAULT_DATA_PATH}).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add a new todo.")
    add_parser.add_argument("title", help="Todo title.")

    subparsers.add_parser("list", help="List all todos.")

    done_parser = subparsers.add_parser("done", help="Mark a todo as completed.")
    done_parser.add_argument("id", type=int, help="Todo ID.")

    delete_parser = subparsers.add_parser("delete", help="Delete a todo.")
    delete_parser.add_argument("id", type=int, help="Todo ID.")
    return parser


def next_id(todos: list[Todo]) -> int:
    return max((todo.id for todo in todos), default=0) + 1


def find_todo(todos: list[Todo], todo_id: int) -> Todo:
    for todo in todos:
        if todo.id == todo_id:
            return todo
    raise ValueError(f"Todo not found: {todo_id}")


def format_todo(todo: Todo) -> str:
    marker = "x" if todo.completed else " "
    return f"{todo.id:>3}. [{marker}] {todo.title}"


def run_command(args: argparse.Namespace) -> str:
    todos = load_todos(args.data_file)

    if args.command == "add":
        todo = Todo(id=next_id(todos), title=args.title)
        todos.append(todo)
        save_todos(args.data_file, todos)
        return f"Added todo {todo.id}: {todo.title}"

    if args.command == "list":
        if not todos:
            return "No todos yet."
        return "\n".join(format_todo(todo) for todo in todos)

    if args.command == "done":
        todo = find_todo(todos, args.id)
        todo.completed = True
        save_todos(args.data_file, todos)
        return f"Completed todo {todo.id}: {todo.title}"

    if args.command == "delete":
        todo = find_todo(todos, args.id)
        todos.remove(todo)
        save_todos(args.data_file, todos)
        return f"Deleted todo {todo.id}: {todo.title}"

    raise ValueError(f"Unknown command: {args.command}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    try:
        print(run_command(args))
    except (TodoStorageError, ValueError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error
