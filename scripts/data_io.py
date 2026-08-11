"""Read and write JSON and CSV files using the Python standard library."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create sample JSON and CSV files.")
    parser.add_argument("output_dir", type=Path, help="Directory for generated examples.")
    args = parser.parse_args()

    sample = {"title": "Attention Is All You Need", "pages": 15}
    rows = [
        {"name": "Alice", "role": "student"},
        {"name": "Bob", "role": "engineer"},
    ]
    json_path = args.output_dir / "sample.json"
    csv_path = args.output_dir / "sample.csv"
    write_json(json_path, sample)
    write_csv(csv_path, rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Read JSON: {read_json(json_path)}")
    print(f"Read CSV: {read_csv(csv_path)}")


if __name__ == "__main__":
    main()
