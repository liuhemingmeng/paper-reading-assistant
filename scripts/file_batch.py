"""Preview or rename files in a directory.

The default mode is dry-run: it prints planned changes without modifying files.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview a batch rename operation.")
    parser.add_argument("directory", type=Path, help="Directory containing files.")
    parser.add_argument(
        "--prefix",
        default="renamed",
        help="Prefix to use for renamed files (default: renamed).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files; without this flag the command is dry-run.",
    )
    return parser


def rename_files(directory: Path, prefix: str, apply: bool = False) -> list[tuple[Path, Path]]:
    """Return planned renames and optionally apply them."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    files = sorted(path for path in directory.iterdir() if path.is_file())
    plans = [
        (path, directory / f"{prefix}_{index:03d}{path.suffix}")
        for index, path in enumerate(files, start=1)
    ]

    for source, target in plans:
        LOGGER.info("%s -> %s", source.name, target.name)
        if apply and source != target:
            source.rename(target)

    return plans


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    try:
        rename_files(args.directory, args.prefix, apply=args.apply)
    except (FileNotFoundError, NotADirectoryError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
