"""Small command-line argument parsing example."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Format a short study note.")
    parser.add_argument("topic", help="Topic to print.")
    parser.add_argument("--pages", type=int, default=1, help="Number of pages (default: 1).")
    parser.add_argument("--tag", action="append", default=[], help="Optional tag; repeatable.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.pages < 1:
        raise SystemExit("--pages must be at least 1")
    tags = ", ".join(args.tag) if args.tag else "none"
    print(f"Topic: {args.topic}")
    print(f"Pages: {args.pages}")
    print(f"Tags: {tags}")


if __name__ == "__main__":
    main()
