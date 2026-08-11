"""Retry a callable with exponential backoff."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from typing import TypeVar

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def with_retry(
    operation: Callable[[], T],
    attempts: int = 3,
    base_delay: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run an operation and retry failures up to ``attempts`` times."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == attempts:
                LOGGER.error("Operation failed after %s attempts", attempts)
                raise
            delay = base_delay * (2 ** (attempt - 1))
            LOGGER.warning("Attempt %s failed: %s; retrying in %.2fs", attempt, error, delay)
            sleep(delay)

    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate retry with exponential backoff.")
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()

    state = {"calls": 0}

    def flaky_operation() -> str:
        state["calls"] += 1
        if state["calls"] < 2:
            raise ConnectionError("simulated temporary failure")
        return "operation succeeded"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print(with_retry(flaky_operation, attempts=args.attempts))


if __name__ == "__main__":
    main()
