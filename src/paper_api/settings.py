"""Small local environment configuration loader."""

from __future__ import annotations

import os
from pathlib import Path


def load_local_env(path: Path = Path(".env")) -> None:
    """Load plain KEY=VALUE lines without overriding real environment variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        os.environ.setdefault(key.strip(), value.strip())
