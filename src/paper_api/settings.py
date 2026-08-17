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


def write_env_values(updates: dict[str, str], path: Path = Path(".env")) -> bool:
    """Persist specific KEY=VALUE pairs into a ``.env`` file, preserving the rest.

    Lines whose key is in ``updates`` are rewritten; comments, blanks and every
    other line are kept verbatim. Keys absent from the file are appended. Returns
    ``True`` when the file was actually written, ``False`` when it does not exist
    (the caller still applies the values to ``os.environ`` for this process).
    """
    target = path.resolve()
    if not target.exists():
        return False
    existing = target.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for raw in existing:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(raw)
            continue
        key = stripped.split("=", maxsplit=1)[0].strip()
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(raw)
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    return True
