"""Minimal API-key authentication for public deployments."""

from __future__ import annotations

import os

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Routes reachable without an API key so the chat UI, static assets, the
# health probe and the interactive API docs load for any visitor. All data
# routes (papers, corpus retrieval, answers) stay protected.
_PUBLIC_EXACT = {"/", "/health", "/docs", "/openapi.json"}
_PUBLIC_PREFIXES = ("/insight", "/static")


def verify_api_key(
    request: Request | None = None,
    api_key: str | None = Security(api_key_header),
) -> None:
    """Require X-API-Key when RAG_API_KEY is configured.

    The key is read from the process environment directly (not via
    load_local_env) so per-request auth never re-injects .env values, which
    would break tests that delete env vars to assert configuration errors.
    Production deployments inject RAG_API_KEY through the container environment.
    """
    if request is not None and (
        request.url.path in _PUBLIC_EXACT or request.url.path.startswith(_PUBLIC_PREFIXES)
    ):
        return
    expected = os.getenv("RAG_API_KEY", "").strip()
    if expected and api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


__all__ = ["api_key_header", "verify_api_key"]
