"""Local vector embeddings and source-aware retrieval primitives."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class EmbeddedText:
    vector: list[float]
    model: str


class TextEmbedder(Protocol):
    def embed(self, text: str) -> EmbeddedText:
        """Turn text into a vector suitable for cosine similarity search."""


class LocalHashingEmbedder:
    """Dependency-free baseline embedding based on deterministic hashed tokens.

    This is a learning baseline for the retrieval pipeline, not a semantic embedding
    model. Keeping it local makes tests deterministic and demonstrates that the
    service layer depends on an interface rather than a vendor SDK.
    """

    model = "local-hashing-v1"

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    def embed(self, text: str) -> EmbeddedText:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return EmbeddedText(vector=normalize(vector), model=self.model)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have identical dimensions")
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


def normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]
