"""Local vector embeddings and source-aware retrieval primitives."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
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


class BM25Retriever:
    """Offline lexical baseline using Okapi BM25.

    Tokenization reuses :data:`TOKEN_PATTERN` from this module so the lexical
    baseline is directly comparable to :class:`LocalHashingEmbedder`. Chinese runs
    are kept whole (not segmented), which intentionally handicaps BM25 on the
    Chinese industry subset — itself an instructive finding about lexical
    retrieval on CJK text.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._doc_freqs: list[Counter] = []
        self._doc_len: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def fit(self, items: list[tuple[str, str]]) -> "BM25Retriever":
        self._doc_ids = [doc_id for doc_id, _ in items]
        tokenized = [Counter(TOKEN_PATTERN.findall(text.lower())) for _, text in items]
        self._doc_freqs = tokenized
        self._doc_len = [sum(freq.values()) for freq in tokenized]
        total = sum(self._doc_len) or 1
        self._avgdl = total / max(len(tokenized), 1)
        df: Counter = Counter()
        for freq in tokenized:
            for token in freq:
                df[token] += 1
        n = len(tokenized)
        self._idf = {
            token: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for token, freq in df.items()
        }
        return self

    def search(self, query: str, top_n: int | None = None) -> list[tuple[str, float]]:
        """Return ``(doc_id, score)`` pairs sorted by descending BM25 score."""
        q_tokens = TOKEN_PATTERN.findall(query.lower())
        scores: list[float] = []
        for i, freq in enumerate(self._doc_freqs):
            score = 0.0
            doc_len = self._doc_len[i]
            for token in q_tokens:
                idf = self._idf.get(token)
                if idf is None:
                    continue
                tf = freq.get(token, 0)
                if tf == 0:
                    continue
                score += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)
                )
            scores.append(score)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        if top_n is not None:
            ranked = ranked[:top_n]
        return [(self._doc_ids[i], scores[i]) for i in ranked]
