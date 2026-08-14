"""Answer-quality evaluation for source-aware RAG.

This module adds the *generation* side of RAG evaluation on top of the
retrieval metrics in :mod:`paper_api.evaluation`:

* Citation correctness (offline): verify that any page numbers the generated
  answer mentions are actually present in the retrieved evidence. An answer that
  cites a page the system never retrieved is a hallucinated source.
* Faithfulness (optional LLM judge): verify the answer is grounded only in the
  supplied evidence. This needs an LLM, so it is skipped when not configured.

Both checks feed a single end-to-end report so retrieval quality and answer
quality can be compared on the same fixed evaluation set.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from .evaluation import EvaluationCase, evaluate_retrieval
from .retrieval import TextEmbedder
from .services import NoRelevantEvidenceError, answer_question, retrieve_chunks
from .settings import load_local_env


# --------------------------------------------------------------------------- #
# Citation correctness (offline, no LLM required)
# --------------------------------------------------------------------------- #
_CITED_PAGE_PATTERNS = [
    re.compile(r"page\s+(\d+)", re.IGNORECASE),
    re.compile(r"p\.?\s*(\d+)", re.IGNORECASE),
    re.compile(r"第\s*(\d+)\s*页"),
]


def extract_cited_pages(answer: str) -> list[int]:
    """Extract explicit page references from free-text generated answers.

    Supports ``page 5``, ``p.5``, ``p 5`` and Chinese ``第5页`` / ``第 5 页``.
    Duplicates are removed while keeping first-seen order.
    """
    if not answer:
        return []
    pages: list[int] = []
    for pattern in _CITED_PAGE_PATTERNS:
        for match in pattern.findall(answer):
            pages.append(int(match))
    seen: set[int] = set()
    result: list[int] = []
    for page in pages:
        if page not in seen:
            seen.add(page)
            result.append(page)
    return result


@dataclass(frozen=True)
class CitationCheck:
    cited_pages: list[int]
    evidence_pages: list[int]
    unsupported_pages: list[int]
    consistent: bool


def check_citation_correctness(answer: str, evidence_pages: set[int]) -> CitationCheck:
    """Check that page numbers mentioned in the answer exist in retrieved evidence.

    An empty ``cited_pages`` is treated as consistent (no violation), but callers
    can inspect ``cited_pages`` to detect answers that never attribute a source.
    """
    cited_pages = extract_cited_pages(answer)
    unsupported_pages = [page for page in cited_pages if page not in evidence_pages]
    return CitationCheck(
        cited_pages=cited_pages,
        evidence_pages=sorted(evidence_pages),
        unsupported_pages=unsupported_pages,
        consistent=len(unsupported_pages) == 0,
    )


# --------------------------------------------------------------------------- #
# Faithfulness judge (optional, needs an LLM)
# --------------------------------------------------------------------------- #
class FaithfulnessError(Exception):
    """Raised when a faithfulness judgement cannot be produced."""


class LLMNotConfiguredForJudgingError(Exception):
    """Raised when the judge LLM settings are missing."""


@dataclass(frozen=True)
class FaithfulVerdict:
    faithful: bool
    reason: str


class FaithfulnessJudge(Protocol):
    def judge(self, question: str, evidence: str, answer: str) -> FaithfulVerdict:
        """Decide whether ``answer`` is grounded only in ``evidence``."""


class OpenAICompatibleFaithfulnessJudge:
    """LLM-as-judge for answer faithfulness, mirroring ``OpenAICompatibleClient``.

    It reuses the same ``LLM_*`` environment variables as the answer generator so
    a single configuration drives both generation and judging.
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = httpx.Timeout(timeout_seconds, connect=10.0, read=timeout_seconds, write=10.0, pool=10.0)

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleFaithfulnessJudge":
        load_local_env()
        base_url = os.getenv("LLM_BASE_URL", "").strip()
        api_key = os.getenv("LLM_API_KEY", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        if not all((base_url, api_key, model)):
            raise LLMNotConfiguredForJudgingError(
                "Set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL before running faithfulness judging"
            )
        return cls(base_url=base_url, api_key=api_key, model=model)

    def judge(self, question: str, evidence: str, answer: str) -> FaithfulVerdict:
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict faithfulness checker for retrieval-augmented answers. "
                        "Return JSON only with keys: faithful (boolean true/false) and reason (one sentence). "
                        "faithful is true only if every factual claim in the answer is supported by the evidence "
                        "and the answer invents nothing outside it."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Evidence:\n{evidence[:40_000]}\n\n"
                        f"Answer to check:\n{answer[:10_000]}"
                    ),
                },
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable judge response", request=response.request, response=response)
                response.raise_for_status()
                return self._parse(response.json())
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as error:
                last_error = error
                if attempt == 2:
                    break
                time.sleep(0.5 * (2**attempt))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise FaithfulnessError("Judge response did not contain a valid faithfulness JSON object") from error
        raise FaithfulnessError("Faithfulness judge failed after 3 attempts") from last_error

    def _parse(self, payload: object) -> FaithfulVerdict:
        if not isinstance(payload, dict):
            raise FaithfulnessError("Judge response root must be an object")
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content) if isinstance(content, str) else content
        if not isinstance(parsed, dict) or "faithful" not in parsed:
            raise FaithfulnessError("Judge output needs a 'faithful' boolean")
        faithful = bool(parsed["faithful"])
        reason = str(parsed.get("reason", "")).strip()
        return FaithfulVerdict(faithful=faithful, reason=reason)


# --------------------------------------------------------------------------- #
# End-to-end answer evaluation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnswerEvaluationCase:
    question: str
    expected_page_numbers: list[int]
    expected_answer_pages: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class AnswerEvaluationCaseResult:
    question: str
    expected_page_numbers: list[int]
    answer: str
    cited_pages: list[int]
    evidence_pages: list[int]
    citation_consistent: bool
    faithful: bool | None
    faithfulness_reason: str | None


@dataclass(frozen=True)
class AnswerEvaluationReport:
    k: int
    case_count: int
    recall_at_k: float
    mean_reciprocal_rank: float
    citation_correct_rate: float
    faithfulness_run: bool
    faithful_rate: float | None
    results: list[AnswerEvaluationCaseResult]


def evaluate_answers(
    session: object,
    paper_id: int,
    cases: list[AnswerEvaluationCase],
    generator: object,
    judge: FaithfulnessJudge | None = None,
    k: int = 3,
    embedder: TextEmbedder | None = None,
) -> AnswerEvaluationReport:
    """Run retrieval, answer generation, citation checks and optional faithfulness.

    ``generator`` must satisfy the ``AnswerGenerator`` protocol; ``judge`` is an
    optional ``FaithfulnessJudge``. The retrieval metrics reuse
    :func:`paper_api.evaluation.evaluate_retrieval` so both quality dimensions share
    one fixed evaluation set.
    """
    if not cases:
        raise ValueError("at least one evaluation case is required")
    if not 1 <= k <= 10:
        raise ValueError("k must be between 1 and 10")

    def retrieve_pages(question: str, limit: int) -> list[int]:
        try:
            return [
                chunk.page_number
                for chunk, _ in retrieve_chunks(session, paper_id, question, limit=limit, embedder=embedder)
            ]
        except NoRelevantEvidenceError:
            return []

    retrieval_report = evaluate_retrieval(
        [EvaluationCase(question=case.question, expected_page_numbers=case.expected_page_numbers) for case in cases],
        retrieve_pages,
        k,
    )

    results: list[AnswerEvaluationCaseResult] = []
    faithful_count = 0
    for case in cases:
        answer, evidence = answer_question(session, paper_id, case.question, generator, limit=k, embedder=embedder)
        evidence_pages = {chunk.page_number for chunk, _ in evidence}
        citation = check_citation_correctness(answer.answer, evidence_pages)

        faithful: bool | None = None
        reason: str | None = None
        if judge is not None:
            evidence_text = "\n\n".join(
                f"[chunk:{chunk.id}; page:{chunk.page_number}; section:{chunk.section_title or 'unknown'}]\n{chunk.content}"
                for chunk, _ in evidence
            )
            verdict = judge.judge(case.question, evidence_text, answer.answer)
            faithful = verdict.faithful
            reason = verdict.reason
            if faithful:
                faithful_count += 1

        results.append(
            AnswerEvaluationCaseResult(
                question=case.question,
                expected_page_numbers=case.expected_page_numbers,
                answer=answer.answer,
                cited_pages=citation.cited_pages,
                evidence_pages=citation.evidence_pages,
                citation_consistent=citation.consistent,
                faithful=faithful,
                faithfulness_reason=reason,
            )
        )

    citation_correct_rate = sum(result.citation_consistent for result in results) / len(results)
    faithful_rate = (faithful_count / len(results)) if judge is not None else None
    return AnswerEvaluationReport(
        k=k,
        case_count=len(results),
        recall_at_k=retrieval_report.recall_at_k,
        mean_reciprocal_rank=retrieval_report.mean_reciprocal_rank,
        citation_correct_rate=citation_correct_rate,
        faithfulness_run=judge is not None,
        faithful_rate=faithful_rate,
        results=results,
    )
