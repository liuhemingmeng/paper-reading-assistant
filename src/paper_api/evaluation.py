"""Deterministic retrieval evaluation for source-aware RAG."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_page_numbers: list[int]


@dataclass(frozen=True)
class EvaluationCaseResult:
    question: str
    expected_page_numbers: list[int]
    retrieved_page_numbers: list[int]
    hit: bool
    reciprocal_rank: float


@dataclass(frozen=True)
class EvaluationReport:
    k: int
    case_count: int
    recall_at_k: float
    mean_reciprocal_rank: float
    results: list[EvaluationCaseResult]


def evaluate_retrieval(
    cases: list[EvaluationCase],
    retrieve_pages: Callable[[str, int], list[int]],
    k: int,
) -> EvaluationReport:
    """Evaluate a retrieval callback against expected evidence page numbers."""
    if not cases:
        raise ValueError("at least one evaluation case is required")
    if not 1 <= k <= 10:
        raise ValueError("k must be between 1 and 10")

    results: list[EvaluationCaseResult] = []
    for case in cases:
        if not case.question.strip():
            raise ValueError("evaluation question must not be blank")
        if not case.expected_page_numbers or any(page < 1 for page in case.expected_page_numbers):
            raise ValueError("expected_page_numbers must contain positive page numbers")
        retrieved_page_numbers = retrieve_pages(case.question, k)
        first_rank = next(
            (
                rank
                for rank, page_number in enumerate(retrieved_page_numbers, start=1)
                if page_number in case.expected_page_numbers
            ),
            None,
        )
        results.append(
            EvaluationCaseResult(
                question=case.question,
                expected_page_numbers=case.expected_page_numbers,
                retrieved_page_numbers=retrieved_page_numbers,
                hit=first_rank is not None,
                reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
            )
        )

    return EvaluationReport(
        k=k,
        case_count=len(results),
        recall_at_k=sum(result.hit for result in results) / len(results),
        mean_reciprocal_rank=sum(result.reciprocal_rank for result in results) / len(results),
        results=results,
    )
