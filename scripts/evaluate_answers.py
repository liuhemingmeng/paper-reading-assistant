"""Run end-to-end answer-quality evaluation against a local database.

Examples
--------
Offline demo (no LLM needed, uses a fake generator that echoes evidence pages)::

    python scripts/evaluate_answers.py --paper-id 1 --cases cases.json --fake

Real run (requires LLM_* in .env)::

    python scripts/evaluate_answers.py --paper-id 1 --cases cases.json --faithfulness

The ``cases.json`` file is a list of objects::

    {"question": "...", "expected_page_numbers": [1], "expected_answer_pages": [1]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paper_api.answer_evaluation import (
    AnswerEvaluationCase,
    FaithfulVerdict,
    OpenAICompatibleFaithfulnessJudge,
    evaluate_answers,
)
from paper_api.api import create_app
from paper_api.llm_client import GroundedAnswer, OpenAICompatibleClient


class FakeAnswerGenerator:
    """Offline generator that cites the evidence pages it was given."""

    def answer(self, question: str, evidence: str) -> GroundedAnswer:
        pages: list[int] = []
        for part in evidence.split("[chunk:"):
            if "page:" in part:
                try:
                    pages.append(int(part.split("page:")[1].split(";")[0]))
                except ValueError:
                    pass
        cited = " ".join(f"See page {p}." for p in sorted(set(pages)))
        return GroundedAnswer(answer=f"Fake answer. {cited}".strip(), model="fake-v1")


class FakeJudge:
    def judge(self, question: str, evidence: str, answer: str) -> FaithfulVerdict:
        return FaithfulVerdict(faithful=True, reason="fake judge: always faithful")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG answer quality")
    parser.add_argument("--paper-id", type=int, required=True)
    parser.add_argument("--cases", type=Path, required=True, help="JSON file with evaluation cases")
    parser.add_argument("--db", type=str, default=None, help="Database URL (default: app default)")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--fake", action="store_true", help="Use offline fake generator (no LLM)")
    parser.add_argument("--faithfulness", action="store_true", help="Run LLM faithfulness judge")
    args = parser.parse_args()

    cases_data = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = [
        AnswerEvaluationCase(
            question=item["question"],
            expected_page_numbers=item["expected_page_numbers"],
            expected_answer_pages=item.get("expected_answer_pages", []),
        )
        for item in cases_data
    ]

    app = create_app(database_url=args.db) if args.db else create_app()
    session_factory = app.state.session_factory
    embedder = app.state.embedder

    generator = FakeAnswerGenerator() if args.fake else OpenAICompatibleClient.from_environment()
    judge = None
    if args.faithfulness:
        judge = FakeJudge() if args.fake else OpenAICompatibleFaithfulnessJudge.from_environment()

    with session_factory() as session:
        report = evaluate_answers(session, args.paper_id, cases, generator, judge=judge, k=args.k, embedder=embedder)

    print(json.dumps(report.__dict__, ensure_ascii=False, indent=2, default=lambda value: value.__dict__))
    return 0


if __name__ == "__main__":
    sys.exit(main())
