"""OpenAI-compatible LLM client for structured reading insights."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from .settings import load_local_env


class LLMConfigurationError(Exception):
    """Raised when required LLM settings are not configured."""


class LLMResponseError(Exception):
    """Raised when an LLM response cannot satisfy the output contract."""


@dataclass(frozen=True)
class ReadingInsight:
    summary: str
    questions: list[str]
    model: str


class InsightGenerator(Protocol):
    def generate(self, text: str) -> ReadingInsight:
        """Generate a summary and five answerable questions."""


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    model: str


class AnswerGenerator(Protocol):
    def answer(self, question: str, evidence: str) -> GroundedAnswer:
        """Answer only from supplied evidence."""


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = httpx.Timeout(timeout_seconds, connect=10.0, read=timeout_seconds, write=10.0, pool=10.0)

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleClient":
        load_local_env()
        base_url = os.getenv("LLM_BASE_URL", "").strip()
        api_key = os.getenv("LLM_API_KEY", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        if not all((base_url, api_key, model)):
            raise LLMConfigurationError("Set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL before generating insights")
        return cls(base_url=base_url, api_key=api_key, model=model)

    def generate(self, text: str) -> ReadingInsight:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise academic reading assistant. Return JSON only with keys: "
                        "summary (string) and questions (array of exactly five concise questions answerable from the text)."
                    ),
                },
                {"role": "user", "content": f"Analyze this paper text:\n\n{text[:40_000]}"},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None

        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable LLM response", request=response.request, response=response)
                response.raise_for_status()
                return self._parse_response(response.json())
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as error:
                last_error = error
                if attempt == 2:
                    break
                time.sleep(0.5 * (2**attempt))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise LLMResponseError("LLM response did not contain a valid insight JSON object") from error

        raise LLMResponseError("LLM request failed after 3 attempts") from last_error

    def answer(self, question: str, evidence: str) -> GroundedAnswer:
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You answer academic questions only from the supplied evidence. "
                        "If the evidence is insufficient, say so clearly. Do not invent facts."
                    ),
                },
                {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{evidence}"},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None

        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable LLM response", request=response.request, response=response)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                answer = str(content).strip()
                if not answer:
                    raise LLMResponseError("LLM returned a blank answer")
                return GroundedAnswer(answer=answer, model=self.model)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as error:
                last_error = error
                if attempt == 2:
                    break
                time.sleep(0.5 * (2**attempt))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise LLMResponseError("LLM response did not contain a valid answer") from error

        raise LLMResponseError("LLM request failed after 3 attempts") from last_error

    def _parse_response(self, payload: object) -> ReadingInsight:
        if not isinstance(payload, dict):
            raise LLMResponseError("LLM response root must be an object")
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content) if isinstance(content, str) else content
        if not isinstance(parsed, dict):
            raise LLMResponseError("LLM content must be a JSON object")
        summary = str(parsed["summary"]).strip()
        questions = parsed["questions"]
        if not summary or not isinstance(questions, list) or len(questions) != 5:
            raise LLMResponseError("LLM output needs a summary and exactly five questions")
        cleaned_questions = [str(question).strip() for question in questions]
        if any(not question for question in cleaned_questions):
            raise LLMResponseError("LLM questions must not be blank")
        return ReadingInsight(summary=summary, questions=cleaned_questions, model=self.model)
