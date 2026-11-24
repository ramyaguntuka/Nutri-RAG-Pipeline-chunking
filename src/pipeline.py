"""End-to-end NutriRAG flow, including medical-safety and unknown-answer gates."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import re
from typing import Any

from src.llm import generate_response
from src.prompts import SYSTEM_PROMPT_V2, build_user_prompt
from src.retrieval import RetrievalResult, TfidfRetriever

_MEDICAL_PATTERN = re.compile(
    r"\b(diagnos(?:e|is|ing)|symptoms?|medicine|medication|drug|prescri(?:be|ption)|"
    r"treat(?:ment)?|dosage|dose|cure)\b",
    re.IGNORECASE,
)
_OUT_OF_SCOPE_PATTERN = re.compile(
    r"\b(stock price|share price|stock market|share market|stock ticker)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PipelineResult:
    answer: str
    sources: list[RetrievalResult]
    safety_status: str


AnswerGenerator = Callable[[str, str], str]


def _profile_value(profile: Mapping[str, Any], key: str) -> str:
    value = profile.get(key)
    return str(value).strip() if value is not None else ""


def _source_line(sources: list[RetrievalResult]) -> str:
    unique_sources: list[str] = []
    for source in sources:
        citation = f"{source.chunk.document} — {source.chunk.section}"
        if citation not in unique_sources:
            unique_sources.append(citation)
    if not unique_sources:
        return ""
    return "\n\nSources: " + "; ".join(unique_sources)


def run_pipeline(
    question: str,
    profile: Mapping[str, Any],
    retriever: TfidfRetriever | None = None,
    answer_generator: AnswerGenerator | None = None,
    top_k: int = 3,
) -> PipelineResult:
    """Answer one question; inject a generator for deterministic offline tests."""
    if not question.strip():
        raise ValueError("Enter a question before asking NutriRAG.")

    if _MEDICAL_PATTERN.search(question):
        return PipelineResult(
            answer=(
                "I can't diagnose symptoms or recommend medicines or treatment. "
                "Please contact a qualified healthcare professional for advice "
                "about your health."
            ),
            sources=[],
            safety_status="medical_refusal",
        )

    if _OUT_OF_SCOPE_PATTERN.search(question):
        return PipelineResult(
            answer=(
                "I couldn't find this information in my nutrition knowledge base, "
                "so I can't give a reliable answer. Please consult a suitable "
                "up-to-date financial source."
            ),
            sources=[],
            safety_status="unknown_no_grounding",
        )

    index = retriever if retriever is not None else TfidfRetriever()
    sources = index.retrieve(question, top_k=top_k)
    if not sources:
        return PipelineResult(
            answer=(
                "I couldn't find this information in my nutrition knowledge base, "
                "so I can't give a reliable answer. Please consult a suitable "
                "up-to-date source."
            ),
            sources=[],
            safety_status="unknown_no_grounding",
        )

    system_prompt = SYSTEM_PROMPT_V2
    user_prompt = build_user_prompt(question, profile, sources)
    generator = answer_generator if answer_generator is not None else generate_response
    answer = generator(system_prompt, user_prompt).strip()
    if not answer:
        raise RuntimeError("The answer generator returned an empty answer.")

    return PipelineResult(
        answer=answer + _source_line(sources),
        sources=sources,
        safety_status="answered_with_sources",
    )
