"""TF-IDF retrieval over the local, section-aware nutrition knowledge base."""

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.knowledge_base import DocumentChunk, load_chunks


@dataclass(frozen=True)
class RetrievalResult:
    chunk: DocumentChunk
    score: float


class TfidfRetriever:
    """Build a TF-IDF index once and return the most relevant chunks per query."""

    def __init__(
        self,
        chunks: list[DocumentChunk] | None = None,
        min_score: float = 0.08,
    ) -> None:
        self.chunks = chunks if chunks is not None else load_chunks()
        if not self.chunks:
            raise ValueError("Cannot build a retrieval index from an empty knowledge base.")
        self.min_score = min_score
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._generic_terms = {
            "food",
            "foods",
            "health",
            "healthy",
            "meal",
            "meals",
            "nutrition",
            "today",
        }
        self._analyzer = self.vectorizer.build_analyzer()
        indexed_text = [
            f"{chunk.document} {chunk.section} {chunk.text}" for chunk in self.chunks
        ]
        self._indexed_terms = {
            term
            for text in indexed_text
            for term in self._analyzer(text)
            if " " not in term and term not in self._generic_terms
        }
        self.matrix = self.vectorizer.fit_transform(indexed_text)

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        if not question.strip():
            raise ValueError("Enter a question before searching the knowledge base.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        informative_terms = {
            term
            for term in self._analyzer(question)
            if " " not in term and term not in self._generic_terms
        }
        if not informative_terms.intersection(self._indexed_terms):
            return []

        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indices = np.argsort(scores)[::-1]
        return [
            RetrievalResult(self.chunks[int(index)], float(scores[index]))
            for index in ranked_indices[:top_k]
            if scores[index] >= self.min_score
        ]


def retrieve(question: str, top_k: int = 3) -> list[RetrievalResult]:
    """Convenience function for a one-off query."""
    return TfidfRetriever().retrieve(question, top_k=top_k)
