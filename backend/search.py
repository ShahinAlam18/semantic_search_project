from __future__ import annotations

from difflib import SequenceMatcher
from typing import List, Tuple

import numpy as np


class SemanticSearcher:
    def __init__(
        self,
        questions: List[str],
        embeddings: np.ndarray | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.questions = questions
        self.embeddings = embeddings
        self.model_name = model_name
        self.model = None
        self.index = None
        self.dim = 0
        self.backend = "keyword"

        if embeddings is None:
            self._startup_error = "Dataset embeddings unavailable"
            return

        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            if embeddings.ndim != 2 or embeddings.shape[0] != len(questions):
                raise ValueError("Embeddings must have shape (num_questions, dim)")

            self.model = SentenceTransformer(model_name)
            self.dim = int(embeddings.shape[1])
            if self.model.get_sentence_embedding_dimension() != self.dim:
                raise ValueError(
                    f"Embedding dimension mismatch: dataset dim={self.dim}, "
                    f"query model dim={self.model.get_sentence_embedding_dimension()}"
                )

            self.index = faiss.IndexFlatIP(self.dim)
            self.backend = "semantic"
        except Exception as exc:
            self._startup_error = str(exc)
            self.model = None
            self.index = None
            self.dim = 0

    def build_index(self) -> None:
        if self.backend != "semantic" or self.embeddings is None:
            return
        self.index.add(self.embeddings.astype("float32"))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.backend != "semantic" or self.model is None or self.index is None:
            return self._keyword_search(query, top_k)

        query_embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")
        query_embedding = self._normalize(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        query_terms = set(self._tokenize(query))
        if not query_terms:
            return []

        scored = []
        for idx, question in enumerate(self.questions):
            terms = set(self._tokenize(question))
            if not terms:
                continue

            token_overlap = len(query_terms & terms) / max(len(query_terms | terms), 1)
            phrase_similarity = SequenceMatcher(None, query.lower(), question.lower()).ratio()

            # Blend exact token overlap with fuzzy phrase similarity so the
            # fallback search stays useful even when the remote dataset is offline.
            score = (0.7 * token_overlap) + (0.3 * phrase_similarity)
            if score >= 0.08:
                scored.append((idx, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        return [token for token in cleaned.split() if len(token) > 1]

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
