from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Retrieval-augmented knowledge base for C3 consistency analysis.

    Stores (command, state, candidate_plans, label, rationale) exemplars
    and retrieves similar past cases to inform LLM-based ambiguity judgment.
    """

    def __init__(self, path: str, embedding_model: Any, top_k: int = 3) -> None:
        self.path = Path(path)
        self._model = embedding_model
        self._top_k = int(top_k)
        self.entries: List[Dict[str, Any]] = self._load()
        self._embeddings_cache: Optional[Any] = None
        if self.entries:
            self._build_embedding_cache()

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            logger.warning("Knowledge base file is not a JSON array: %s", self.path)
            return []
        except Exception as exc:
            logger.warning("Failed to load knowledge base %s: %s", self.path, exc)
            return []

    def _build_embedding_cache(self) -> None:
        commands = [entry.get("command", "") for entry in self.entries]
        if not any(commands):
            self._embeddings_cache = None
            return
        try:
            self._embeddings_cache = self._model.encode(
                commands,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.warning("Failed to build knowledge base embedding cache: %s", exc)
            self._embeddings_cache = None

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.entries or self._embeddings_cache is None:
            return []
        k = top_k if top_k is not None else self._top_k
        try:
            query_embedding = self._model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
        except Exception as exc:
            logger.warning("Knowledge base query encoding failed: %s", exc)
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        scores = cosine_similarity([query_embedding], self._embeddings_cache)[0]
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        results: List[Dict[str, Any]] = []
        for idx, score in indexed[:k]:
            if score <= 0:
                break
            entry = dict(self.entries[idx])
            entry["retrieval_score"] = float(score)
            results.append(entry)
        return results

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return len(self.entries) > 0
