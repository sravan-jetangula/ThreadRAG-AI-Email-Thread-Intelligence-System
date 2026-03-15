from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


TOKEN_RE = re.compile(r"\b\w+\b")


@dataclass(slots=True)
class RetrievalResult:
    doc_id: str
    text: str
    metadata: dict[str, object]
    bm25_score: float
    vector_score: float
    fused_score: float


class HybridRetriever:
    def __init__(
        self,
        index_dir: str | Path = "indexes",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.index_dir = Path(index_dir)
        self.docs = []
        self.doc_lookup: dict[str, dict[str, object]] = {}
        self.doc_count = 0
        self.faiss_index: faiss.Index | None = None
        self.bm25_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self.thread_catalog: list[dict[str, object]] = []
        self.model = SentenceTransformer(model_name)

        docs_path = self.index_dir / "docs.json"
        faiss_path = self.index_dir / "faiss.index"
        bm25_path = self.index_dir / "bm25_tokens.json"
        threads_path = self.index_dir / "threads.json"
        if not all(path.exists() for path in (docs_path, faiss_path, bm25_path, threads_path)):
            return

        self.docs = json.loads(docs_path.read_text(encoding="utf-8"))
        self.doc_lookup = {doc["doc_id"]: doc for doc in self.docs}
        self.doc_count = len(self.docs)
        self.faiss_index = faiss.read_index(str(faiss_path))
        self.bm25_corpus = json.loads(bm25_path.read_text(encoding="utf-8"))
        self.bm25 = BM25Okapi(self.bm25_corpus)
        self.thread_catalog = json.loads(threads_path.read_text(encoding="utf-8"))

    def available_threads(self) -> list[dict[str, object]]:
        return self.thread_catalog

    def search(
        self,
        query: str,
        thread_id: str | None,
        top_k: int = 8,
        allow_cross_thread: bool = False,
    ) -> list[RetrievalResult]:
        if not self.doc_count or self.faiss_index is None or self.bm25 is None:
            return []
        candidate_pool = min(max(top_k * 4, 20), max(self.doc_count, 1))
        allowed_ids = None if allow_cross_thread or not thread_id else {
            doc["doc_id"] for doc in self.docs if doc["metadata"]["thread_id"] == thread_id
        }

        bm25_hits = self._bm25_search(query, candidate_pool, allowed_ids)
        vector_hits = self._vector_search(query, candidate_pool, allowed_ids)
        return self._late_fusion(bm25_hits, vector_hits, top_k)

    def _bm25_search(
        self,
        query: str,
        limit: int,
        allowed_ids: set[str] | None,
    ) -> list[tuple[str, float]]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = np.argsort(scores)[::-1]
        hits: list[tuple[str, float]] = []
        for doc_index in ranked:
            doc = self.docs[int(doc_index)]
            if allowed_ids is not None and doc["doc_id"] not in allowed_ids:
                continue
            score = float(scores[int(doc_index)])
            if score <= 0:
                continue
            hits.append((doc["doc_id"], score))
            if len(hits) >= limit:
                break
        return hits

    def _vector_search(
        self,
        query: str,
        limit: int,
        allowed_ids: set[str] | None,
    ) -> list[tuple[str, float]]:
        embedding = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, indexes = self.faiss_index.search(embedding.astype("float32"), min(limit * 3, self.doc_count))
        hits: list[tuple[str, float]] = []
        for score, index in zip(scores[0], indexes[0], strict=False):
            if index < 0:
                continue
            doc = self.docs[int(index)]
            if allowed_ids is not None and doc["doc_id"] not in allowed_ids:
                continue
            hits.append((doc["doc_id"], float(score)))
            if len(hits) >= limit:
                break
        return hits

    def _late_fusion(
        self,
        bm25_hits: list[tuple[str, float]],
        vector_hits: list[tuple[str, float]],
        top_k: int,
    ) -> list[RetrievalResult]:
        bm25_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_hits, start=1)}
        vector_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(vector_hits, start=1)}
        bm25_scores = dict(bm25_hits)
        vector_scores = dict(vector_hits)

        all_ids = set(bm25_rank) | set(vector_rank)
        fused: list[RetrievalResult] = []
        for doc_id in all_ids:
            fused_score = 0.0
            if doc_id in bm25_rank:
                fused_score += 1.0 / (60 + bm25_rank[doc_id])
            if doc_id in vector_rank:
                fused_score += 1.0 / (60 + vector_rank[doc_id])
            doc = self.doc_lookup[doc_id]
            fused.append(
                RetrievalResult(
                    doc_id=doc_id,
                    text=doc["text"],
                    metadata=doc["metadata"],
                    bm25_score=float(bm25_scores.get(doc_id, 0.0)),
                    vector_score=float(vector_scores.get(doc_id, 0.0)),
                    fused_score=fused_score,
                )
            )

        return sorted(fused, key=lambda item: item.fused_score, reverse=True)[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())