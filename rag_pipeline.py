from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

from memory import SessionManager
from retriever import HybridRetriever, RetrievalResult


class RAGPipeline:
    def __init__(self, index_dir: str | Path = "indexes", runs_dir: str | Path = "runs") -> None:
        self.retriever = HybridRetriever(index_dir=index_dir)
        self.sessions = SessionManager()
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def start_session(self, thread_id: str) -> dict[str, object]:
        session = self.sessions.start_session(thread_id)
        run_path = self._ensure_run_dir(session.session_id)
        return {
            "session_id": session.session_id,
            "thread_id": session.thread_id,
            "run_dir": str(run_path),
            "available_threads": self.retriever.available_threads(),
        }

    def switch_thread(self, thread_id: str, session_id: str | None = None) -> dict[str, object]:
        session = self.sessions.switch_thread(thread_id, session_id=session_id)
        return {
            "session_id": session.session_id,
            "thread_id": session.thread_id,
            "available_threads": self.retriever.available_threads(),
        }

    def reset_session(self, session_id: str | None = None) -> dict[str, object]:
        self.sessions.reset(session_id=session_id)
        return {"status": "reset", "session_id": session_id}

    def list_threads(self) -> list[dict[str, object]]:
        return self.retriever.available_threads()

    def ask(
        self,
        session_id: str,
        text: str,
        allow_cross_thread: bool = False,
        top_k: int = 8,
    ) -> dict[str, object]:
        started = time.perf_counter()
        trace_id = str(uuid.uuid4())
        session = self.sessions.get(session_id)
        rewrite = self.sessions.rewrite_query(session_id, text)
        retrieved = self.retriever.search(
            query=rewrite,
            thread_id=session.thread_id,
            top_k=min(top_k, 8),
            allow_cross_thread=allow_cross_thread,
        )
        answer, citations, used_docs = self._answer_question(text, retrieved)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        self.sessions.remember_turn(session_id, text, answer, rewrite)
        self._log_turn(
            session_id=session_id,
            trace_id=trace_id,
            thread_id=session.thread_id,
            user_query=text,
            rewritten_query=rewrite,
            retrieved=retrieved,
            used_docs=used_docs,
            answer=answer,
            citations=citations,
            latency_ms=latency_ms,
        )

        return {
            "answer": answer,
            "citations": citations,
            "rewrite": rewrite,
            "retrieved": [self._serialize_result(item) for item in retrieved],
            "trace_id": trace_id,
        }

    def _answer_question(
        self,
        user_text: str,
        retrieved: list[RetrievalResult],
    ) -> tuple[str, list[str], list[str]]:
        if not retrieved:
            return (
                "I do not have evidence in the selected thread yet. Please mention a message, attachment, sender, or date to narrow the search.",
                [],
                [],
            )

        ranked_sentences: list[tuple[float, str, str, str]] = []
        query_terms = set(re.findall(r"\b\w+\b", user_text.lower()))
        compare_mode = any(term in query_terms for term in {"compare", "difference", "earlier", "later"})

        for result in retrieved[:5]:
            citation = self._format_citation(result.metadata)
            sentences = re.split(r"(?<=[.!?])\s+", result.text)
            for sentence in sentences:
                cleaned = sentence.strip()
                if len(cleaned) < 25:
                    continue
                overlap = len(query_terms & set(re.findall(r"\b\w+\b", cleaned.lower())))
                score = overlap + result.fused_score * 100
                if compare_mode and result.metadata.get("page_no") is not None:
                    score += 0.5
                ranked_sentences.append((score, cleaned, citation, result.doc_id))

        ranked_sentences.sort(key=lambda item: item[0], reverse=True)
        selected: list[tuple[str, str, str]] = []
        seen_sentences: set[str] = set()
        for _, sentence, citation, doc_id in ranked_sentences:
            if sentence in seen_sentences:
                continue
            seen_sentences.add(sentence)
            selected.append((sentence, citation, doc_id))
            if len(selected) >= 3:
                break

        if not selected:
            return (
                "I found potentially related documents, but not enough directly relevant evidence. Please ask with a sender, filename, amount, or date.",
                [],
                [],
            )

        answer_lines = [f"{sentence} {citation}" for sentence, citation, _ in selected]
        citations = [citation for _, citation, _ in selected]
        used_docs = [doc_id for _, _, doc_id in selected]
        return ("\n".join(answer_lines), citations, used_docs)

    def _log_turn(
        self,
        session_id: str,
        trace_id: str,
        thread_id: str,
        user_query: str,
        rewritten_query: str,
        retrieved: list[RetrievalResult],
        used_docs: list[str],
        answer: str,
        citations: list[str],
        latency_ms: float,
    ) -> None:
        run_dir = self._ensure_run_dir(session_id)
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace_id": trace_id,
            "thread_id": thread_id,
            "user_query": user_query,
            "rewritten_query": rewritten_query,
            "retrieved_documents": [self._serialize_result(item) for item in retrieved],
            "documents_used": used_docs,
            "answer": answer,
            "citations": citations,
            "latency_ms": latency_ms,
        }
        with (run_dir / "trace.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _ensure_run_dir(self, session_id: str) -> Path:
        run_dir = self.runs_dir / session_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def _format_citation(metadata: dict[str, object]) -> str:
        message_id = metadata.get("message_id") or "unknown-message"
        page_no = metadata.get("page_no")
        if page_no:
            return f"[msg: {message_id}, page: {page_no}]"
        return f"[msg: {message_id}]"

    def _serialize_result(self, item: RetrievalResult) -> dict[str, object]:
        return {
            "doc_id": item.doc_id,
            "text": item.text[:500],
            "metadata": item.metadata,
            "bm25_score": round(item.bm25_score, 4),
            "vector_score": round(item.vector_score, 4),
            "fused_score": round(item.fused_score, 6),
            "citation": self._format_citation(item.metadata),
        }
