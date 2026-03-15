from __future__ import annotations

import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone


PRONOUN_CUES = {
    "it",
    "that",
    "those",
    "they",
    "them",
    "he",
    "she",
    "this",
    "earlier",
    "approval",
    "version",
    "file",
    "attachment",
}

FILENAME_RE = re.compile(r"\b[\w\-. ]+\.(?:pdf|doc|docx|txt|html|htm)\b", flags=re.IGNORECASE)
AMOUNT_RE = re.compile(r"\$\s?[\d,]+(?:\.\d{2})?")
DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?)\b", flags=re.IGNORECASE)
NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


@dataclass(slots=True)
class ChatTurn:
    user: str
    answer: str
    rewrite: str
    created_at: str


@dataclass(slots=True)
class SessionState:
    session_id: str
    thread_id: str
    recent_turns: deque[ChatTurn] = field(default_factory=lambda: deque(maxlen=5))
    entity_notes: dict[str, list[str]] = field(default_factory=lambda: {
        "people": [],
        "dates": [],
        "filenames": [],
        "amounts": [],
    })


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._active_session_id: str | None = None

    def start_session(self, thread_id: str) -> SessionState:
        session_id = str(uuid.uuid4())
        state = SessionState(session_id=session_id, thread_id=thread_id)
        self._sessions[session_id] = state
        self._active_session_id = session_id
        return state

    def get(self, session_id: str | None = None) -> SessionState:
        resolved_session_id = session_id or self._active_session_id
        if not resolved_session_id or resolved_session_id not in self._sessions:
            raise KeyError("Session not found")
        return self._sessions[resolved_session_id]

    def switch_thread(self, thread_id: str, session_id: str | None = None) -> SessionState:
        state = self.get(session_id)
        state.thread_id = thread_id
        state.recent_turns.clear()
        state.entity_notes = {"people": [], "dates": [], "filenames": [], "amounts": []}
        return state

    def reset(self, session_id: str | None = None) -> None:
        if session_id:
            self._sessions.pop(session_id, None)
            if self._active_session_id == session_id:
                self._active_session_id = None
            return
        self._sessions.clear()
        self._active_session_id = None

    def remember_turn(self, session_id: str, user: str, answer: str, rewrite: str) -> SessionState:
        state = self.get(session_id)
        state.recent_turns.append(
            ChatTurn(
                user=user,
                answer=answer,
                rewrite=rewrite,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        self._merge_entities(state.entity_notes, self._extract_entities(user))
        self._merge_entities(state.entity_notes, self._extract_entities(answer))
        return state

    def rewrite_query(self, session_id: str, user_text: str) -> str:
        state = self.get(session_id)
        normalized = user_text.strip()
        if not normalized:
            return normalized

        lowered_tokens = set(re.findall(r"\b\w+\b", normalized.lower()))
        needs_memory = bool(lowered_tokens & PRONOUN_CUES) or len(normalized.split()) <= 6
        if not needs_memory:
            return normalized

        notes = []
        for key in ("filenames", "people", "dates", "amounts"):
            values = state.entity_notes.get(key, [])[:2]
            if values:
                notes.append(f"{key}: {', '.join(values)}")

        turns = list(state.recent_turns)[-3:]
        turn_summaries = [f"Q: {turn.user} | A: {turn.answer}" for turn in turns]
        context_parts = [f"thread {state.thread_id}"]
        if notes:
            context_parts.append("notes " + "; ".join(notes))
        if turn_summaries:
            context_parts.append("recent context " + " || ".join(turn_summaries))
        context = "; ".join(context_parts)
        return f"Resolve references using {context}. User question: {normalized}"

    @staticmethod
    def _extract_entities(text: str) -> dict[str, list[str]]:
        return {
            "people": _dedupe(NAME_RE.findall(text)),
            "dates": _dedupe(DATE_RE.findall(text)),
            "filenames": _dedupe([match.group(0) for match in FILENAME_RE.finditer(text)]),
            "amounts": _dedupe(AMOUNT_RE.findall(text)),
        }

    @staticmethod
    def _merge_entities(target: dict[str, list[str]], extracted: dict[str, list[str]]) -> None:
        for key, values in extracted.items():
            combined = target.get(key, []) + values
            target[key] = _dedupe(combined)[:5]


def _dedupe(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique