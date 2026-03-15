from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_pipeline import RAGPipeline


class StartSessionRequest(BaseModel):
    thread_id: str = Field(..., description="Thread identifier to lock the session to.")


class AskRequest(BaseModel):
    session_id: str
    text: str
    search_outside_thread: bool = False


class SwitchThreadRequest(BaseModel):
    thread_id: str
    session_id: str | None = None


class ResetSessionRequest(BaseModel):
    session_id: str | None = None


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


app = FastAPI(title="Email Attachment RAG", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/threads")
def threads() -> list[dict[str, object]]:
    return get_pipeline().list_threads()


@app.post("/start_session")
def start_session(payload: StartSessionRequest) -> dict[str, object]:
    return get_pipeline().start_session(payload.thread_id)


@app.post("/ask")
def ask(payload: AskRequest) -> dict[str, object]:
    try:
        return get_pipeline().ask(
            session_id=payload.session_id,
            text=payload.text,
            allow_cross_thread=payload.search_outside_thread,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/switch_thread")
def switch_thread(payload: SwitchThreadRequest) -> dict[str, object]:
    try:
        return get_pipeline().switch_thread(thread_id=payload.thread_id, session_id=payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/reset_session")
def reset_session(payload: ResetSessionRequest | None = None) -> dict[str, object]:
    session_id = payload.session_id if payload else None
    return get_pipeline().reset_session(session_id=session_id)