from __future__ import annotations

import re
from pathlib import Path

import fitz
from bs4 import BeautifulSoup
from docx import Document


def _normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _best_effort_doc_text(path: Path) -> str:
    raw = path.read_bytes()
    decoded = raw.decode("latin-1", errors="ignore")
    matches = re.findall(r"[A-Za-z0-9][A-Za-z0-9 ,.;:\-/()$%]{4,}", decoded)
    return _normalize_whitespace("\n".join(matches))


def extract_attachment_text(path: str | Path) -> list[dict[str, object]]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        pages: list[dict[str, object]] = []
        with fitz.open(file_path) as document:
            for index, page in enumerate(document, start=1):
                text = _normalize_whitespace(page.get_text("text"))
                if text:
                    pages.append({"page_no": index, "text": text})
        return pages

    if suffix == ".docx":
        document = Document(file_path)
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        normalized = _normalize_whitespace(text)
        return [{"page_no": None, "text": normalized}] if normalized else []

    if suffix == ".doc":
        normalized = _normalize_whitespace(_best_effort_doc_text(file_path))
        return [{"page_no": None, "text": normalized}] if normalized else []

    if suffix in {".txt", ".log", ".csv"}:
        normalized = _normalize_whitespace(file_path.read_text(encoding="utf-8", errors="ignore"))
        return [{"page_no": None, "text": normalized}] if normalized else []

    if suffix in {".html", ".htm"}:
        html = file_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        normalized = _normalize_whitespace(soup.get_text("\n"))
        return [{"page_no": None, "text": normalized}] if normalized else []

    return []


def chunk_text(text: str, chunk_tokens: int = 280, overlap_tokens: int = 40) -> list[str]:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return []
    if len(tokens) <= chunk_tokens:
        return [" ".join(tokens)]

    chunks: list[str] = []
    step = max(chunk_tokens - overlap_tokens, 1)
    for start in range(0, len(tokens), step):
        end = start + chunk_tokens
        chunk = tokens[start:end]
        if not chunk:
            continue
        chunks.append(" ".join(chunk))
        if end >= len(tokens):
            break
    return chunks