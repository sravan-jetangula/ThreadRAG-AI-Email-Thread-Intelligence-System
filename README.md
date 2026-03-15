# Email + Attachment RAG with Thread Memory

This project implements a local, thread-locked retrieval-augmented chatbot for email conversations and their attachments. It indexes a small Enron slice, combines BM25 and FAISS retrieval, keeps short conversational memory, rewrites follow-up questions, and returns grounded answers with message-level and page-level citations.

## Features

- Thread-specific retrieval with a hard thread lock by session.
- Hybrid search using BM25 and FAISS with late fusion.
- Attachment parsing for PDF, DOC, DOCX, TXT, and HTML.
- Page-aware citations for PDFs in the form `[msg: <message_id>, page: <page_number>]`.
- Session memory with the last 3 to 5 turns plus lightweight entity notes.
- JSONL tracing for each user turn under `runs/<session_id>/trace.jsonl`.
- FastAPI backend and Streamlit frontend.

## Project Layout

```text
project/
├── ingest.py
├── retriever.py
├── memory.py
├── rag_pipeline.py
├── api.py
├── app.py
├── utils/
│   ├── email_parser.py
│   ├── attachment_parser.py
├── data/
├── indexes/
├── runs/
├── requirements.txt
├── README.md
├── DATASET.md
├── Dockerfile
└── docker-compose.yml
```

## Setup

1. Create a virtual environment.
2. Install dependencies.
3. Download and prepare the Enron slice as described in `DATASET.md`.
4. Run ingestion.
5. Start the API.
6. Start the Streamlit UI.

Example commands:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python ingest.py --input-dir data/raw --index-dir indexes --attachments-dir data/attachments
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
streamlit run app.py
```

## Dataset Description

The intended dataset is a small Enron Email Dataset slice sourced from Kaggle. Target size:

- 10 to 20 threads
- 100 to 300 messages
- 20 to 50 attachments
- total indexed message text under 100 MB

The ingestion script can read both `.eml` files and raw RFC822 message files without an extension, which makes it compatible with the common maildir-style Enron exports on Kaggle.

See `DATASET.md` for preparation details and expected directory structure.

## Indexing Flow

`ingest.py` performs the following steps:

1. Walks the input dataset and parses raw emails.
2. Extracts required metadata: `message_id`, `thread_id`, `date`, `from`, `to`, `cc`, `subject`, and plain text body.
3. Saves MIME attachments to `data/attachments/<thread_id>/<message_id>/`.
4. Chunks emails one message per chunk.
5. Chunks attachments into 200 to 400 token windows with overlap.
6. Builds a FAISS cosine-similarity index from `all-MiniLM-L6-v2` embeddings.
7. Builds a BM25 corpus for keyword retrieval.
8. Writes document metadata and thread summaries to `indexes/`.

Artifacts produced in `indexes/`:

- `faiss.index`
- `docs.json`
- `bm25_tokens.json`
- `threads.json`
- `ingest_report.json`

## Running the System

Backend:

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
streamlit run app.py
```

API endpoints:

- `POST /start_session` with `{ "thread_id": "T-0042" }`
- `POST /ask` with `{ "session_id": "...", "text": "...", "search_outside_thread": false }`
- `POST /switch_thread` with `{ "thread_id": "T-0042", "session_id": "..." }`
- `POST /reset_session` with `{ "session_id": "..." }`
- `GET /threads`
- `GET /health`

Each `/ask` response returns:

```json
{
  "answer": "...",
  "citations": ["[msg: ...]"],
  "rewrite": "...",
  "retrieved": [{"doc_id": "..."}],
  "trace_id": "..."
}
```

## Architecture

The system is built around a small number of focused components:

- `utils/email_parser.py`: parses raw email messages with `mailparser` and derives a stable thread id.
- `utils/attachment_parser.py`: extracts text from supported attachment formats and chunks them.
- `ingest.py`: builds document objects and persists retrieval indexes.
- `retriever.py`: performs BM25 retrieval, FAISS vector retrieval, and late fusion ranking.
- `memory.py`: stores the last few turns and entity notes for conversational query rewriting.
- `rag_pipeline.py`: applies thread locking, query rewriting, retrieval, evidence selection, answer assembly, and trace logging.
- `api.py`: exposes the system through FastAPI.
- `app.py`: provides a Streamlit interface with a thread selector, chat panel, cross-thread toggle, and debug panel.

## Grounding and Citations

Answers are built only from retrieved evidence. Every factual sentence in the assembled answer is suffixed with a citation:

- Email citation: `[msg: <message_id>]`
- Attachment citation: `[msg: <message_id>, page: <page_number>]`

If the evidence is weak or absent, the system asks the user to narrow the request instead of inventing an answer.

## Logging and Traces

Each user turn is appended to:

```text
runs/<session_id>/trace.jsonl
```

Logged fields:

- user query
- rewritten query
- retrieved documents and scores
- documents used
- answer
- citations
- latency

## Example Evaluation Queries

For one selected thread, try:

- `What did finance approve for the storage vendor?`
- `When was that approval sent?`
- `Compare it with the earlier attachment.`

These are intended to demonstrate thread-specific retrieval, follow-up rewriting, and attachment citations.

## Performance Notes

The implementation is designed for a laptop-sized local index.

- Retrieval uses `top_k <= 8`.
- Hybrid search is lightweight enough for small local corpora.
- The answer stage is extractive and citation-first, which keeps latency lower than using a heavier local generator.

## Limitations

- Legacy `.doc` parsing is best-effort because binary Word documents are difficult to handle reliably without heavier external dependencies.
- Thread ids are derived heuristically from headers and normalized subject lines, so malformed headers can reduce thread quality.
- The answer generator is extractive rather than abstractive. It favors grounded evidence and speed over fluent synthesis.
- The project assumes a local prepared dataset slice and does not download Kaggle data automatically.