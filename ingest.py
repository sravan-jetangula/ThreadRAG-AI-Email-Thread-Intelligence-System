from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def extract_subject(message: str) -> str:
    """Extract subject from raw email text."""
    match = re.search(r"Subject:(.*)", message)
    if match:
        return match.group(1).strip()
    return "No subject"


def tokenize(text: str):
    """Simple tokenizer."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return text.split()


def build_indexes(docs, index_dir: Path, model_name: str):

    print("Preparing corpus...")

    texts = [doc["text"] for doc in docs]

    tokenized_corpus = [tokenize(text) for text in texts]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    print("Loading embedding model...")
    model = SentenceTransformer(model_name)

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]

    print("Building FAISS index...")
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(faiss_index, str(index_dir / "faiss.index"))

    (index_dir / "docs.json").write_text(
        json.dumps(docs, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    (index_dir / "bm25_tokens.json").write_text(
        json.dumps(tokenized_corpus),
        encoding="utf-8"
    )

    # ---------- THREAD SUMMARY ----------
    thread_map = defaultdict(list)

    for doc in docs:
        thread_map[doc["metadata"]["thread_id"]].append(doc)

    threads = []

    for thread_id, messages in thread_map.items():

        subject = messages[0]["metadata"]["subject"]

        threads.append({
            "thread_id": thread_id,
            "subject": subject,
            "message_count": len(messages),
            "attachments": []
        })

    threads.sort(key=lambda x: x["message_count"], reverse=True)

    (index_dir / "threads.json").write_text(
        json.dumps(threads, indent=2),
        encoding="utf-8"
    )


def run_ingestion(args: argparse.Namespace):

    print("Loading CSV dataset...")

    df = pd.read_csv("data/enron/emails.csv")

    if args.limit:
        df = df.head(args.limit)

    docs = []

    print("Preparing documents...")

    thread_map = {}

    for i, row in df.iterrows():

        message = str(row["message"])

        subject = extract_subject(message)

        thread_id = subject.lower().replace(" ", "_")[:60]

        doc = {
            "doc_id": f"email_{i}",
            "text": message,
            "metadata": {
                "doc_id": f"email_{i}",
                "thread_id": thread_id,
                "message_id": i,
                "subject": subject,
                "source_type": "email"
            }
        }

        docs.append(doc)

    if not docs:
        raise RuntimeError("No documents found in dataset")

    print(f"Loaded {len(docs)} emails")

    build_indexes(
        docs,
        index_dir=Path(args.index_dir),
        model_name=args.embedding_model
    )

    report = {
        "emails_indexed": len(docs),
        "document_chunks": len(docs)
    }

    (Path(args.index_dir) / "ingest_report.json").write_text(
        json.dumps(report, indent=2)
    )

    print("✅ Ingestion complete")

    return report


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index-dir",
        default="indexes"
    )

    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20000,
        help="Limit emails for testing"
    )

    return parser.parse_args()


if __name__ == "__main__":

    result = run_ingestion(parse_args())

    print(json.dumps(result, indent=2))