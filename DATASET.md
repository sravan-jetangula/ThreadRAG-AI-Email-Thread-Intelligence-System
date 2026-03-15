# Dataset Preparation

This project expects a small local slice of the Enron Email Dataset from Kaggle.

## Target Slice

Prepare a subset with the following bounds:

- 10 to 20 threads
- 100 to 300 messages
- 20 to 50 attachments
- total indexed text at or below 100 MB

## Suggested Directory Layout

```text
data/
├── raw/
│   ├── thread_a_email_01.eml
│   ├── thread_a_email_02.eml
│   ├── ...
└── attachments/
```

The ingestion code can also read raw RFC822 files without an extension, which matches common `maildir` exports from Enron corpus releases.

## How To Prepare the Slice

1. Download the Enron Email Dataset from Kaggle.
2. Copy a small set of email files into `data/raw/`.
3. Prefer messages that include MIME attachments in PDF, DOC, DOCX, TXT, or HTML format.
4. Keep the total slice under the project limits for local indexing speed.

## Selection Guidance

Use threads that show:

- approvals or decisions
- budget or vendor discussion
- references to earlier versions of attachments
- multiple participants and dates

That gives the conversational memory and citation logic something meaningful to demonstrate.

## Notes on Attachments

The parser extracts attachments directly from the email MIME payloads during ingestion. If your chosen Enron subset does not contain enough attachment-bearing emails, replace the slice with a different attachment-bearing subset before indexing.

## Ingestion Command

```powershell
python ingest.py --input-dir data/raw --index-dir indexes --attachments-dir data/attachments --max-threads 15 --max-messages 250 --max-attachments 40 --max-bytes-mb 100
```

## Expected Outputs

After ingestion, the following should exist:

- `indexes/faiss.index`
- `indexes/docs.json`
- `indexes/bm25_tokens.json`
- `indexes/threads.json`
- `indexes/ingest_report.json`
- extracted attachments under `data/attachments/`