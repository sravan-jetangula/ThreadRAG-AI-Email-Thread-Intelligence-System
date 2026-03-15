from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mailparser


THREAD_PREFIX = "T-"


@dataclass(slots=True)
class ParsedAttachment:
    filename: str
    content_type: str | None
    payload: bytes


@dataclass(slots=True)
class ParsedEmail:
    message_id: str
    thread_id: str
    date: str | None
    sender: str | None
    to: list[str]
    cc: list[str]
    subject: str | None
    body: str
    attachments: list[ParsedAttachment]
    source_path: str


def _clean_header_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        flat = [part for part in value if part]
        return ", ".join(flat) if flat else None
    text = str(value).strip()
    return text or None


def _normalize_subject(subject: str | None) -> str:
    if not subject:
        return "no-subject"
    normalized = re.sub(r"^(re|fw|fwd)\s*:\s*", "", subject, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized or "no-subject"


def _message_identifier(raw_message_id: str | None, source_path: Path) -> str:
    cleaned = _clean_header_value(raw_message_id)
    if cleaned:
        return cleaned.strip("<>")
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:16]
    return f"msg-{digest}"


def _thread_identifier(parsed_mail: mailparser.MailParser, subject: str | None, message_id: str) -> str:
    references = parsed_mail.headers.get("References") or parsed_mail.headers.get("references")
    in_reply_to = parsed_mail.headers.get("In-Reply-To") or parsed_mail.headers.get("in-reply-to")

    if references:
        refs = re.findall(r"<([^>]+)>", str(references))
        anchor = refs[0] if refs else str(references)
    elif in_reply_to:
        matches = re.findall(r"<([^>]+)>", str(in_reply_to))
        anchor = matches[0] if matches else str(in_reply_to)
    else:
        anchor = _normalize_subject(subject)

    stable_input = anchor or message_id
    digest = hashlib.sha1(stable_input.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"{THREAD_PREFIX}{digest}"


def _collect_recipients(values: Any) -> list[str]:
    if not values:
        return []
    recipients: list[str] = []
    if isinstance(values, list):
        for item in values:
            if isinstance(item, tuple):
                address = ", ".join(str(part) for part in item if part)
            else:
                address = str(item)
            cleaned = address.strip()
            if cleaned:
                recipients.append(cleaned)
        return recipients

    text = str(values)
    for part in re.split(r"[;,]", text):
        cleaned = part.strip()
        if cleaned:
            recipients.append(cleaned)
    return recipients


def _extract_plain_text(parsed_mail: mailparser.MailParser) -> str:
    text_plain = parsed_mail.text_plain or []
    if text_plain:
        parts = [part.strip() for part in text_plain if part and part.strip()]
        if parts:
            return "\n\n".join(parts)
    body = parsed_mail.body or ""
    return body.strip()


def parse_email(source_path: str | Path) -> ParsedEmail:
    path = Path(source_path)
    parsed_mail = mailparser.parse_from_file(str(path))

    subject = _clean_header_value(parsed_mail.subject)
    message_id = _message_identifier(parsed_mail.message_id, path)
    thread_id = _thread_identifier(parsed_mail, subject, message_id)
    attachments: list[ParsedAttachment] = []

    for attachment in parsed_mail.attachments or []:
        filename = attachment.get("filename") or attachment.get("mail_content_type") or "attachment.bin"
        payload = attachment.get("payload") or b""
        if isinstance(payload, str):
            payload = payload.encode("utf-8", errors="ignore")
        attachments.append(
            ParsedAttachment(
                filename=filename,
                content_type=attachment.get("mail_content_type"),
                payload=payload,
            )
        )

    return ParsedEmail(
        message_id=message_id,
        thread_id=thread_id,
        date=_clean_header_value(parsed_mail.date),
        sender=_clean_header_value(parsed_mail.from_),
        to=_collect_recipients(parsed_mail.to),
        cc=_collect_recipients(parsed_mail.cc),
        subject=subject,
        body=_extract_plain_text(parsed_mail),
        attachments=attachments,
        source_path=str(path),
    )