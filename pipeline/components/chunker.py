"""Splits documents into overlapping text chunks using recursive character splitting."""

import hashlib
import logging

logger = logging.getLogger(__name__)

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Splits text by the first separator that produces sub-chunks within chunk_size.

    Falls back through the separator list until reaching single-character splits.
    """
    if len(text) <= chunk_size:
        return [text]

    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks: list[str] = []
            current = ""
            for part in parts:
                candidate = current + (sep if current else "") + part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    if len(part) > chunk_size:
                        sub = _recursive_split(part, chunk_size, separators[separators.index(sep) + 1:] or [" "])
                        chunks.extend(sub)
                        current = ""
                    else:
                        current = part
            if current:
                chunks.append(current)
            return chunks

    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _make_chunk_id(source_url: str, text: str) -> str:
    """Returns a deterministic sha256 hex digest of source_url + first 64 chars of text."""
    payload = source_url + text[:64]
    return hashlib.sha256(payload.encode()).hexdigest()


def chunk_documents(
    docs: list[dict],
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[dict]:
    """Splits each document into overlapping chunks with deterministic IDs.

    Splitting priority: double-newline, newline, period-space, space.
    Each chunk overlaps the previous by overlap characters.
    """
    all_chunks: list[dict] = []

    for doc in docs:
        raw_chunks = _recursive_split(doc["content"], chunk_size, _SEPARATORS)

        overlapped: list[str] = []
        for i, chunk in enumerate(raw_chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                tail = raw_chunks[i - 1][-overlap:]
                overlapped.append(tail + chunk)

        for chunk_text in overlapped:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            all_chunks.append(
                {
                    "chunk_id": _make_chunk_id(doc["url"], chunk_text),
                    "source_url": doc["url"],
                    "title": doc["title"],
                    "text": chunk_text,
                }
            )

    logger.info("Chunking complete. Total chunks produced: %d", len(all_chunks))
    return all_chunks
