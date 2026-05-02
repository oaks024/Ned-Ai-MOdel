"""Word-based text chunking.

We approximate "1000 tokens" as roughly 750 words (1 token ~ 0.75 words for
English text). This avoids a hard dependency on tiktoken and keeps the
behavior predictable for beginners.
"""
from typing import Iterable


def split_text(text: str, chunk_tokens: int = 1000, overlap_tokens: int = 150) -> list[str]:
    words = text.split()
    if not words:
        return []
    target = max(1, int(chunk_tokens * 0.75))
    overlap = max(0, int(overlap_tokens * 0.75))
    step = max(1, target - overlap)

    chunks: list[str] = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + target])
        if chunk.strip():
            chunks.append(chunk)
        if i + target >= len(words):
            break
    return chunks


def chunk_documents(documents: Iterable[dict], chunk_tokens: int = 1000,
                    overlap_tokens: int = 150) -> list[dict]:
    """Split each document's ``text`` field into chunks while preserving metadata.

    Returns a list of {"text": str, "metadata": dict} records.
    """
    out: list[dict] = []
    for doc in documents:
        text = doc.get("text", "")
        chunks = split_text(text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
        meta_base = {k: v for k, v in doc.items() if k != "text"}
        for j, c in enumerate(chunks):
            meta = dict(meta_base)
            meta["chunk_index"] = j
            out.append({"text": c, "metadata": meta})
    return out
