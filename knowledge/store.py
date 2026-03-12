"""Local vector knowledge base backed by ChromaDB.

Handles both document storage and RAG-style retrieval.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import chromadb

import config
from .models import VisaDocument

_COLLECTION_NAME = "visa_docs"


class KnowledgeStore:
    """Persistent vector store for visa documents."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or config.KB_PATH
        self._path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self._path))
        self._col = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_document(self, doc: VisaDocument) -> None:
        """Upsert a document (de-duplicates by URL hash)."""
        # Chunk long documents to stay within embedding limits
        chunks = _chunk(doc.content, max_chars=1500)
        ids = [f"{doc.doc_id}_{i}" for i in range(len(chunks))]
        metas = [doc.to_metadata() for _ in chunks]

        self._col.upsert(documents=chunks, metadatas=metas, ids=ids)

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        n_results: int = 5,
    ) -> list[dict]:
        """Semantic search; returns list of {content, metadata, distance}."""
        total = self._col.count()
        if total == 0:
            return []

        where = _build_where(origin, destination)
        kwargs: dict = {
            "query_texts": [query],
            "n_results": min(n_results, total),
        }
        if where:
            kwargs["where"] = where

        try:
            res = self._col.query(**kwargs)
        except Exception:
            return []

        docs = []
        for i, text in enumerate(res["documents"][0]):
            docs.append(
                {
                    "content": text,
                    "metadata": res["metadatas"][0][i],
                    "distance": (res.get("distances") or [[]])[0][i]
                    if res.get("distances")
                    else 0.0,
                }
            )
        return docs

    def has_recent_data(
        self,
        origin: str,
        destination: str,
        hours: Optional[int] = None,
    ) -> bool:
        """Return True if fresh documents exist for this pair."""
        ttl = hours or config.CACHE_TTL_HOURS
        cutoff = datetime.utcnow() - timedelta(hours=ttl)

        try:
            res = self._col.get(where=_build_where(origin, destination))
            for meta in res.get("metadatas") or []:
                try:
                    ts = datetime.fromisoformat(meta["retrieval_time"])
                    if ts > cutoff:
                        return True
                except (KeyError, ValueError):
                    continue
        except Exception:
            pass
        return False

    def count(self) -> int:
        return self._col.count()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk(text: str, max_chars: int = 1500) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    overlap = 200
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def _build_where(origin: Optional[str], destination: Optional[str]) -> dict:
    if origin and destination:
        return {
            "$and": [
                {"origin_country": {"$eq": origin}},
                {"destination_country": {"$eq": destination}},
            ]
        }
    if destination:
        return {"destination_country": {"$eq": destination}}
    if origin:
        return {"origin_country": {"$eq": origin}}
    return {}
