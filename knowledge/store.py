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
# Cosine distance threshold: 0 = identical, 2 = opposite.
# Results above this value are too dissimilar to be useful.
# Cosine distance threshold: 0 = identical, 1 = orthogonal, 2 = opposite.
# 0.75 keeps chunks that are at least moderately on-topic; above this the
# content is too dissimilar to be useful context for the guide.
_DEFAULT_MAX_DISTANCE = 0.75


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

    def add_document(self, doc: VisaDocument, _evict: bool = True) -> None:
        """Upsert a document (de-duplicates by URL hash).

        Evicts stale documents for the same country pair before indexing so
        that outdated visa rules are replaced by the new data.
        Pass _evict=False when adding a batch to avoid redundant DB reads
        (call evict_stale() once before the loop instead).
        """
        if _evict:
            self.evict_stale(doc.origin_country, doc.destination_country)
        chunks = _chunk(doc.content, max_chars=2500)
        ids = [f"{doc.doc_id}_{i}" for i in range(len(chunks))]
        metas = [doc.to_metadata() for _ in chunks]

        self._col.upsert(documents=chunks, metadatas=metas, ids=ids)

    def evict_stale(
        self,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        hours: Optional[int] = None,
    ) -> int:
        """Delete documents older than *hours* for the given country pair.

        Returns the number of deleted chunks.
        """
        ttl = hours or config.CACHE_TTL_HOURS
        cutoff = datetime.utcnow() - timedelta(hours=ttl)
        where = _build_where(origin, destination)
        try:
            kwargs: dict = {"where": where} if where else {}
            res = self._col.get(**kwargs)
            stale_ids = []
            for doc_id, meta in zip(res.get("ids") or [], res.get("metadatas") or []):
                try:
                    ts = datetime.fromisoformat(meta["retrieval_time"])
                    if ts <= cutoff:
                        stale_ids.append(doc_id)
                except (KeyError, ValueError):
                    stale_ids.append(doc_id)  # malformed metadata → evict
            if stale_ids:
                self._col.delete(ids=stale_ids)
            return len(stale_ids)
        except Exception:
            return 0

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        n_results: int = 5,
        max_distance: float = _DEFAULT_MAX_DISTANCE,
    ) -> list[dict]:
        """Semantic search; returns list of {content, metadata, distance}.

        Results with cosine distance above *max_distance* are excluded so
        low-relevance chunks don't pollute the LLM context.
        """
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

        distances = (res.get("distances") or [[]])[0]
        docs = []
        for i, text in enumerate(res["documents"][0]):
            dist = distances[i] if i < len(distances) else 0.0
            if dist > max_distance:
                continue
            docs.append(
                {
                    "content": text,
                    "metadata": res["metadatas"][0][i],
                    "distance": dist,
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

def _chunk(text: str, max_chars: int = 2500) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries.

    Breaking at '\n\n' keeps semantically coherent paragraphs together.
    Falls back to sentence boundaries, then hard character cuts.
    """
    if len(text) <= max_chars:
        return [text]

    overlap = 400
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break

        # 1. Prefer paragraph boundary in the last `overlap` chars of the window
        boundary = text.rfind("\n\n", end - overlap, end)
        if boundary != -1 and boundary > start:
            chunks.append(text[start:boundary].strip())
            start = boundary - overlap
            continue

        # 2. Sentence boundary in the last 300 chars
        boundary = text.rfind(". ", end - 300, end)
        if boundary != -1 and boundary > start:
            chunks.append(text[start:boundary + 1].strip())
            start = boundary + 1 - overlap
            continue

        # 3. Hard cut
        chunks.append(text[start:end].strip())
        start = end - overlap

        if start < 0:
            start = 0

    return [c for c in chunks if c.strip()]


def _build_where(origin: Optional[str], destination: Optional[str]) -> dict:
    """Build a ChromaDB where-filter, normalizing to lowercase for consistency."""
    origin = origin.strip().lower() if origin else None
    destination = destination.strip().lower() if destination else None

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
