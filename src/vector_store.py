"""Thin wrapper around ChromaDB.

We use ``PersistentClient`` so the database survives between runs without
needing a separate server process.
"""
import os
import datetime
from typing import Iterable

import chromadb


class VectorStore:
    def __init__(self, db_path: str, collection_name: str = "ned_admissions"):
        os.makedirs(db_path, exist_ok=True)
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # --------------- write ---------------
    def add(self, texts: Iterable[str], embeddings: Iterable[list[float]],
            metadatas: Iterable[dict], ids: Iterable[str] | None = None) -> None:
        texts = list(texts)
        embeddings = list(embeddings)
        metadatas = [self._sanitize_meta(m) for m in metadatas]
        if ids is None:
            ts = int(datetime.datetime.utcnow().timestamp() * 1000)
            ids = [f"chunk_{ts}_{i}" for i in range(len(texts))]
        else:
            ids = list(ids)
        self.collection.add(
            documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

    @staticmethod
    def _sanitize_meta(meta: dict) -> dict:
        """Chroma only accepts str/int/float/bool/None values in metadata."""
        clean = {}
        for k, v in meta.items():
            if v is None or isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    # --------------- read ---------------
    def query(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[embedding], n_results=top_k
        )
        if not results.get("ids") or not results["ids"][0]:
            return []
        out: list[dict] = []
        for i in range(len(results["ids"][0])):
            out.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] or {},
                "distance": (results["distances"][0][i]
                             if results.get("distances") else None),
            })
        return out

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    # --------------- maintenance ---------------
    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
