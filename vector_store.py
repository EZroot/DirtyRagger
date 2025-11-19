import os
import json
import hashlib
from typing import List, Dict

import faiss
import numpy as np


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PersistentVectorStore:
    """
    FAISS-based persistent vector store with simple metadata and id-tracking.
    Stores:
      ./data/index.faiss
      ./data/meta.jsonl   (one JSON object per line)
      ./data/ids.json     (list of chunk ids)
    """

    def __init__(self, dim: int, data_dir: str = "./data"):
        self.dim = dim
        self.data_dir = data_dir
        self.index_path = os.path.join(data_dir, "index.faiss")
        self.meta_path = os.path.join(data_dir, "meta.jsonl")
        self.ids_path = os.path.join(data_dir, "ids.json")

        os.makedirs(data_dir, exist_ok=True)

        # Load or create FAISS index (inner product on normalized vectors => cosine)
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(dim)

        # Load metadata list (order must match FAISS vector order)
        self.metadata: List[Dict] = []
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.metadata.append(json.loads(line))

        # Known chunk ids (for dedup/incremental)
        if os.path.exists(self.ids_path):
            with open(self.ids_path, "r", encoding="utf-8") as f:
                self.known_ids = set(json.load(f))
        else:
            self.known_ids = set()

    def has_id(self, cid: str) -> bool:
        return cid in self.known_ids

    def save_ids(self):
        with open(self.ids_path, "w", encoding="utf-8") as f:
            json.dump(list(self.known_ids), f)

    def add(self, embeddings: np.ndarray, docs: List[Dict]):
        """
        embeddings: numpy array shape (N, dim)
        docs: list of metadata dicts length N, each must include 'id' and 'text' and 'source'
        """
        if embeddings.shape[0] != len(docs):
            raise ValueError("embeddings rows must match number of docs")

        # normalize for cosine via inner product
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        # add vectors
        self.index.add(embeddings)

        # append metadata lines
        with open(self.meta_path, "a", encoding="utf-8") as f:
            for meta in docs:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        self.metadata.extend(docs)

        # update ids and persist
        for meta in docs:
            self.known_ids.add(meta["id"])
        self.save_ids()
        faiss.write_index(self.index, self.index_path)

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Dict]:
        """
        query_vec: numpy 1D vector (dim,)
        returns list of dicts {text, source, chunk_id, score}
        """
        if self.index.ntotal == 0:
            return []

        q = query_vec.astype("float32").reshape(1, -1)
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, k)

        results = []
        for idx, score in zip(ids[0], scores[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({
                "text": meta["text"],
                "source": meta.get("source"),
                "chunk_id": meta.get("id"),
                "score": float(score)
            })
        return results
