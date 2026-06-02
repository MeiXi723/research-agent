import os
import json
import numpy as np
import faiss


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "documents"):
        os.makedirs(persist_dir, exist_ok=True)
        self.index_path = os.path.join(persist_dir, f"{collection_name}.index")
        self.data_path = os.path.join(persist_dir, f"{collection_name}.data.json")

        self.index: faiss.IndexFlatL2 | None = None
        self.records: list[dict] = []

        if os.path.exists(self.index_path) and os.path.exists(self.data_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.records = json.load(f)

    def add(self, ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]):
        vectors = np.array(embeddings).astype("float32")
        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)

        for i, chunk_id in enumerate(ids):
            self.records.append({
                "id": chunk_id,
                "text": texts[i],
                "embedding": embeddings[i],
                "document_id": metadatas[i].get("document_id", ""),
                "document_name": metadatas[i].get("document_name", ""),
                "chunk_index": metadatas[i].get("chunk_index", 0),
                "page": metadatas[i].get("page"),
            })
        self._save()

    def search(self, query_embedding: list[float], n_results: int = 3) -> dict:
        if self.index is None or self.index.ntotal == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query = np.array([query_embedding]).astype("float32")
        k = min(n_results, self.index.ntotal)
        distances, indices = self.index.search(query, k)

        ids, docs, metas = [], [], []
        for idx in indices[0]:
            if 0 <= idx < len(self.records):
                r = self.records[idx]
                ids.append(r["id"])
                docs.append(r["text"])
                metas.append({
                    "document_id": r["document_id"],
                    "document_name": r["document_name"],
                    "chunk_index": r["chunk_index"],
                    "page": r["page"],
                })

        return {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [distances[0].tolist()]
        }

    def delete_by_document_id(self, document_id: str):
        old_count = len(self.records)
        self.records = [r for r in self.records if r["document_id"] != document_id]
        if len(self.records) == old_count:
            return
        self._rebuild_index()
        self._save()

    def _rebuild_index(self):
        if not self.records:
            self.index = None
            return
        dim = len(self.records[0]["embedding"])
        self.index = faiss.IndexFlatL2(dim)
        vectors = np.array([r["embedding"] for r in self.records]).astype("float32")
        self.index.add(vectors)

    def _save(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    def close(self):
        self._save()
