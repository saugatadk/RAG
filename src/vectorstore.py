import os
import faiss
import numpy as np
import pickle
from typing import List, Any, Optional
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline, get_shared_embedding_model
from src.paths import resolve_path

class FaissVectorStore:
    def __init__(self, persist_dir: Optional[str] = None, embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = str(resolve_path(persist_dir, "data/faiss_store")) if persist_dir else None
        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = get_shared_embedding_model(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def add_documents(self, documents: List[Any]) -> int:
        if not documents:
            return 0
        emb_pipe = EmbeddingPipeline(model=self.model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        if not chunks:
            return 0
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source"),
                "page": chunk.metadata.get("page"),
            }
            for chunk in chunks
        ]
        self.add_embeddings(np.array(embeddings).astype("float32"), metadatas)
        if self.persist_dir:
            self.save()
        return len(chunks)

    def clear(self):
        self.index = None
        self.metadata = []

    def build_from_documents(self, documents: List[Any]):
        self.clear()
        self.add_documents(documents)

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        if embeddings.size == 0:
            raise ValueError("No embeddings provided for indexing.")
        # Normalize for cosine similarity (cosine == inner product on unit vectors)
        embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        if self.index is None:
            # IndexFlatIP + L2-normalized vectors => cosine similarity
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index (cosine).")

    def save(self):
        if not self.persist_dir or self.index is None:
            return
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        if not self.persist_dir:
            return
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(faiss_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None or getattr(self.index, "ntotal", 0) == 0:
            return []
        query_embedding = np.ascontiguousarray(query_embedding, dtype="float32")
        faiss.normalize_L2(query_embedding)
        k = min(int(top_k), self.index.ntotal)
        D, I = self.index.search(query_embedding, k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            meta = self.metadata[int(idx)] if int(idx) < len(self.metadata) else None
            # `score` is cosine similarity in [-1, 1]; higher = more similar.
            # Keep `distance` (= 1 - score) for backwards compatibility with callers.
            score_f = float(score)
            results.append({
                "index": int(idx),
                "score": score_f,
                "distance": 1.0 - score_f,
                "metadata": meta,
            })
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        if self.index is None or getattr(self.index, "ntotal", 0) == 0:
            print("[WARN] Vector store is empty.")
            return []
        query_emb = self.model.encode([query_text]).astype("float32")
        return self.search(query_emb, top_k=top_k)

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    store = FaissVectorStore()
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is EventX?", top_k=3)) 