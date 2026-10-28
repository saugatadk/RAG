import os
import faiss
import numpy as np
import pickle
from typing import List, Any, Optional
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline
from src.dataloader import load_all_documents
from src.paths import resolve_path

class FaissVectorStore:
    def __init__(self, persist_dir: Optional[str] = None, embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        persist_path = resolve_path(persist_dir, "data/faiss_store")
        self.persist_dir = str(persist_path)
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        if not documents:
            raise ValueError("No documents available to build the index.")
        self.index = None
        self.metadata = []
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source"),
                "page": chunk.metadata.get("page"),
            }
            for chunk in chunks
        ]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

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
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        # Ensure the query is L2-normalized so IP == cosine similarity
        query_embedding = np.ascontiguousarray(query_embedding, dtype="float32")
        faiss.normalize_L2(query_embedding)
        D, I = self.index.search(query_embedding, top_k)
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