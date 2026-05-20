from src.dataloader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
import os

if __name__ == "__main__":
    docs = load_all_documents("data")
    store = FaissVectorStore()
    faiss_path = os.path.join(store.persist_dir, "faiss.index")
    meta_path = os.path.join(store.persist_dir, "metadata.pkl")
    if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
        store.build_from_documents(docs)
    else:
        store.load()
    rag_search = RAGSearch()
    query = "What are frontend and backend implementaton used in EventX?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
    
