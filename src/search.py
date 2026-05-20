import os
from typing import Optional, List, Dict, Any
import requests
from dotenv import load_dotenv
from src.dataloader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.paths import resolve_path

load_dotenv()

class RAGSearch:
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-120b:free",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout: int = 60,
    ):
        self.persist_dir = str(resolve_path(persist_dir, "data/faiss_store"))
        self.data_dir = str(resolve_path(data_dir, "data"))
        self.vectorstore = FaissVectorStore(self.persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            docs = load_all_documents(self.data_dir)
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        self.llm_model = llm_model
        self.base_url = base_url
        self.timeout = timeout
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required. Set it in your environment.")
        print(f"[INFO] OpenRouter LLM initialized: {llm_model}")

    def _openrouter_chat(self, prompt: str, max_tokens: int = 1024) -> str:
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def search_with_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        results = self.vectorstore.query(query, top_k=top_k)
        contexts: List[Dict[str, Any]] = []
        for result in results:
            metadata = result.get("metadata") or {}
            contexts.append(
                {
                    "text": metadata.get("text", ""),
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "distance": result.get("distance"),
                }
            )
        context_text = "\n\n".join([c["text"] for c in contexts if c["text"]])
        if not context_text:
            return {"summary": "No relevant documents found.", "results": contexts}
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context_text}\n\nSummary:"""
        summary = self._openrouter_chat(prompt)
        return {"summary": summary, "results": contexts}

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        response = self.search_with_context(query, top_k=top_k)
        return response["summary"]

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is EventX"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)