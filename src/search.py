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
        vectorstore: Optional[FaissVectorStore] = None,
        persist_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: Optional[str] = None,
        timeout: int = 60,
    ):
        self.persist_dir = str(resolve_path(persist_dir, "data/faiss_store")) if persist_dir else None
        self.data_dir = str(resolve_path(data_dir, "data")) if data_dir else None
        self.vectorstore = vectorstore or FaissVectorStore(persist_dir=self.persist_dir, embedding_model=embedding_model)
        if self.persist_dir:
            self.vectorstore.load()

        self.timeout = timeout
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.llm_model = llm_model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

        if self.gemini_key:
            print(f"[INFO] Gemini LLM initialized: {self.llm_model}")
        else:
            print("[WARNING] GEMINI_API_KEY is not set. Add GEMINI_API_KEY to your environment or Space Secrets.")

    def _chat(self, prompt: str, max_tokens: int = 1024) -> str:
        api_key = os.getenv("GEMINI_API_KEY") or self.gemini_key
        if not api_key:
            return (
                "GEMINI_API_KEY is missing. "
                "Please configure GEMINI_API_KEY in your Hugging Face Space Settings > Secrets (or in your local .env file)."
            )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.llm_model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": max_tokens,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code == 429:
                return "⚠️ Gemini API rate limit reached (429: Too Many Requests). The free tier request quota was temporarily exceeded. Please wait a minute before querying again, or verify your quota at https://ai.google.dev/."
            if not response.ok:
                return f"⚠️ Gemini API request failed with status code {response.status_code}. Please verify your GEMINI_API_KEY and model settings."

            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
            return "No response generated."
        except requests.exceptions.Timeout:
            return "⚠️ Gemini API request timed out. Please try again."
        except requests.exceptions.RequestException:
            return "⚠️ Network error communicating with Gemini API. Please check your connection."

    def search_with_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        if self.vectorstore.index is None or getattr(self.vectorstore.index, "ntotal", 0) == 0:
            return {
                "summary": "No documents have been uploaded in this session yet. Please upload documents in the 'Upload & Manage Documents' tab first.",
                "results": [],
            }
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
            return {"summary": "No relevant context found in uploaded session documents.", "results": contexts}
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context_text}\n\nSummary:"""
        summary = self._chat(prompt)
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